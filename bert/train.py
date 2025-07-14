from bert import BERT

import torch
from torch.nn import NLLLoss

from transformers import BertTokenizer, BertTokenizerFast
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import numpy as np
import math

import os
import inspect
import time

OPTIMIZE_SPEED = True
USE_COMPILE = True #True #True
DATASET_PATH = "/home/adam/python_dojo/bert/data/imdb_tokenized/"
DATASET_PATH = "/mnt/data/backup/adam/llm_train_data/imdb_tokenized_tst/"

VAL_DATASET_SIZE= 16000
DATASET_SIZE = 74004228-VAL_DATASET_SIZE #20000 #book corpus samples count #16000 = eval size (1000*batch(16) )

SHARD_SIZE = 25600 #499968 #2560 #= 256*1953 ~500k shards #1000000

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, f"log.txt")
MODEL_DIR = "/mnt/ssd/home/adam/huggingface_models/" #cache for models
HUGGINGFACE_DATA_CACHE = "/mnt/data/backup/adam/huggingface_data_cache/"

#init distributed processing
# simple launch:
# python train.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=2 train.py

IS_DDP = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

if IS_DDP:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "Should not run DDP without CUDA..."
    init_process_group(backend='nccl')
    DDP_RANK = int(os.environ['RANK'])
    DDP_LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    DDP_WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{DDP_LOCAL_RANK}'
    torch.cuda.set_device(device)
    MASTER_PROCESS = DDP_RANK == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    DDP_RANK = 0
    DDP_LOCAL_RANK = 0
    DDP_WORLD_SIZE = 1
    MASTER_PROCESS = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print("Using device: {}".format(device))

#training parameters
#74004228-16000 = 73,988,228
#parameters set as in the BERT paper, WARMUP_STEPS/MAX_STEPS adjusted to iterate over the entire book corpus dataset
#learning rate scheduler settings
MAX_LR = 1e-4
# MAX_LR = 1e-3
# MAX_LR = 5e-05
MIN_LR = MAX_LR * 0.1 #not defined in the BERT paper, using same as GPT-2
WARMUP_STEPS = 2890 #=0.01*MAX_STEPS #1211 # 1211 == 39% , similar ratio as bert paper but for smaller imdb dataset #10000 #based on BERT paper

#learning steps settings
MAX_STEPS = 289016 #289079 # 289079 = 74,004,228-eval_size(16000) / 256 , ie. bookcropus sample size/ batch size #12204 #3051 #3051 = imdb dataset 1 epoch for all shards #25177 #1000 #10 #25177 # == total dataset size (3.3B token corpus) / total tokens per batch = 3.3e9 / 131072 = 25177 (in the paper its 40 epochs, we do 1 for start) #4
#data processing settings
ACCUM_BATCH_TOKEN_COUNT = 131072 # according to paper 256*512 batch size in tokens(batch size oO, "just a bit big" -> need to do gradient accumulation for this!)
#this is adjustable given hardware
MICRO_BATCH_SIZE = 128 #16 #128 #16 #128 #256 #64 #16
MICRO_BATCH_TOKEN_COUNT = 512 # BERT paper = 512
ACCUM_BATCH_SIZE = 256// (MICRO_BATCH_SIZE* DDP_WORLD_SIZE)  # as per bert paper total batch size should be 256 #ACCUM_BATCH_TOKEN_COUNT // (MICRO_BATCH_SIZE * MICRO_BATCH_TOKEN_COUNT * DDP_WORLD_SIZE) #calculated given above
#evaluation
EVAL_EVERY_STEPS = 500 #25 #500 #250
GEN_EVERY_STEPS = 250 #100
EVAL_HELLASWAG_EVERY_STEPS = 1000 


#hyper-parameters from BERT article
N_LAYERS = 12
EMBEDDING_SIZE = 768
N_ATTN_HEADS = 12
DROPOUT = 0.1

N_SEGMENTS = 3 #segments no for segment embedding
MAX_INPUT_LEN = 512
VOCABULARY_SIZE = 30000


#-----------------------------------------------------------------------------------------------
def tokenize_function(examples,tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MICRO_BATCH_TOKEN_COUNT)

def next_shard_iterator(shard_start,shard_end,tokenizer):
    dataset = load_dataset("bookcorpus/bookcorpus", cache_dir=HUGGINGFACE_DATA_CACHE, split="train[{}:{}]".format(shard_start,shard_end)) #[:20000] # full = 74,004,228
    tokenized_dataset = dataset.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    data_loader = torch.utils.data.DataLoader(tokenized_dataset['input_ids'], batch_size=MICRO_BATCH_SIZE, collate_fn=data_collator)
    data_iter = iter(data_loader)
    return data_iter

#-----------------------------------------------------------------------------------------------

def evaluation_step(model, val_loader, device, ddp, master_process):
    loss_fn = F.cross_entropy
    model.eval()
    data_iter = iter(val_loader)
    with torch.no_grad():
        validation_loss_accum = 0.0 #loss accumulated over all distrubted processes
        validation_acc_accum = 0.0
        validation_steps = VAL_DATASET_SIZE // MICRO_BATCH_SIZE
        #iterate entire validation set
        for _ in range(validation_steps):
            data_batch = next(data_iter)
            x = data_batch['input_ids'].to(device)
            y = data_batch['labels'].to(device)

            # replace tokenizer errors with [UNK]
            special_char_mask = x >= 30000 #it seems sometimes tokenier can return values outside of 30k vocab if its some odd character like chinese
            x[special_char_mask] = 100
            special_char_mask = y >= 30000 #it seems sometimes tokenier can return values outside of 30k vocab if its some odd character like chinese
            y[special_char_mask] = 100

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits_missing_words = model(x)
            # loss_val1 = loss_fn(logits_next_stentece, y_next) #TEMP remove next sentence prediction
            loss_val2 = loss_fn(logits_missing_words.transpose(1, 2), y,ignore_index=-100) #data["bert_label"]
            loss = loss_val2 #loss_val1 +  #TEMP remove next sentence prediction
            loss = loss / validation_steps
            acc = get_masked_accuracy(logits_missing_words,y) / validation_steps
            validation_loss_accum += loss.detach()
            validation_acc_accum += acc.detach()
    if ddp:
        dist.all_reduce(validation_loss_accum, op=dist.ReduceOp.AVG)
    return validation_loss_accum.item(), validation_acc_accum.item()

#scheduled Learning rate: 
# - at start tiny fraction of MAX_LR
# - climb up until WARMUP STEPS
# - after reaching max, go down again using cosine rate until reaching min rate
#deremine learning rate during training based on step number (from GPT3 paper)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < WARMUP_STEPS:
        return MAX_LR * (it+1) / WARMUP_STEPS
    # 2) if it > lr_decay_iters, return min learning rate
    if it > MAX_STEPS:
        return MIN_LR
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return MIN_LR + coeff * (MAX_LR - MIN_LR)

#custom init optmizer (following GPT3 indications, and some extra stuff recommended by Karapathy)
#most of the logic is selectivly setting parameters that will have weight decay
def configure_optimizers(model, weight_decay, learning_rate):
    # start with all of the candidate parameters (that require grad)
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    if MASTER_PROCESS:
        print("\nWEIGHT DECAY CONFIG:")
        print("* Number of tensors with weight decay / without weight decay: {} ({} params)/ {} ({} params)".format(len(decay_params), num_decay_params, len(nodecay_params), num_nodecay_params) )
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available #and 'cuda' in device
    if MASTER_PROCESS:
        print("* Using fused AdamW: {}\n".format(use_fused))
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, fused=use_fused)
    return optimizer

#-----------------------------------------------------------------------------------------------

def get_prediction(model, x, x_seg):
    model.eval()
    if OPTIMIZE_SPEED:
        with torch.autocast(device_type=device, dtype=torch.bfloat16): #adding bfloat16 (insead of float32)
            logits_missing_words, logits_next_stentece = model(x, x_seg)
    else:
        logits_missing_words, logits_next_stentece = model(x, x_seg)
    return 

#logits_missing_words[0,1,:]
#@sample - if sample on random from top, or just pick most likely
def format_prediction(logits, logits_next, tokenizer, sample=False, topk =1):
    probs = F.softmax(logits, dim=-1) #change un-normalized logits into probabilities
    topk_probs, topk_indices= torch.topk(probs, topk, dim=-1)
    if sample==True:
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)
        ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
    else:
        #topk_indices
        decoded = tokenizer.decode(torch.squeeze(topk_indices, dim=1))
    return decoded


def get_masked_accuracy(logits,y, topk=1): #HF pre-trained bert on bookcorpus has about ~0.5+ acc
    probs = F.softmax(logits, dim=-1) #change un-normalized logits into probabilities
    topk_probs, topk_indices= torch.topk(probs, topk, dim=-1)
    topk_indices = topk_indices.squeeze(dim=2)
    mask = y != -100
    if len(y[mask])==0: #in the event nothing was masked
        return 1
    acc = sum(y[mask] == topk_indices[mask]) / len(y[mask])
    return acc

#-----------------------------------------------------------------------------------------------


#print current config
if MASTER_PROCESS:
    print("\nBERT training configuration")
    print("------------------------------------------------------------------------------------")
    print("MAX_LR:          {}".format(MAX_LR))
    print("MIN_LR:          {}".format(MIN_LR))
    print("WARMUP_STEPS:    {}".format(WARMUP_STEPS))
    print("------------------------------------------------------------------------------------")
    print("MAX_STEPS:       {}".format(MAX_STEPS))
    print("------------------------------------------------------------------------------------")
    print("MICRO_BATCH_SIZE (samples in a batch):           {}".format(MICRO_BATCH_SIZE))
    print("MICRO_BATCH_TOKEN_COUNT (total tokens in batch): {}".format(MICRO_BATCH_TOKEN_COUNT))
    print("\n[Gradient Acculumation to match GPT2 batch sizes]")
    print("ACCUM_BATCH_SIZE:                                {}".format(ACCUM_BATCH_SIZE))
    print("ACCUM_BATCH_TOKEN_COUNT:                         {}".format(ACCUM_BATCH_TOKEN_COUNT))
    print("\nEVAL_EVERY_STEPS:                              {}".format(EVAL_EVERY_STEPS))
    print("EVAL_HELLASWAG_EVERY_STEPS:                      {}".format(EVAL_HELLASWAG_EVERY_STEPS))
    print("GEN_EVERY_STEPS:                                 {}".format(GEN_EVERY_STEPS))
    print("------------------------------------------------------------------------------------")
    print("DATASET_PATH:    {}".format(DATASET_PATH))
    print("------------------------------------------------------------------------------------")
    print("DISTRUBTED:      {}".format(IS_DDP))
    print("OPTIMIZE_SPEED:  {}".format(OPTIMIZE_SPEED))
    print("USE_COMPILE:     {}".format(USE_COMPILE))
    print("------------------------------------------------------------------------------------")
    print("DDP_RANK:        {}".format(DDP_RANK))
    print("DDP_LOCAL_RANK:  {}".format(DDP_LOCAL_RANK))
    print("DDP_WORLD_SIZE:  {}".format(DDP_WORLD_SIZE))
    print("------------------------------------------------------------------------------------\n\n")


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased", cache_dir=MODEL_DIR)
model = BERT(embedding_size=EMBEDDING_SIZE
            , heads_no=N_ATTN_HEADS
            , vocabulary_size=VOCABULARY_SIZE
            , max_input_length=MAX_INPUT_LEN
            , segment_no=N_SEGMENTS
            , dropout=DROPOUT
            , device=device)
model.to(device)

if OPTIMIZE_SPEED & USE_COMPILE: # torch.compile interferes with Generation.
    model = torch.compile(model)

raw_model = model
if IS_DDP: #wrapper around model for DDP to keep things in sync
    model = DDP(model, device_ids=[DDP_LOCAL_RANK])
    raw_model = model.module

#------ hf dataset
shard_start = VAL_DATASET_SIZE+DDP_RANK*SHARD_SIZE
shard_size= SHARD_SIZE #499,968
shard_end = shard_start+shard_size

print("DDP_RANK: {}, shard_start:{}, shard_end:{}".format(DDP_RANK, shard_start, shard_end))
dataset = load_dataset("bookcorpus/bookcorpus", cache_dir=HUGGINGFACE_DATA_CACHE, split="train[{}:{}]".format(shard_start,shard_end)) #[:20000] # full = 74,004,228
val_dataset = load_dataset("bookcorpus/bookcorpus", cache_dir=HUGGINGFACE_DATA_CACHE, split="train[0:{}]".format(VAL_DATASET_SIZE))
tokenized_dataset = dataset.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
data_loader = torch.utils.data.DataLoader(tokenized_dataset['input_ids'], batch_size=MICRO_BATCH_SIZE, collate_fn=data_collator)
val_data_loader = torch.utils.data.DataLoader(tokenized_val_dataset['input_ids'], batch_size=MICRO_BATCH_SIZE, collate_fn=data_collator)
data_iter = iter(data_loader)

# [optional: optimalization] enable TF32 instead of float32 for faster processing
if OPTIMIZE_SPEED:
    torch.set_float32_matmul_precision('high')

# create the log directory we will write checkpoints to and log to
os.makedirs(LOG_DIR, exist_ok=True)
with open(LOG_FILE, "w") as f: # open for writing to clear the file
    pass

# TRAINING LOOP
# optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, betas=(0.9, 0.999), eps=1e-8,weight_decay=0.01) #optimizer paramas like BERT paper (somewhat)
optimizer = configure_optimizers(raw_model, weight_decay=0.01, learning_rate=MAX_LR)
# loss = paper doesnt say much, HF implementation uses CrossEntropy
# loss = [...] Then, Ti will be used to predict the original token with cross entropy loss. [...]
# loss = [...] The training loss is the sum of the mean masked LM [laguage model] likelihood and the mean next sentence prediction likelihood. [...]
# loss_fn = CrossEntropyLoss(ignore_index=0) #NLLLoss(ignore_index=0)
loss_fn = F.cross_entropy
validation_loss = 0
validation_acc = 0
hellaswag_acc_norm = 0
lr = 0
micro_step_no=0
for step_no in range(MAX_STEPS):
    st_time = time.time()
    last_step = (step_no == MAX_STEPS - 1)

    #do evaluation every EVAL_EVERY_STEPS steps
    if step_no % EVAL_EVERY_STEPS == 0 or last_step:
        validation_loss, validation_acc = evaluation_step(model, val_data_loader, device, IS_DDP, MASTER_PROCESS)
    # validation_loss, validation_acc = 0,0

    model.train(True)
    optimizer.zero_grad()
    #gradient accumulation (calc multiple micro batches before doing gradient update)
    acc_accum = 0.0 #last batch acc
    loss_accum = 0.0 #log gradient accumulation loss 
    for micro_batch in range(ACCUM_BATCH_SIZE):
        try:
            data_batch = next(data_iter)
        except StopIteration:
            #if no more data in the shard, try to get next shard
            if (shard_start + DDP_WORLD_SIZE*shard_size) > DATASET_SIZE:
                print("END OF DATA!")
                if IS_DDP:
                    destroy_process_group()
                exit()
            else:
                shard_start+=DDP_WORLD_SIZE*shard_size
                if (shard_end + shard_size) > DATASET_SIZE:
                    shard_end=DATASET_SIZE
                else:
                    shard_end=shard_start+shard_size
                data_iter = next_shard_iterator(shard_start,shard_end,tokenizer)
                print("[INFO][DDP_RANK:{}] STARTING NEW SHARD: {} - {}".format(DDP_RANK,shard_start,shard_end))
                data_batch = next(data_iter)
        x = data_batch['input_ids'].to(device)
        y = data_batch['labels'].to(device)        
        # replace tokenizer errors with [UNK]
        special_char_mask = x >= 30000 #it seems sometimes tokenier can return values outside of 30k vocab if its some odd character like chinese
        x[special_char_mask] = 100
        special_char_mask = y >= 30000 #it seems sometimes tokenier can return values outside of 30k vocab if its some odd character like chinese
        y[special_char_mask] = 100

        if IS_DDP: #for distrubted - disable sync loss across GPUs until final calculation of this loop
            model.require_backward_grad_sync = (micro_batch == ACCUM_BATCH_SIZE - 1) 
        if OPTIMIZE_SPEED:
            with torch.autocast(device_type=device, dtype=torch.bfloat16): #adding bfloat16 (insead of float32)
                # logits_missing_words, logits_next_stentece, = model(x, x_seg) #TEMP remove Next Sentence Prediction
                logits_missing_words = model(x)
        else:
            logits_missing_words = model(x)
        # loss_val1 = loss_fn(logits_next_stentece, y_next) #data["is_next"]  #TEMP remove Next Sentence Prediction
        loss_val2 = loss_fn(logits_missing_words.transpose(1, 2), y,ignore_index=-100) #data["bert_label"]
        acc = get_masked_accuracy(logits_missing_words,y) / ACCUM_BATCH_SIZE
        loss = loss_val2  # loss_val1 +  #TEMP remove Next Sentence Prediction
        loss = loss / ACCUM_BATCH_SIZE #this accounts for the fact that we loss in a mean and by adding multiple losses we dont simulate doing mean over all of samples in one go
        acc_accum += acc.detach() # acc_accum += 0
        loss_accum += loss.detach() #logging purpose only
        loss.backward()
        micro_step_no+=1

    if IS_DDP:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) #for logging only , average out logged loss across all GPUs

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #added gradient norm clipping (from GPT3 paper). apparently this prevents the model to get big shocks from bad/odd batches

    # determine and set the learning rate for this iteration (from GPT3 paper)
    lr = get_lr(step_no)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    lr = optimizer.param_groups[0]['lr']

    optimizer.step()
    torch.cuda.synchronize() # wait for the GPU to finish work (done for time measurement reasons only)
    ed_time = time.time()
    batch_time = (ed_time - st_time)
    tokens_per_sec = (MICRO_BATCH_SIZE * MICRO_BATCH_TOKEN_COUNT * ACCUM_BATCH_SIZE * DDP_WORLD_SIZE) / batch_time

    if MASTER_PROCESS:
        eval_log = ""
        hellaswag_eval_log = ""
        if step_no % EVAL_EVERY_STEPS == 0 or last_step: #log evaluation loss every 100 steps
            eval_log = ", val_loss: {:.4f}, val_acc: {:.4f}".format(validation_loss,validation_acc)
            with open(LOG_FILE, "a") as f:
                f.write(f"{step_no} val {validation_loss:.4f}\n")
        print("Step: {:5d}, MicroStep: {:5d}, batch_acc: {:.6f}, loss: {:.6f}, lr: {:.5e}, time(ms): {:.2f}, token/sec:{:.2f}{}{}".format(step_no, micro_step_no,acc_accum,loss_accum.item(),lr, batch_time*1000, tokens_per_sec,eval_log,hellaswag_eval_log))


if IS_DDP:
    destroy_process_group()