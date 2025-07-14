from bert import BERT

import torch
from torch.nn import NLLLoss

from transformers import BertTokenizer, BertTokenizerFast
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
USE_COMPILE = False #True #True
DATASET_PATH = "/home/adam/python_dojo/bert/data/imdb_tokenized/"
DATASET_PATH = "/home/adam/python_dojo/bert/data/imdb_tokenized_tst/"

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, f"log.txt")
MODEL_DIR = "/home/adam/huggingface_models/" #cache for models

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

#parameters set as in the GPT-2/3 paper, WARMUP_STEPS/MAX_STEPS adjusted to iterate over the entire fineweb dataset
#learning rate scheduler settings
MAX_LR = 1e-4
# MAX_LR = 1e-3
# MAX_LR = 5e-05
MIN_LR = MAX_LR * 0.1 #not defined in the BERT paper, using same as GPT-2
WARMUP_STEPS = 1211 # 1211 == 39% , similar ratio as bert paper but for smaller imdb dataset #10000 #based on BERT paper
#learning steps settings
MAX_STEPS = 12204 #3051 #3051 = imdb dataset 1 epoch for all shards #25177 #1000 #10 #25177 # == total dataset size (3.3B token corpus) / total tokens per batch = 3.3e9 / 131072 = 25177 (in the paper its 40 epochs, we do 1 for start) #4
#data processing settings
ACCUM_BATCH_TOKEN_COUNT = 131072 # according to paper 256*512 batch size in tokens(batch size oO, "just a bit big" -> need to do gradient accumulation for this!)
#this is adjustable given hardware
MICRO_BATCH_SIZE = 128 #256 #64 #16
MICRO_BATCH_TOKEN_COUNT = 385 # BERT paper = 512
ACCUM_BATCH_SIZE = ACCUM_BATCH_TOKEN_COUNT // (MICRO_BATCH_SIZE * MICRO_BATCH_TOKEN_COUNT * DDP_WORLD_SIZE) #calculated given above
#evaluation
EVAL_EVERY_STEPS = 25 #500 #250
GEN_EVERY_STEPS = 250 #100
EVAL_HELLASWAG_EVERY_STEPS = 1000 


#hyper-parameters from BERT article
N_LAYERS = 12
EMBEDDING_SIZE = 768
N_ATTN_HEADS = 12
DROPOUT = 0.3

N_SEGMENTS = 3 #segments no for segment embedding
MAX_INPUT_LEN = 512
VOCABULARY_SIZE = 30000

#dataset params
SEQ_LENGTH=128
SAMPLE_LENGTH=SEQ_LENGTH+SEQ_LENGTH+SEQ_LENGTH+1 # =385 = "masked sentence" + "mask_labels" +"segment_mask" + is_next_label
MAX_SENTENCE_WORDS= SEQ_LENGTH//2 # Maximum number of words in one sentence (ie. seq len / 2 as we have max 2 sentences)

#-----------------------------------------------------------------------------------------------
#extended version of dataloader to cope with a bigger dataset split into multiple files
#work with DDP / distributed processing, splits dataset per node with offset the size of batch in tokens
# @token_count = total tokens per sample written in data file (input+input_segments + output_mask+output_next_label)
# @sequence_length= length of single sequence in batch (input/input_segements/output_mask)
class IMDBDataLoader():
    def __init__(self, batch_size, token_count, accum_batch_total_token_count, sequence_length, tokenizer, process_rank, num_processes, dataset_path ="", split="train"): #amount of samples in one batch , amount of tokens per sample
        self.batch_size = batch_size
        self.token_count = token_count
        self.sequence_length = sequence_length
        self.process_rank = process_rank
        self.num_processes = num_processes

        assert split in {'train', 'val'}

        shards = os.listdir(dataset_path) #list all data files
        shards = [s for s in shards if split in s] #filer out valid/train split
        #shards = sorted(shards)
        shards = [os.path.join(dataset_path, s) for s in shards] #add full path
        self.shards = shards

        if MASTER_PROCESS: #check if any data loaded
            assert len(shards) > 0, "Missing data for split: {}".format(split)

        self.reset_positions()

        if MASTER_PROCESS:
            print("Found {} files. Current file - loaded {} tokens (1 epoch= {} steps / {} microsteps / 1 microstep = {} tokens)".format(len(shards),len(self.tokens), len(self.tokens)//accum_batch_total_token_count, len(self.tokens) // (batch_size * token_count), self.batch_size * self.token_count))


    #data batches are some <token_count> portion of dataset; the targets is same size matrix as we provide prediction for every
    #token in the input (e.g. for input[0,0] next token is output[0,0]; for input[0:1,0] target is output[1,0] etc.)
    def next_batch(self):
        data_batch = self.tokens[self.current_position:self.current_position+self.batch_size * self.token_count] #+1 is to expand for ground truth of the last element (ie. next token)
        #TODO: fix later when creating shard to make bigger ones , 4xseq_lengh, labal_next= filled with zeroes, this is better to load data later
        #data_batch[].view(-1,4,SEQ_LEN)
        
        #reshape batch data to split it into samples
        samples = data_batch.view(self.batch_size, self.token_count) # samples with all input/output data serialized
        x = samples[:, :self.sequence_length] # inputs (2 sentences with masked)
        y = samples[:, self.sequence_length:2*self.sequence_length] # outputs (masked tokens)
        x_seg = samples[:, 2*self.sequence_length:3*self.sequence_length] # input for sentence segments
        y_next = samples[:, 3*self.sequence_length:3*self.sequence_length+1] # output for next sentences (true/false)
        y_next = torch.squeeze(y_next, dim=1)

        #temp fix, bad characters that cause crash (normally this should be done during dataset processing and writing to shards)
        #mask chars that cause crash (input) 1620 / 30147 == ASCII star (in star ratings)
        special_char_mask = x == 1620
        special_char_mask2 = x == 30147
        x[special_char_mask] = 2732
        x[special_char_mask2] = 2732
        #mask chars that cause crash (output)
        special_char_mask = y == 1620
        special_char_mask2 = y == 30147
        y[special_char_mask] = 2732
        y[special_char_mask2] = 2732

        #iterate to next batch
        self.current_position += self.batch_size * self.token_count * self.num_processes #move the pointer to next batch
        #if end of file / the dataset, go to next file / start all over again (+1 is due to prediction of next token for GT)
        if self.current_position+self.batch_size * self.token_count * self.num_processes > len(self.tokens): 
            self.current_shard = (self.current_shard + 1) % len(self.shards) #set current file to next one
            self.tokens = load_tokens(self.shards[self.current_shard]) #load the file
            self.current_position= self.batch_size * self.token_count * self.process_rank #reset to starting point for current node with its offset
        return x,x_seg,y,y_next

    #reset all points to start from the start of dataset
    def reset_positions(self):
        self.current_shard = 0 
        self.tokens = load_tokens(self.shards[self.current_shard]) #load data
        #state which sample is currently loaded
        #for each node in DDP this will be a differnt starting point, we condition it on node number
        self.current_position = self.batch_size * self.token_count * self.process_rank 

#read dataset file (earlier serialized into tokens with BertTokenizerFast and coverted to numpy array
def load_tokens(filename):
    tokens = np.load(filename)
    tokens = tokens.astype(np.int32)
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    return tokens_tensor

#-----------------------------------------------------------------------------------------------

def evaluation_step(model, val_loader, device, ddp, master_process):
    loss_fn = F.cross_entropy
    model.eval()
    val_loader.reset_positions()
    with torch.no_grad():
        validation_loss_accum = 0.0 #loss accumulated over all distrubted processes
        validation_acc_accum = 0.0
        #1batch = 64 samples = 64 * 385 = 24,640 tokens
        validation_steps = 100 #4058 #full eval set == 99999900 tokens = 259,740 samples 
        #iterate entire validation set
        # for micro_batch in range(ACCUM_BATCH_SIZE):
        for _ in range(validation_steps):
            x, x_seg, y, y_next = val_loader.next_batch()
            x, x_seg, y, y_next = x.to(device), x_seg.to(device), y.to(device), y_next.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits_missing_words, logits_next_stentece = model(x, x_seg)
            loss_val1 = loss_fn(logits_next_stentece, y_next) #data["is_next"]
            loss_val2 = loss_fn(logits_missing_words.transpose(1, 2), y,ignore_index=0) #data["bert_label"]
            loss = loss_val1 + loss_val2
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


def get_masked_accuracy(logits,y, topk=1):
    probs = F.softmax(logits, dim=-1) #change un-normalized logits into probabilities
    topk_probs, topk_indices= torch.topk(probs, topk, dim=-1)
    topk_indices = topk_indices.squeeze(dim=2)
    mask = y != 0
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
    print("------------------------------------------------------------------------------------\n\n")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased", cache_dir=MODEL_DIR)
# if OPTIMIZE_SPEED: #nice number, rather than odd default GPT2 vocab size
#     config.vocabulary_size=50304
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

#load data in microbatches that fit to our GPU
# train_loader = TinyShakrespearDataLoader(batch_size=MICRO_BATCH_SIZE, token_count=MICRO_BATCH_TOKEN_COUNT,tokenizer=tokenizer, process_rank=DDP_RANK, num_processes=DDP_WORLD_SIZE) 
train_loader = IMDBDataLoader(batch_size=MICRO_BATCH_SIZE
                                , token_count=MICRO_BATCH_TOKEN_COUNT
                                , accum_batch_total_token_count=ACCUM_BATCH_TOKEN_COUNT
                                , sequence_length=SEQ_LENGTH
                                , tokenizer=tokenizer
                                , process_rank=DDP_RANK
                                , num_processes=DDP_WORLD_SIZE
                                , dataset_path=DATASET_PATH
                                , split="train") 
val_loader =  IMDBDataLoader(batch_size=MICRO_BATCH_SIZE
                                , token_count=MICRO_BATCH_TOKEN_COUNT
                                , accum_batch_total_token_count=ACCUM_BATCH_TOKEN_COUNT
                                , sequence_length=SEQ_LENGTH
                                , tokenizer=tokenizer
                                , process_rank=DDP_RANK
                                , num_processes=DDP_WORLD_SIZE
                                , dataset_path=DATASET_PATH
                                , split="val") 


# [optional: optimalization] enable TF32 instead of float32 for faster processing
if OPTIMIZE_SPEED:
    torch.set_float32_matmul_precision('high')

# create the log directory we will write checkpoints to and log to
os.makedirs(LOG_DIR, exist_ok=True)
with open(LOG_FILE, "w") as f: # open for writing to clear the file
    pass

# TRAINING LOOP
#optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, betas=(0.9, 0.999), eps=1e-8,weight_decay=0.01) #optimizer paramas like BERT paper (somewhat)
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
for step_no in range(MAX_STEPS):
    # if step_no % 500 == 0:
    #     train_loader.reset_positions()
    st_time = time.time()
    last_step = (step_no == MAX_STEPS - 1)

    #do evaluation every EVAL_EVERY_STEPS steps
    if step_no % EVAL_EVERY_STEPS == 0 or last_step:
        validation_loss, validation_acc = evaluation_step(model, val_loader, device, IS_DDP, MASTER_PROCESS)

    # # do hellaswag evaluation every EVAL_HELLASWAG_EVERY_STEPS steps
    # if (step_no % EVAL_HELLASWAG_EVERY_STEPS == 0 or last_step):
    #     hellaswag_acc_norm, hellaswag_num_correct_norm, hellaswag_num_total = evaluation_hellaswag(model, tokenizer, device, IS_DDP, DDP_WORLD_SIZE,DDP_RANK)

    # # every GEN_EVERY_STEPS generate from model to see how its doing
    # if step_no > 0 and step_no % GEN_EVERY_STEPS == 0:
    #     generate_samples(model,tokenizer,device, DDP_RANK)

    model.train(True)
    optimizer.zero_grad()
    #gradient accumulation (calc multiple micro batches before doing gradient update)
    acc_accum = 0.0 #last batch acc
    loss_accum = 0.0 #log gradient accumulation loss 
    for micro_batch in range(ACCUM_BATCH_SIZE):
        x, x_seg, y, y_next = train_loader.next_batch()
        x, x_seg, y, y_next = x.to(device), x_seg.to(device), y.to(device), y_next.to(device)
        tokenizer2 = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased", cache_dir=MODEL_DIR)
        if IS_DDP: #for distrubted - disable sync loss across GPUs until final calculation of this loop
            model.require_backward_grad_sync = (micro_batch == ACCUM_BATCH_SIZE - 1) 
        if OPTIMIZE_SPEED:
            with torch.autocast(device_type=device, dtype=torch.bfloat16): #adding bfloat16 (insead of float32)
                # logits_missing_words, logits_missing_words_prob, logits_next_stentece, logits_next_stentece_prob = model(x, x_seg)
                logits_missing_words, logits_next_stentece = model(x, x_seg)
        else:
            logits_missing_words, logits_next_stentece = model(x, x_seg)
        loss_val1 = loss_fn(logits_next_stentece, y_next) #data["is_next"]
        loss_val2 = loss_fn(logits_missing_words.transpose(1, 2), y,ignore_index=0) #data["bert_label"]
        acc = get_masked_accuracy(logits_missing_words,y) / ACCUM_BATCH_SIZE
        loss = loss_val1 + loss_val2
        loss = loss / ACCUM_BATCH_SIZE #this accounts for the fact that we loss in a mean and by adding multiple losses we dont simulate doing mean over all of samples in one go
        acc_accum += acc.detach()
        loss_accum += loss.detach() #logging purpose only
        loss.backward()

    if IS_DDP:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) #for logging only , average out logged loss across all GPUs

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #added gradient norm clipping (from GPT3 paper). apparently this prevents the model to get big shocks from bad/odd batches

    # determine and set the learning rate for this iteration (from GPT3 paper)
    #temp remove
    lr = get_lr(step_no)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize() # wait for the GPU to finish work (done for time measurement reasons only)
    ed_time = time.time()
    batch_time = (ed_time - st_time) # time difference
    tokens_per_sec = (train_loader.batch_size * train_loader.token_count * ACCUM_BATCH_SIZE * DDP_WORLD_SIZE) / batch_time

    if MASTER_PROCESS:
        eval_log = ""
        hellaswag_eval_log = ""
        if step_no % EVAL_EVERY_STEPS == 0 or last_step: #log evaluation loss every 100 steps
            eval_log = ", val_loss: {:.4f}, val_acc: {:.4f}".format(validation_loss,validation_acc)
            with open(LOG_FILE, "a") as f:
                f.write(f"{step_no} val {validation_loss:.4f}\n")
        # if step_no % EVAL_HELLASWAG_EVERY_STEPS == 0: #log evaluation loss every 100 steps
        #     hellaswag_eval_log = ", hellaswag_acc: {:.4f}".format(hellaswag_acc_norm)
        #     with open(LOG_FILE, "a") as f:
        #         f.write(f"{step_no} hella {hellaswag_acc_norm:.4f}\n")
        # print("Step: {:5d}, loss: {:.6f}, norm: {:.4f}, time(ms): {:.2f}, token/sec:{:.2f}{}{}".format(step_no, loss_accum.item(), norm, batch_time*1000, tokens_per_sec,eval_log,hellaswag_eval_log))
        # if step_no % 1000 == 0:
            # print(logits_missing_words.shape, logits_next_stentece.shape)
            # probs = logits_missing_words[0,1,:]
            # topk_probs, topk_indices= torch.topk(probs, 50, dim=-1)
            # print(topk_probs, topk_indices)
            # print("----------------------")
            # probs = logits_missing_words[0,2,:]
            # topk_probs, topk_indices= torch.topk(probs, 50, dim=-1)
            # print(topk_probs, topk_indices)
            # print("----------------------")
            # probs = logits_missing_words[0,3,:]
            # topk_probs, topk_indices= torch.topk(probs, 50, dim=-1)
            # print(topk_probs, topk_indices)
            # ref_decoded = tokenizer.decode(x[0])
            # print("ref: {}".format(ref_decoded))
            # decoded_output = format_prediction(logits_missing_words[0],logits_next_stentece[0],tokenizer = tokenizer)
            # print("pred: {}".format(decoded_output))

        # if step_no % 50 == 0:
        #     print("Step: {:5d}, batch_acc: {}, loss: {:.6f}, time(ms): {:.2f}, token/sec:{:.2f}{}{}".format(step_no, acc_accum,loss_accum.item(), batch_time*1000, tokens_per_sec,eval_log,hellaswag_eval_log))
        print("Step: {:5d}, batch_acc: {:.6f}, loss: {:.6f}, lr: {:.5e}, time(ms): {:.2f}, token/sec:{:.2f}{}{}".format(step_no, acc_accum,loss_accum.item(),lr, batch_time*1000, tokens_per_sec,eval_log,hellaswag_eval_log))


if IS_DDP:
    destroy_process_group()