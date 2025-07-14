#testing script to see how BERT trained by HuggingFace performs and how some I/O works

from transformers import BertTokenizer
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
from datasets import load_dataset

from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import torch
import torch.nn.functional as F


from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import time
import os

import pandas as pd

HUGGINGFACE_DATA_CACHE = "/media/sda/adam/huggingface_data_cache/"
MODEL_DIR = "/media/sda/adam/huggingface_models/" #cache for models

# ---------------------------------------------------


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
# ---------------------------------------------------

USE_COMPILE = False
OPTIMIZE_SPEED = True

HUGGINGFACE_DATA_CACHE = "/media/sda/adam/huggingface_data_cache/"
MODEL_DIR = "/media/sda/adam/huggingface_models/" #cache for models

BATCH_SIZE = 256
VAL_DATASET_SIZE= 16000
TRAIN_DATASET_SIZE = 74004228-VAL_DATASET_SIZE #20000 #book corpus samples count #16000 = eval size (1000*batch(16) )

MAX_STEPS = TRAIN_DATASET_SIZE//BATCH_SIZE #289016
MAX_EPOCHS = 1
MICRO_BATCH_SIZE = 64 #16 #16 #256 #64 #16
TOKENS_PER_SAMPLE = 512
GRAD_ACCUM = True
GRAD_ACCUM_STEPS = BATCH_SIZE// MICRO_BATCH_SIZE #21

EVAL_EVERY_STEPS = 500 #500 #10

SHARD_SIZE = 25600

print("---------------------------------------------")
print("BATCH_SIZE:          {}".format(BATCH_SIZE))
print("TRAIN_DATASET_SIZE:  {}".format(TRAIN_DATASET_SIZE))
print("VAL_DATASET_SIZE:    {}".format(VAL_DATASET_SIZE))
print("---------------------------------------------")
print("MAX_EPOCHS:          {}".format(MAX_EPOCHS))
print("MAX_STEPS:           {}".format(MAX_STEPS))
print("---------------------------------------------")
print("MICRO_BATCH_SIZE:    {}".format(MICRO_BATCH_SIZE))
print("TOKENS_PER_SAMPLE:   {}".format(TOKENS_PER_SAMPLE))
print("GRAD_ACCUM:          {}".format(GRAD_ACCUM))
print("GRAD_ACCUM_STEPS:    {}".format(GRAD_ACCUM_STEPS))
print("---------------------------------------------")
print("DISTRUBTED:          {}".format(IS_DDP))
print("OPTIMIZE_SPEED:      {}".format(OPTIMIZE_SPEED))
print("USE_COMPILE:         {}".format(USE_COMPILE))
print("---------------------------------------------")

#same validation script as used for BERT from scratch
def evaluation_step(model, val_loader, device, ddp, master_process):
    loss_fn = F.cross_entropy
    model.eval()
    data_iter = iter(val_loader)
    with torch.no_grad():
        validation_loss_accum = 0.0 #loss accumulated over all distrubted processes
        validation_acc_accum = 0.0
        validation_steps = VAL_DATASET_SIZE // MICRO_BATCH_SIZE #10 #00 #00 #4058 #full eval set == 99999900 tokens = 259,740 samples 
        for _ in range(validation_steps):
            data_batch = next(data_iter)
            x = data_batch['input_ids'].to(device)
            y = data_batch['labels'].to(device)
            if OPTIMIZE_SPEED:
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits_missing_words = model(x)
            else:
                logits_missing_words = model(x)
            # loss_val1 = loss_fn(logits_next_stentece, y_next) #TEMP remove next sentence prediction
            loss_val2 = loss_fn(logits_missing_words.logits.transpose(1, 2), y,ignore_index=-100) #data["bert_label"]
            loss = loss_val2 #loss_val1 +  #TEMP remove next sentence prediction
            loss = loss / validation_steps
            acc = get_masked_accuracy(logits_missing_words.logits,y) / validation_steps
            validation_loss_accum += loss.detach()
            validation_acc_accum += acc.detach()
    if ddp:
        dist.all_reduce(validation_loss_accum, op=dist.ReduceOp.AVG)
    return validation_loss_accum.item(), validation_acc_accum.item()

#logits_missing_words[0,1,:]
#@sample - if sample on random from top, or just pick most likely
def format_prediction(logits, tokenizer, sample=False, topk =1):
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
    mask = y != -100
    # print(y[mask])
    # print(tokenizer.decode(y[mask]))
    if len(y[mask])==0:
        return 1
    acc = sum(y[mask] == topk_indices[mask]) / len(y[mask])
    return acc

def tokenize_function(examples,tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=TOKENS_PER_SAMPLE)

def next_shard_iterator(shard_start,shard_end,tokenizer):
    dataset = load_dataset("bookcorpus/bookcorpus", cache_dir=HUGGINGFACE_DATA_CACHE, split="train[{}:{}]".format(shard_start,shard_end)) #[:20000] # full = 74,004,228
    tokenized_dataset = dataset.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})

    # print(tokenized_dataset)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    data_loader = torch.utils.data.DataLoader(tokenized_dataset['input_ids'], batch_size=MICRO_BATCH_SIZE, collate_fn=data_collator)
    data_iter = iter(data_loader)
    return data_iter

config = BertConfig(
    vocab_size=50000,
    hidden_size=768, 
    num_hidden_layers=6, 
    num_attention_heads=12,
    max_position_embeddings=512,
    cache_dir=MODEL_DIR
)
 
model = BertForMaskedLM.from_pretrained('bert-base-uncased', cache_dir=MODEL_DIR ).to(device) #, torchscript=True)
print('No of parameters: ', model.num_parameters())

raw_model = model
if IS_DDP: #wrapper around model for DDP to keep things in sync
    model = DDP(model, device_ids=[DDP_LOCAL_RANK])
    raw_model = model.module

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", cache_dir=MODEL_DIR)

#load data (same as original script, just added an extra loader to quickly test attention)
shard_start = 16000+DDP_RANK*SHARD_SIZE
shard_size= SHARD_SIZE #499,968
shard_end = shard_start+shard_size

print("PROCESS: {}, shard_start: {} - shard_end: {}".format(DDP_RANK,shard_start,shard_end))
dataset = load_dataset("bookcorpus/bookcorpus", cache_dir=HUGGINGFACE_DATA_CACHE, split="train[{}:{}]".format(shard_start,shard_end))
val_dataset = load_dataset("bookcorpus/bookcorpus", cache_dir=HUGGINGFACE_DATA_CACHE, split="train[0:{}]".format(VAL_DATASET_SIZE))
tokenized_dataset = dataset.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

val_data_loader = torch.utils.data.DataLoader(tokenized_val_dataset['input_ids'], batch_size=MICRO_BATCH_SIZE, collate_fn=data_collator)
data_loader = torch.utils.data.DataLoader(tokenized_dataset['input_ids'], batch_size=MICRO_BATCH_SIZE, collate_fn=data_collator)
data_loader_attn = torch.utils.data.DataLoader(tokenized_dataset['attention_mask'], batch_size=MICRO_BATCH_SIZE)
data_iter = iter(data_loader)
data_iter_attn = iter(data_loader_attn)

# ========= test outputs for single sample from batch ========
sample_no = 4 #sample number in batch

model.eval()
data_batch = next(data_iter)
data_batch_attn = next(data_iter_attn)

#print unencoded text for first batch
print("\n------- FIRST BATCH -------") 
print(dataset[:64])
print("---------------------------")
print("---------- pred with attn mask -----------") 
x = data_batch['input_ids'].to(device)
y = data_batch['labels'].to(device)
logits_missing_words = model(x,torch.Tensor(tokenized_dataset['attention_mask'][:64]).to(device))
acc = get_masked_accuracy(logits_missing_words.logits,y)
res = format_prediction(logits=logits_missing_words.logits[sample_no], tokenizer=tokenizer)

print("batch [MASKED] prediction accuracy (with attn mask): {}".format(acc.detach()))
print("-----------------------") 

print("Prediction result: {}".format(res))
ref_decoded = tokenizer.decode(x[sample_no])
y_decoded = tokenizer.decode(y[sample_no])
print("Reference: {} / {}".format(ref_decoded,y_decoded))

#v2 - without attn mask
print("----- pred no attn mask ---------")
logits_missing_words = model(x)
acc = get_masked_accuracy(logits_missing_words.logits,y)
res = format_prediction(logits=logits_missing_words.logits[sample_no], tokenizer=tokenizer)
print("batch [MASKED] prediction accuracy (no attn mask): {}".format(acc.detach()))
print("Prediction result (no attn mask): {}".format(res))
print("-----------------------")

validation_loss, validation_acc = evaluation_step(model, val_data_loader, device, IS_DDP, MASTER_PROCESS)
print("Validation set, loss: {}, accuracy: {}".format(validation_loss, validation_acc))


