#script replicates training my own model, same training logic but this one uses HuggingFace model instead of model written by myself.

from transformers import BertTokenizer
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import torch
import torch.nn.functional as F
from datasets import load_dataset

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import time
import os

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

# ----------------------------

from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling

config = BertConfig(
    vocab_size=50000,
    hidden_size=768, 
    num_hidden_layers=6, 
    num_attention_heads=12,
    max_position_embeddings=512,
    cache_dir=MODEL_DIR
)
 
model = BertForMaskedLM(config).to(device)
print('No of parameters: ', model.num_parameters())

raw_model = model
if IS_DDP: #wrapper around model for DDP to keep things in sync
    model = DDP(model, device_ids=[DDP_LOCAL_RANK])
    raw_model = model.module


def evaluation_step(model, val_loader, device, ddp, master_process):
    loss_fn = F.cross_entropy
    model.eval()
    data_iter = iter(val_loader)
    with torch.no_grad():
        validation_loss_accum = 0.0 #loss accumulated over all distrubted processes
        validation_acc_accum = 0.0
        #1batch = 64 samples = 64 * 385 = 24,640 tokens
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

def get_masked_accuracy(logits,y, topk=1):
    probs = F.softmax(logits, dim=-1) #change un-normalized logits into probabilities
    topk_probs, topk_indices= torch.topk(probs, topk, dim=-1)
    topk_indices = topk_indices.squeeze(dim=2)
    mask = y != -100
    if len(y[mask])==0: #in the event nothing was masked
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

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", cache_dir=MODEL_DIR)

shard_start = 16000+DDP_RANK*SHARD_SIZE
shard_size= SHARD_SIZE #499,968
shard_end = shard_start+shard_size

print("PROCESS: {}, shard_start: {} - shard_end: {}".format(DDP_RANK,shard_start,shard_end))
dataset = load_dataset("bookcorpus/bookcorpus", cache_dir=HUGGINGFACE_DATA_CACHE, split="train[{}:{}]".format(shard_start,shard_end))
val_dataset = load_dataset("bookcorpus/bookcorpus", cache_dir=HUGGINGFACE_DATA_CACHE, split="train[0:{}]".format(VAL_DATASET_SIZE))
tokenized_dataset = dataset.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

data_loader = torch.utils.data.DataLoader(tokenized_dataset['input_ids'], batch_size=MICRO_BATCH_SIZE, collate_fn=data_collator)
val_data_loader = torch.utils.data.DataLoader(tokenized_val_dataset['input_ids'], batch_size=MICRO_BATCH_SIZE, collate_fn=data_collator)
data_iter = iter(data_loader)

# -------------------------------
optimizer = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8,weight_decay=0.01)
loss_fn = CrossEntropyLoss()
loss_fn = F.cross_entropy
model.train()
step_no = 0
micro_step_no=0
loss=0
batch_time=0
tokens_per_sec=0
tokens_processesed =0

if GRAD_ACCUM:
  for epoch in range(MAX_EPOCHS):
    for batch_no in range(MAX_STEPS): ##20k / batch size*grad_accum
      st_time = time.time()
      last_step = (batch_no == MAX_STEPS - 1) and (epoch == MAX_EPOCHS - 1)
      #do evaluation every EVAL_EVERY_STEPS steps
      if batch_no % EVAL_EVERY_STEPS == 0 or last_step:
        validation_loss, validation_acc = evaluation_step(model, val_data_loader, device, IS_DDP, MASTER_PROCESS)
        # validation_loss, validation_acc = 0, 0
      model.train(True)
      acc_accum=0
      loss_accum=0
      for micro_batch_no in range(GRAD_ACCUM_STEPS): 
        # for batch in data_loader:
        try:
            batch = next(data_iter)
        except StopIteration:
            #if no more data in the shard, try to get next shard
            if (shard_start + DDP_WORLD_SIZE*shard_size) > TRAIN_DATASET_SIZE:
                print("END OF DATA!")
                if IS_DDP:
                    destroy_process_group()
                exit()
            else:
                shard_start+=DDP_WORLD_SIZE*shard_size
                if (shard_end + shard_size) > TRAIN_DATASET_SIZE:
                    shard_end=TRAIN_DATASET_SIZE
                else:
                    shard_end=shard_start+shard_size
                data_iter = next_shard_iterator(shard_start,shard_end,tokenizer)
                print("[INFO] STARTING NEW SHARD: {} - {}".format(shard_start,shard_end))
                batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        if OPTIMIZE_SPEED:
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                outputs = model(batch['input_ids'])
        else:
            outputs = model(batch['input_ids'])
        loss = loss_fn(outputs.logits.transpose(1, 2), batch['labels'],ignore_index=-100)
        acc = get_masked_accuracy(outputs.logits,batch['labels']) #/ ACCUM_BATCH_SIZE
        acc_accum += acc.detach() / GRAD_ACCUM_STEPS
        loss_accum += loss.detach() / GRAD_ACCUM_STEPS
        loss.backward()
        micro_step_no+=1

      # if IS_DDP:
      #     dist.all_reduce(loss, op=dist.ReduceOp.AVG) #for logging only , average out logged loss across all GPUs
      optimizer.step()
      optimizer.zero_grad()
      ed_time = time.time()
      batch_time = (ed_time - st_time) # time difference
      tokens_per_sec = (MICRO_BATCH_SIZE * TOKENS_PER_SAMPLE * GRAD_ACCUM_STEPS * DDP_WORLD_SIZE) / batch_time
      lr = optimizer.param_groups[0]['lr']
      if MASTER_PROCESS:
        eval_log = ""
        if step_no % EVAL_EVERY_STEPS == 0 or last_step: #log evaluation loss every 100 steps
            eval_log = ", val_loss: {:.4f}, val_acc: {:.4f}".format(validation_loss,validation_acc)
            # with open(LOG_FILE, "a") as f:
            #     f.write(f"{step_no} val {validation_loss:.4f}\n")
        print("Step: {:5d}, MicroStep: {:5d}, batch_acc: {}, loss: {:.6f}, lr: {:.5e}, time(ms): {:.2f}, token/sec:{:.2f}{}".format(step_no, micro_step_no, acc_accum,loss_accum,lr, batch_time*1000, tokens_per_sec,eval_log))
      step_no+=1
else: #testing for simplest case without all the extra code to make things efficient
  for epoch in range(1):
    # for batch in data_loader:
    for micro_batch in range(1250): #20k / batch size
      batch = next(data_iter)
      batch = {k: v.to(device) for k, v in batch.items()}
      outputs = model(batch['input_ids'])
      loss = loss_fn(outputs.logits.transpose(1, 2), batch['labels'],ignore_index=-100)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      print("Step: {:5d}, batch_acc: {}, loss: {:.6f}, time(ms): {:.2f}, token/sec:{:.2f}".format(step_no, acc_accum,loss.detach(), batch_time*1000, tokens_per_sec))
      step_no+=1

if IS_DDP:
    destroy_process_group()
