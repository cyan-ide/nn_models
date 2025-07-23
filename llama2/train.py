from llama2 import Llama2, Llama_Config
from hellaswag_eval import render_example, iterate_examples

import tiktoken

import torch
import torch.nn.functional as F
import numpy as np

import os
import inspect
import time
import math

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

OPTIMIZE_SPEED = True
USE_COMPILE = True
DATASET_PATH = "/mnt/data/backup/adam/llm_train_data/fineweb_edu_10BT/"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, f"log.txt")

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

#parameters set as in the GPT-2/3 paper, WARMUP_STEPS/MAX_STEPS adjusted to iterate over the entire fineweb dataset
#learning rate scheduler settings
MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 715 #based on GPT-3 paper, 375M tokens, so 375M/ 524288 = 715 #10
#learning steps settings
MAX_STEPS = 19073 # == total dataset size (10B fineweb) / total tokens per batch = 10e9 / 524288 = 19073 #4
#data processing settings
ACCUM_BATCH_TOKEN_COUNT = 524288 # 2**19, ~0.5M, in number of tokens (this is the original GPT2 batch size oO, "just a bit big" -> need to do gradient accumulation for this!)
#this is adjustable given hardware
MICRO_BATCH_SIZE = 32 #64 #16
MICRO_BATCH_TOKEN_COUNT = 1024
ACCUM_BATCH_SIZE = ACCUM_BATCH_TOKEN_COUNT // (MICRO_BATCH_SIZE * MICRO_BATCH_TOKEN_COUNT * DDP_WORLD_SIZE) #calculated given above
#evaluation
EVAL_EVERY_STEPS = 250
GEN_EVERY_STEPS = 250 #100
EVAL_HELLASWAG_EVERY_STEPS = 1000 

assert ACCUM_BATCH_TOKEN_COUNT % (MICRO_BATCH_SIZE * MICRO_BATCH_TOKEN_COUNT * DDP_WORLD_SIZE) == 0, "BATCH_TOKEN_COUNT needs to be divisible by MICRO_BATCH_SIZE * MICRO_BATCH_TOKEN_COUNT * DDP_WORLD_SIZE"
if MASTER_PROCESS:
    print("Total desired batch size: {} tokens / {} steps".format(ACCUM_BATCH_TOKEN_COUNT,ACCUM_BATCH_SIZE))

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
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer

#dataloader for simple/short dataset
#work with DDP / distributed processing, splits dataset per node with offset the size of batch in tokens
class TinyShakrespearDataLoader():
    def __init__(self, batch_size, token_count, tokenizer, process_rank, num_processes): #amount of samples in one batch , amount of tokens per sample
        self.batch_size = batch_size
        self.token_count = token_count
        self.process_rank = process_rank
        self.num_processes = num_processes

        #for testing small file, so load everyting at init and store in memory
        with open('input.txt','r') as file:
            text = file.read()

        self.tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(self.tokens, dtype=torch.long) # (8,)

        if MASTER_PROCESS:
            print("Loaded {} tokens (1 epoch= {})".format(len(self.tokens), len(self.tokens) // (batch_size * token_count)))

        #state which sample is currently loaded
        #for each node in DDP this will be a differnt starting point, we condition it on node number
        self.current_position = self.batch_size * self.token_count * self.process_rank 

        #data batches are some <token_count> portion of dataset; the targets is same size matrix as we provide prediction for every
        #token in the input (e.g. for input[0,0] next token is output[0,0]; for input[0:1,0] target is output[1,0] etc.)
    def next_batch(self):
        data_batch = self.tokens[self.current_position:self.current_position+self.batch_size * self.token_count+1] #+1 is to expand for ground truth of the last element (ie. next token)

        #reshape batch data to split it into samples
        x = (data_batch[:-1]).view(self.batch_size, self.token_count) # inputs (all except final token which is for prediction)
        y = (data_batch[1:]).view(self.batch_size, self.token_count) # outputs (all except first token)

        #iterate to next batch
        self.current_position += self.batch_size * self.token_count * self.num_processes #move the pointer to next batch
        if self.current_position+self.batch_size * self.token_count * self.num_processes +1 > len(self.tokens): #if end of the dataset, start all over again (+1 is due to prediction of next token for GT)
            self.current_position= self.batch_size * self.token_count * self.process_rank #reset to starting point for current node with its offset

        return x,y


#extended version of dataloader to cope with a bigger dataset split into multiple files
#work with DDP / distributed processing, splits dataset per node with offset the size of batch in tokens
class FineWebDataLoader():
    def __init__(self, batch_size, token_count, tokenizer, process_rank, num_processes, dataset_path ="", split="train"): #amount of samples in one batch , amount of tokens per sample
        self.batch_size = batch_size
        self.token_count = token_count
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
            print("Found {} files. Current file - loaded {} tokens (1 epoch= {} steps / 1 step = {} tokens)".format(len(shards),len(self.tokens), len(self.tokens) // (batch_size * token_count), self.batch_size * self.token_count))


    #data batches are some <token_count> portion of dataset; the targets is same size matrix as we provide prediction for every
    #token in the input (e.g. for input[0,0] next token is output[0,0]; for input[0:1,0] target is output[1,0] etc.)
    def next_batch(self):
        data_batch = self.tokens[self.current_position:self.current_position+self.batch_size * self.token_count+1] #+1 is to expand for ground truth of the last element (ie. next token)

        #reshape batch data to split it into samples
        x = (data_batch[:-1]).view(self.batch_size, self.token_count) # inputs (all except final token which is for prediction)
        y = (data_batch[1:]).view(self.batch_size, self.token_count) # outputs (all except first token)

        #iterate to next batch
        self.current_position += self.batch_size * self.token_count * self.num_processes #move the pointer to next batch
        #if end of file / the dataset, go to next file / start all over again (+1 is due to prediction of next token for GT)
        if self.current_position+self.batch_size * self.token_count * self.num_processes +1 > len(self.tokens): 
            self.current_shard = (self.current_shard + 1) % len(self.shards) #set current file to next one
            self.tokens = load_tokens(self.shards[self.current_shard]) #load the file
            self.current_position= self.batch_size * self.token_count * self.process_rank #reset to starting point for current node with its offset
        return x,y

    #reset all points to start from the start of dataset
    def reset_positions(self):
        self.current_shard = 0 
        self.tokens = load_tokens(self.shards[self.current_shard]) #load data
        #state which sample is currently loaded
        #for each node in DDP this will be a differnt starting point, we condition it on node number
        self.current_position = self.batch_size * self.token_count * self.process_rank 

#read dataset file (earlier serialized into tokens with tiktoken and coverted to numpy array
def load_tokens(filename):
    tokens = np.load(filename)
    tokens = tokens.astype(np.int32)
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    return tokens_tensor


def evaluation_step(model, val_loader, device, ddp, master_process):
    model.eval()
    val_loader.reset_positions()
    with torch.no_grad():
        validation_loss_accum = 0.0 #loss accumulated over all distrubted processes
        validation_steps = 20
        #iterate entire validation set
        # for micro_batch in range(ACCUM_BATCH_SIZE):
        for _ in range(validation_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / validation_steps
            validation_loss_accum += loss.detach()
    if ddp:
        dist.all_reduce(validation_loss_accum, op=dist.ReduceOp.AVG)
    # if master_process:
    #     print(f"validation loss: {validation_loss_accum.item():.4f}")
    return validation_loss_accum.item()

def generate_samples(model, tokenizer, device, ddp_rank):
    model.eval()
    num_return_sequences = 4
    max_length = 32
    tokens = tokenizer.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    #pad input == torch.compile needs same model input size (ie. MICRO_BATCH_SIZE x MICRO_BATCH_TOKEN_COUNT)

    #pad token dimention
    at_idx = tokens.size(1) #original input size in tokens
    pad_x_to = (MICRO_BATCH_TOKEN_COUNT if USE_COMPILE else max_length) - tokens.size(1) #pading size in tokens
    tokens = torch.cat((tokens, torch.zeros([tokens.size(0), pad_x_to], dtype=torch.long)), dim=1) #do the padding

    # pad batch size (ie. increase number of samples in batch to match MICRO_BATCH_SIZE)
    if num_return_sequences != MICRO_BATCH_SIZE and USE_COMPILE:
        assert num_return_sequences <= MICRO_BATCH_SIZE, f"TODO: {num_return_sequences=} > {MICRO_BATCH_SIZE=}; not supported, add smaller gen input!"
        tokens = F.pad(tokens, (0, 0, 0, MICRO_BATCH_SIZE - num_return_sequences), 'constant', tokenizer.eot_token) #pad some placeholder samples that will be ignored later
        # When MICRO_BATCH_SIZE is smaller than the number of examples we generate, we can simply loop over the example generation code.

    x = tokens.to(device)
    #local rng generator that doesnt mess up main
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)

    # while x.size(1) < max_length:
    while at_idx < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x)
            # take the logits at the last position (other discard as they show probabilities at position we already know in "x")
            #logits = logits[:, -1, :] # (B, vocab_size)
            logits = logits[:, at_idx-1, :] # (B, vocab_size) , only read the logits for non-padded predictions
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            # x = torch.cat((x, xcol), dim=1)
            x[:, at_idx:at_idx + 1] = xcol
        at_idx += 1

    # print the generated text
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = tokenizer.decode(tokens)
        print(f"rank {ddp_rank} sample {i}: {decoded}")


#run LLM bankmarch for comparison with original GPT2/3
def evaluation_hellaswag(model, tokenizer, device, ddp, ddp_world_size, ddp_rank):
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        # only process examples where i % ddp_world_size == ddp_rank
        if i % ddp_world_size != ddp_rank:
            continue
        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)

        pad_x_to = (MICRO_BATCH_TOKEN_COUNT if USE_COMPILE else tokens.size(1)) - tokens.size(1)
        tokens = torch.cat((tokens, torch.zeros([tokens.size(0), pad_x_to], dtype=torch.long)), dim=1)
        mask = torch.cat((mask, torch.zeros([mask.size(0), pad_x_to], dtype=torch.long)), dim=1)

        # pad batch size (ie. increase number of samples in batch to match MICRO_BATCH_SIZE)
        if tokens.size(0) != MICRO_BATCH_SIZE and USE_COMPILE:
            assert tokens.size(0) <= MICRO_BATCH_SIZE, f"TODO: {tokens.size(0)=} > {MICRO_BATCH_SIZE=}; not supported, add smaller hellaswag batch size input!"
            tokens = F.pad(tokens, (0, 0, 0, MICRO_BATCH_SIZE - tokens.size(0)), 'constant', tokenizer.eot_token) #pad some placeholder samples that will be ignored later
            mask = F.pad(mask, (0, 0, 0, MICRO_BATCH_SIZE - mask.size(0)), 'constant', tokenizer.eot_token) #pad some placeholder samples that will be ignored later
            # When MICRO_BATCH_SIZE is smaller than the number of examples we generate, we can simply loop over the example generation code.

        # print(tokens.size())
        tokens = tokens.to(device)
        mask = mask.to(device)
        # get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    # reduce the stats across all processes
    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    return acc_norm, num_correct_norm, num_total

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

#print current config
if MASTER_PROCESS:
    print("\n\nGPT-2 training configuration")
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

tokenizer = tiktoken.get_encoding('gpt2')
# model = GPT_2.from_pretrained('gpt2')
config = Llama_Config()
if OPTIMIZE_SPEED: #nice number, rather than odd default GPT2 vocab size
    config.vocabulary_size=50304
model = Llama2(config)
model.to(device)

if OPTIMIZE_SPEED & USE_COMPILE: # torch.compile interferes with Generation.
    model = torch.compile(model)

raw_model = model
if IS_DDP: #wrapper around model for DDP to keep things in sync
    model = DDP(model, device_ids=[DDP_LOCAL_RANK])
    raw_model = model.module

#load data in microbatches that fit to our GPU
# train_loader = TinyShakrespearDataLoader(batch_size=MICRO_BATCH_SIZE, token_count=MICRO_BATCH_TOKEN_COUNT,tokenizer=tokenizer, process_rank=DDP_RANK, num_processes=DDP_WORLD_SIZE) 
train_loader = FineWebDataLoader(batch_size=MICRO_BATCH_SIZE
                                , token_count=MICRO_BATCH_TOKEN_COUNT
                                , tokenizer=tokenizer
                                , process_rank=DDP_RANK
                                , num_processes=DDP_WORLD_SIZE
                                , dataset_path=DATASET_PATH
                                , split="train") 
val_loader =  FineWebDataLoader(batch_size=MICRO_BATCH_SIZE
                                , token_count=MICRO_BATCH_TOKEN_COUNT
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
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8) #optimizer paramas like from GPT3 paper
optimizer = configure_optimizers(raw_model, weight_decay=0.1, learning_rate=6e-4)
validation_loss = 0
hellaswag_acc_norm = 0
for step_no in range(MAX_STEPS):
    st_time = time.time()
    last_step = (step_no == MAX_STEPS - 1)

    #do evaluation every EVAL_EVERY_STEPS steps
    if step_no % EVAL_EVERY_STEPS == 0 or last_step:
        validation_loss = evaluation_step(model, val_loader, device, IS_DDP, MASTER_PROCESS)

    # do hellaswag evaluation every EVAL_HELLASWAG_EVERY_STEPS steps
    if (step_no % EVAL_HELLASWAG_EVERY_STEPS == 0 or last_step):
        hellaswag_acc_norm, hellaswag_num_correct_norm, hellaswag_num_total = evaluation_hellaswag(model, tokenizer, device, IS_DDP, DDP_WORLD_SIZE,DDP_RANK)

    # every GEN_EVERY_STEPS generate from model to see how its doing
    if step_no > 0 and step_no % GEN_EVERY_STEPS == 0:
        generate_samples(model,tokenizer,device, DDP_RANK)

    model.train(True)
    optimizer.zero_grad()
    #gradient accumulation (calc multiple micro batches before doing gradient update)
    loss_accum = 0.0 #log gradient accumulation loss 
    for micro_batch in range(ACCUM_BATCH_SIZE):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if IS_DDP: #for distrubted - disable sync loss across GPUs until final calculation of this loop
            model.require_backward_grad_sync = (micro_batch == ACCUM_BATCH_SIZE - 1) 
        if OPTIMIZE_SPEED:
            with torch.autocast(device_type=device, dtype=torch.bfloat16): #adding bfloat16 (insead of float32)
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        loss = loss / ACCUM_BATCH_SIZE #this accounts for the fact that we loss in a mean and by adding multiple losses we dont simulate doing mean over all of samples in one go
        loss_accum += loss.detach() #logging purpose only
        loss.backward()

    if IS_DDP:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) #for logging only , average out logged loss across all GPUs

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #added gradient norm clipping (from GPT3 paper). apparently this prevents the model to get big shocks from bad/odd batches

    # determine and set the learning rate for this iteration (from GPT3 paper)
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
        if step_no % EVAL_EVERY_STEPS == 0: #log evaluation loss every 100 steps
            eval_log = ", val_loss: {:.4f}".format(validation_loss)
            with open(LOG_FILE, "a") as f:
                f.write(f"{step_no} val {validation_loss:.4f}\n")
        if step_no % EVAL_HELLASWAG_EVERY_STEPS == 0: #log evaluation loss every 100 steps
            hellaswag_eval_log = ", hellaswag_acc: {:.4f}".format(hellaswag_acc_norm)
            with open(LOG_FILE, "a") as f:
                f.write(f"{step_no} hella {hellaswag_acc_norm:.4f}\n")
        print("Step: {:5d}, loss: {:.6f}, norm: {:.4f}, time(ms): {:.2f}, token/sec:{:.2f}{}{}".format(step_no, loss_accum.item(), norm, batch_time*1000, tokens_per_sec,eval_log,hellaswag_eval_log))


if IS_DDP:
    destroy_process_group()
#test model
# num_return_sequences = 5
# max_length = 30

# model.eval()

# # prefix tokens
# tokens = tokenizer.encode("Hello, I'm a language model,")
# tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
# x = tokens.to(device)
# # print(x.size())
# # print("*")
# #since we know whats gonna be generated , pick one line to see good loss
# tokens = torch.tensor(tokenizer.encode("Hello, I'm a language model, I really like languages. I like languages because like, they're good. And the way we talk about languages"), dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
# y = tokens.to(device)

# # generate! right now x is (B, T) where B = 5, T = 8
# # set the seed to 42
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) < max_length:
#     # forward the model to get the logits
#     with torch.no_grad():
#         y_tmp = y[:,:x.size(1)].contiguous()  # output = (B, T, vocab_size) / contiguous is done to store new tensor in one chunk of memeory (so that we can do .view later)
#         logits, loss = model(x, y_tmp)
#         # take the logits at the last position (other discard as they show probabilities at position we already know in "x")
#         logits = logits[:, -1, :] # (B, vocab_size)
#         # get the probabilities
#         probs = F.softmax(logits, dim=-1)
#         # do top-k sampling of 50 (huggingface pipeline default)
#         # topk_probs here becomes (5, 50), topk_indices is (5, 50)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#         # select a token from the top-k probabilities
#         # note: multinomial does not demand the input to sum to 1
#         ix = torch.multinomial(topk_probs, 1) # (B, 1)
#         # gather the corresponding indices
#         xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
#         # append to the sequence
#         x = torch.cat((x, xcol), dim=1)
#         print(loss)

# # print the generated text
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = tokenizer.decode(tokens)
#     print(">", decoded)


