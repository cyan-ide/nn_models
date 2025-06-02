from gpt2 import GPT_2, GPT_Config
import tiktoken

import torch
import torch.nn.functional as F

import os
import inspect
import time
import math

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

OPTIMIZE_SPEED = True

#init distributed processing
# simple launch:
# python train_gpt2.pyAdd commentMore actions
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

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

#learning rate scheduler settings
MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 10
#learning steps settings
MAX_STEPS = 50
#data processing settings
ACCUM_BATCH_TOKEN_COUNT = 524288 # 2**19, ~0.5M, in number of tokens (this is the original GPT2 batch size oO, "just a bit big" -> need to do gradient accumulation for this!)
#this is adjustable given hardware
MICRO_BATCH_SIZE = 16
MICRO_BATCH_TOKEN_COUNT = 1024
ACCUM_BATCH_SIZE = ACCUM_BATCH_TOKEN_COUNT // (MICRO_BATCH_SIZE * MICRO_BATCH_TOKEN_COUNT * DDP_WORLD_SIZE) #calculated given above

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
    print("\nWEIGHT DECAY CONFIG:")
    print("* Number of tensors with weight decay / without weight decay: {} ({} params)/ {} ({} params)".format(len(decay_params), num_decay_params, len(nodecay_params), num_nodecay_params) )
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available #and 'cuda' in device
    print("* Using fused AdamW: {}".format(use_fused))
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer

#work with DDP / distributed processing, splits dataset per node with offset the size of batch in tokens
class DataLoader():
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

        # print("self.tokens: {}".format(len(self.tokens)))
        # print("self.tokens: {}".format(self.current_position))
        # print("data_batch: {}".format(data_batch.size()))
        #reshape batch data to split it into samples
        x = (data_batch[:-1]).view(self.batch_size, self.token_count) # inputs (all except final token which is for prediction)
        y = (data_batch[1:]).view(self.batch_size, self.token_count) # outputs (all except first token)

        #iterate to next batch
        self.current_position += self.batch_size * self.token_count * self.num_processes #move the pointer to next batch
        if self.current_position+self.batch_size * self.token_count * self.num_processes +1 > len(self.tokens): #if end of the dataset, start all over again (+1 is due to prediction of next token for GT)
            self.current_position= self.batch_size * self.token_count * self.process_rank #reset to starting point for current node with its offset

        return x,y


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

tokenizer = tiktoken.get_encoding('gpt2')
# model = GPT_2.from_pretrained('gpt2')
config = GPT_Config()
if OPTIMIZE_SPEED: #nice number, rather than odd default GPT2 vocab size
    config.vocabulary_size=50304
model = GPT_2(config)
model.to(device)
if OPTIMIZE_SPEED:
    model = torch.compile(model)

raw_model = model
if IS_DDP: #wrapper around model for DDP to keep things in sync
    model = DDP(model, device_ids=[DDP_LOCAL_RANK])
    raw_model = model.module

#load data in microbatches that fit to our GPU
train_loader = DataLoader(batch_size=MICRO_BATCH_SIZE, token_count=MICRO_BATCH_TOKEN_COUNT,tokenizer=tokenizer, process_rank=DDP_RANK, num_processes=DDP_WORLD_SIZE) 

# [optional: optimalization] enable TF32 instead of float32 for faster processing
if OPTIMIZE_SPEED:
    torch.set_float32_matmul_precision('high')


#training loop
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8) #optimizer paramas like from GPT3 paper
optimizer = configure_optimizers(raw_model, weight_decay=0.1, learning_rate=6e-4)
model.train(True)
for step_no in range(MAX_STEPS):
    st_time = time.time()
    optimizer.zero_grad()

    #gradient accumulation (calc multiple micro batches before doing gradient update)
    loss_accum = 0.0 #log gradient accumulation loss 
    for micro_batch in range(ACCUM_BATCH_SIZE):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if OPTIMIZE_SPEED:
            with torch.autocast(device_type=device, dtype=torch.bfloat16): #adding bfloat16 (insead of float32)
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        loss = loss / ACCUM_BATCH_SIZE #this accounts for the fact that we loss in a mean and by adding multiple losses we dont simulate doing mean over all of samples in one go
        loss_accum += loss.detach() #logging purpose only
        if IS_DDP:
            model.require_backward_grad_sync = (micro_batch == ACCUM_BATCH_SIZE - 1) #disable sync loss across GPUs until final calculation of this loop
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
        print("Step: {:4d}, loss: {:.6f}, norm: {:.4f}, time(ms): {:.2f}, token/sec:{:.2f}".format(step_no, loss_accum.item(), norm, batch_time*1000, tokens_per_sec))


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


