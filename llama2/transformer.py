import torch
from torch import nn
import torch.nn.functional as F

from datasets import load_dataset
from collections import OrderedDict

import math

VOCAB_SIZE = 30522 #30,5223000 #bert vocabulary size
MAX_INPUT_SIZE = 512

EMBEDDING_SIZE=512
ATTENTION_HEADS=8
TRANSFORMER_BLOCK_NO = 6
NUM_CLS = 2
        
#------------------------------------------------------------------------------------------

#rotary embeddings (replace regular absolute positional embeddings that are calculated at trasformer block level)
# this implementation disagregates the usual rotary matrix into two component for more computational complexity
# implementation follows "Build a Large Language Model From Scratch" book by Sebastian Raschka

def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    positions = torch.arange(context_length)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def compute_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)

#multi-head masked self-attention (used for decoder, layer names are matching hf implementation for weight loading)

class MaskedSelfAttention(nn.Module):
    def __init__(self, embedding_size, max_context_size, heads_no=8, optimize_speed = True):
        super().__init__()

        self.optimize_speed = optimize_speed

        # (self.values, self.keys, self.queries) all concatinated into a single layer
        self.c_attn = nn.Linear(in_features=embedding_size,out_features=3 * embedding_size, bias=False)

        self.head_no = heads_no
        self.embedding_size = embedding_size
        self.head_size = embedding_size // heads_no #multi-head attention - split embedding into chunks, each processes by respective head

        self.c_proj = nn.Linear(in_features=embedding_size, out_features=embedding_size, bias=False) #(self.unifyheads) multi-head attention, layer used to merge all head outputs together
        self.c_proj.CUSTOM_INIT_SCALING = 1 #custom attributed used during GPT-2 weight initalization

        self.scale_factor = self.head_size ** (1 / 2) #1/math.sqrt(emb // heads)

        # adding mask that is later used for masked attention
        self.register_buffer("bias", torch.tril(torch.ones(max_context_size, max_context_size))
                                     .view(1, max_context_size, max_context_size))

        #rotary matrix split into two components (independent from input, this essentially encode all rotations depending on token position in input)
        cos, sin = precompute_rope_params(head_dim=self.head_size, context_length=max_context_size)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
    
    #yi= sumj(wij * vj)
    #wij = qiT * kj
    #qi = Wq * xi / kj = Wk * xj  / vj = Wv *xj
    def forward(self, x):
        N = x.shape[0] #number of samples
        batch_size, token_count, embedding_size = x.size()

        #qi = Wq * xi / kj = Wk * xj  / vj = Wv *xj (all done in one operation, as qkv are concat into batch dimention)
        qkv = self.c_attn(x)
        queries, keys, values = qkv.split(self.embedding_size, dim=2)

        #multi-head attention (split input into heads)
        keys    = keys.view(batch_size, token_count, self.head_no, self.head_size)
        queries = queries.view(batch_size, token_count, self.head_no, self.head_size)
        values  = values.view(batch_size, token_count, self.head_no, self.head_size)

        keys = keys.transpose(1, 2) #swap head count with token count first 
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        #prior to attention dot product apply the rotations (for rotary positional embedding)
        keys = compute_rope(keys, self.cos, self.sin)
        queries = compute_rope(queries, self.cos, self.sin)

        #attention mechanism below:
        if self.optimize_speed: #flash-attention (using pytorch version), about 30% speed-up
            out = F.scaled_dot_product_attention(query=queries, key=keys, value=values, is_causal=True) # flash attention
        else: #my original vanilla implementation
            # in order to do single matrix multiplication for Q*K - fold heads into the batch dimension
            keys = keys.contiguous().view(batch_size * self.head_no, token_count, self.head_size) #reform the tensor pulling head_count into batch
            queries = queries.contiguous().view(batch_size * self.head_no, token_count, self.head_size)
            values = values.contiguous().view(batch_size * self.head_no, token_count, self.head_size)

            # wij = qiT * kj
            dot = torch.bmm(queries, keys.transpose(1, 2)) / self.scale_factor # moved scaling factor here instead of after softmax to match Karapathy implementation
            dot = dot.masked_fill(self.bias[:,:token_count,:token_count] == 0, float('-inf')) # <-- this is the masked attention part (only different to regular attention)
            dot = F.softmax(dot, dim=2) 
            dot = dot #/ self.scale_factor # this is added for training stability

            #yi= sumj(wij * vj)
            out = torch.bmm(dot, values) #.view(b, h, t, s)

            #extra for multi-head: inverse the folding of heads done prior to self-attetion
            out = out.view(batch_size, self.head_no, token_count, self.head_size) #unfold back again the head dimention

        out = out.transpose(1, 2).contiguous().view(batch_size, token_count, self.head_size * self.head_no)
        return self.c_proj(out)

#------------------------------------------------------------------------------------------
# decoder block used for most LLMs (parameter names are dont to match hf implementation for easier weight loading)
class TransformerDecoderBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        #transformer block == (masked)self-attention -> norm -> feedForward -> norm
        self.ln_1 = nn.RMSNorm(normalized_shape=config.embedding_size) #nn.LayerNorm(config.embedding_size)
        self.attn = MaskedSelfAttention(embedding_size=config.embedding_size, max_context_size= config.max_context_size, heads_no=config.head_count, optimize_speed= config.optimize_speed)
        # self.attn = CausalSelfAttention(config)
        
        self.ln_2 = nn.RMSNorm(normalized_shape=config.embedding_size) #nn.LayerNorm(config.embedding_size)
        #nlp output =4 * embedding_size, arbitrary choice to increase size 4x . This seem to be some "heuristic" done in all implementations.
        self.fc1 = nn.Linear(in_features=config.embedding_size, out_features=config.transformer_feed_forward_hidden_dim, bias=False) #, dtype=cfg["dtype"]
        self.fc2 = nn.Linear(in_features=config.embedding_size, out_features=config.transformer_feed_forward_hidden_dim, bias=False)
        self.fc3 = nn.Linear(in_features=config.transformer_feed_forward_hidden_dim, out_features=config.embedding_size, bias=False)
        self.silu = nn.SiLU()
        # self.mlp = nn.Sequential(OrderedDict([
        #   ('c_fc', nn.Linear(in_features=config.embedding_size, out_features= 4 * config.embedding_size)),
        #   ('gelu', nn.SiLU()), #nn.GELU(approximate='tanh')),
        #   ('c_proj', nn.Linear(in_features=4 * config.embedding_size, out_features=config.embedding_size)),
        # ]))

        self.fc3.CUSTOM_INIT_SCALING = 1

    def forward(self,x):
        #TODO check why Karapathy applie norm before rather than after those layer (in Blum implementation it was the other way around)
        output = x + self.attn(self.ln_1(x))
        #Feed forward with SwiGLU below
        residual_connection = output
        output = self.ln_2(output)
        output1 = self.fc1(output)
        output2 = self.fc2(output)
        output = self.silu(output1) * output2
        output = self.fc3(output)
        output = output+ residual_connection
        return output


#------------------------------------------------------------------------------------------
