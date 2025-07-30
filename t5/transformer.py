import torch
from torch import nn
import torch.nn.functional as F

from datasets import load_dataset
from collections import OrderedDict

VOCAB_SIZE = 30522 #30,5223000 #bert vocabulary size
MAX_INPUT_SIZE = 1024

EMBEDDING_SIZE=512 #emb size has to be divisable by heads 
ATTENTION_HEADS=8
TRANSFORMER_BLOCK_NO = 6
NUM_CLS = 2

#multi-head masked self-attention (used for encoder/decoder)
#if masked=True, doing masked self-attention with lower diagonal qk matrix zeroed out (this is used for decoder to avoid looking into the future labels when doing next token generation)
class SelfAttention(nn.Module):
    def __init__(self, embedding_size, max_context_size, heads_no=8, masked=False,optimize_speed = True):
        super().__init__()

        self.optimize_speed = optimize_speed
        self.masked = masked

        self.values = nn.Linear(in_features=embedding_size,out_features=embedding_size, bias=False)
        self.keys = nn.Linear(in_features=embedding_size,out_features=embedding_size, bias=False)
        self.queries = nn.Linear(in_features=embedding_size,out_features=embedding_size, bias=False)

        self.head_no = heads_no
        self.embedding_size = embedding_size
        self.head_size = embedding_size // heads_no #multi-head attention - split embedding into chunks, each processes by respective head

        self.c_proj = nn.Linear(in_features=embedding_size, out_features=embedding_size) #(self.unifyheads) multi-head attention, layer used to merge all head outputs together
        self.c_proj.CUSTOM_INIT_SCALING = 1 #custom attributed used during GPT-2 weight initalization

        self.scale_factor = self.head_size ** (1 / 2) #1/math.sqrt(emb // heads)

        # adding mask that is later used for masked attention
        self.register_buffer("bias", torch.tril(torch.ones(max_context_size, max_context_size))
                                     .view(1, max_context_size, max_context_size))
    
    #yi= sumj(wij * vj)
    #wij = qiT * kj
    #qi = Wq * xi / kj = Wk * xj  / vj = Wv *xj
    def forward(self, queries, keys, values):
        N = values.shape[0] #number of samples #number of samples
        batch_size, token_count, embedding_size = values.size()

        #qi = Wq * xi / kj = Wk * xj  / vj = Wv *xj
        queries = self.queries(queries)
        keys = self.keys(keys)
        values = self.values(values)

        #multi-head attention (split input into heads)
        keys    = keys.view(batch_size, token_count, self.head_no, self.head_size)
        queries = queries.view(batch_size, token_count, self.head_no, self.head_size)
        values  = values.view(batch_size, token_count, self.head_no, self.head_size)

        keys = keys.transpose(1, 2) #swap head count with token count first 
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        #attention mechanism below:
        if self.optimize_speed: #flash-attention (using pytorch version), about 30% speed-up
            out = F.scaled_dot_product_attention(query=queries, key=keys, value=values, is_causal=self.masked) # flash attention
        else: #my original vanilla implementation
            # in order to do single matrix multiplication for Q*K - fold heads into the batch dimension
            keys = keys.contiguous().view(batch_size * self.head_no, token_count, self.head_size) #reform the tensor pulling head_count into batch
            queries = queries.contiguous().view(batch_size * self.head_no, token_count, self.head_size)
            values = values.contiguous().view(batch_size * self.head_no, token_count, self.head_size)

            # wij = qiT * kj
            dot = torch.bmm(queries, keys.transpose(1, 2)) / self.scale_factor # moved scaling factor here instead of after softmax to match Karapathy implementation
            if self.masked:
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
        self.ln_1 = nn.LayerNorm(config.embedding_size)
        self.attn = SelfAttention(embedding_size=config.embedding_size, max_context_size= config.max_context_size, heads_no=config.head_count, masked=True, optimize_speed= config.optimize_speed)
        
        self.ln_2 = nn.LayerNorm(config.embedding_size)
        self.cross_attn = SelfAttention(embedding_size=config.embedding_size, max_context_size= config.max_context_size, heads_no=config.head_count, masked=False, optimize_speed= config.optimize_speed)
        
        self.ln_3 = nn.LayerNorm(config.embedding_size)
        #nlp output =4 * embedding_size, arbitrary choice to increase size 4x . This seem to be some "heuristic" done in all implementations.
        self.mlp = nn.Sequential(OrderedDict([
          ('c_fc', nn.Linear(in_features=config.embedding_size, out_features= 4 * config.embedding_size)),
          ('gelu', nn.GELU(approximate='tanh')),
          ('c_proj', nn.Linear(in_features=4 * config.embedding_size, out_features=config.embedding_size)),
        ]))

        self.mlp.c_proj.CUSTOM_INIT_SCALING = 1

    #input = tuple (x, encoderOutput) ; encoderOutput - used for cross attention in encoder-decoder setup
    def forward(self,inputs):
        x, encoderOutput= inputs
        output = self.ln_1(x)
        output = x + self.attn(queries=output, keys=output, values=output)
        if encoderOutput is not None:
            output = output + self.cross_attn(queries=encoderOutput, keys=encoderOutput, values=self.ln_2(output))
        output = output + self.mlp(self.ln_3(output))
        return (output, encoderOutput)

#------------------------------------------------------------------------------------------


class TransformerEncoderBlock(nn.Module):
    def __init__(self, config): #embedding_size, heads_no
        super().__init__()
        #transformer block == self-attention -> norm -> feedForward -> norm
        self.attention = SelfAttention(embedding_size=config.embedding_size, max_context_size= config.max_context_size, heads_no=config.head_count, masked=False, optimize_speed= config.optimize_speed )
        self.norm1 = nn.LayerNorm(config.embedding_size)
        self.norm2 = nn.LayerNorm(config.embedding_size)
        #output =4 * embedding_size, arbitrary choice to increase size 4x . This seem to be some "heuristic" done in all implementations.
        self.ff = nn.Sequential(
            nn.Linear(in_features=config.embedding_size, out_features= 4 * config.embedding_size), 
            nn.GELU(approximate='tanh'),
            nn.Linear(in_features=4 * config.embedding_size, out_features=config.embedding_size))

    def forward(self, x):
        output = self.norm1(x)
        output = x + self.attention(values=output ,keys=output,queries=output)
        output = output + self.ff(self.norm2(output))
        return output

#------------------------------------------------------------------------------------------

#extra, only for testing: transformer made for classification
class Classifier_Transformer(nn.Module):
    def __init__(self, embedding_size, heads_no, transformer_block_no, vocabulary_size, max_input_length, num_classes, device, max_pool=True, dropout=0.0):
        super().__init__()
        self.max_pool = max_pool #If true, use global max pooling in the last layer. If false, use global average pooling.

        #embeddings which are used to wrap the regular input
        self.token_embedding = nn.Embedding(embedding_dim=embedding_size, num_embeddings=vocabulary_size)
        self.pos_embedding = nn.Embedding(embedding_dim=embedding_size, num_embeddings=max_input_length)

        #dropout (done after embeddings for stabilising output)
        self.dropout = nn.Dropout(dropout)

        #transformer blocks
        transformer_blocks = []
        for i in range(transformer_block_no):
            transformer_blocks.append( TransformerBlock(embedding_size=embedding_size, heads_no=heads_no) )

        self.transformer_blocks = nn.Sequential(*transformer_blocks)

        #classification head
        self.classification_head = nn.Linear(in_features=embedding_size, out_features=num_classes) #one output value per each class

        self.device = device

    def forward(self, x):
        # return x
        #1. encode input into embedding (for each "word" in the input we create an embedding)
        tokens = self.token_embedding(x)
        # additionally calculate positional embedding, ie. sequential numbers for position of every "word" in the input
        batch_size, token_count, embedding_size = tokens.size()  #token_count == amount of input "words"
        positions = torch.arange(token_count,device=self.device) #actual value == sequence of word positions [0...#word_count]
        positions = self.pos_embedding(positions) #calculate embedding
        positions = positions[None, :, :] #add batch size to tensor (set as 1)
        positions = positions.expand(batch_size, token_count, embedding_size) # reshape to final embedding size, increasing to input batch size 

        x= tokens + positions

        x = tokens
        x= self.dropout(x)

        #2. go through all transformer blocks
        #out = self.transformerBlock(x)
        x= self.transformer_blocks(x)
        x= self.dropout(x)

        # pool over the time dimension
        if self.max_pool:
            x = x.max(dim=1)[0]
        else: 
            x.mean(dim=1) 

        x = self.classification_head(x)

        return F.softmax(x, dim=1) #make into probability
        # return F.log_softmax(x, dim=1) #make into probability



#test run
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    #test input (2 samples, each 9 "words" encoded with some dictonary)
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0]
                    , [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)

    #test input, generate random 2 inputs with tokens up to max vocab-size, and using max embedding size
    x = torch.randint(low=0, high=VOCAB_SIZE, size=(2, EMBEDDING_SIZE), dtype=torch.long).to(device)    

    transformer = Classifier_Transformer(embedding_size=EMBEDDING_SIZE
                            , heads_no=ATTENTION_HEADS
                            , transformer_block_no = TRANSFORMER_BLOCK_NO
                            , vocabulary_size= VOCAB_SIZE
                            , max_input_length=MAX_INPUT_SIZE
                            , num_classes = NUM_CLS
                            , device=device).to(device)
    out = transformer(x)
    print("INPUT size: {}".format(x.shape))
    print("OUTPUT size: {}".format(out.shape))
    print("---------------------------------")
    print("Output: {}".format(out))