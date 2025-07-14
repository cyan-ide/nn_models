import torch
from torch import nn
import torch.nn.functional as F

from datasets import load_dataset

VOCAB_SIZE = 30522 #30,5223000 #bert vocabulary size
MAX_INPUT_SIZE = 1024

EMBEDDING_SIZE=512 #emb size has to be divisable by heads 
ATTENTION_HEADS=8
TRANSFORMER_BLOCK_NO = 6
NUM_CLS = 2

#multi-head version
#implemented using "Transformers from Scratch" tutorial by Peter Bloem (https://peterbloem.nl/blog/transformers)
#(and some other web resources)
class SelfAttention(nn.Module):
    def __init__(self, embedding_size, heads_no=8):
        super().__init__()

        self.head_no = heads_no
        self.head_size = embedding_size // heads_no #multi-head attention - split embedding into chunks, each processes by respective head
        
        self.values = nn.Linear(in_features=embedding_size,out_features=embedding_size, bias=False)
        self.keys = nn.Linear(in_features=embedding_size,out_features=embedding_size, bias=False)
        self.queries = nn.Linear(in_features=embedding_size,out_features=embedding_size, bias=False)

        self.unifyheads = nn.Linear(embedding_size, embedding_size) #multi-head attention, layer used to merge all head outputs together

        self.scale_factor = self.head_size ** (1 / 2) #1/math.sqrt(emb // heads)
    
    #yi= sumj(wij * vj)
    #wij = qiT * kj
    #qi = Wq * xi / kj = Wk * xj  / vj = Wv *xj
    def forward(self, values,keys,queries):

        N = values.shape[0] #number of samples
        batch_size, token_count, embedding_size = values.size()

        #qi = Wq * xi / kj = Wk * xj  / vj = Wv *xj
        keys = self.keys(keys)
        queries = self.queries(queries)
        values = self.values(values)

        #multi-head attention (split input into heads)
        keys    = keys.view(batch_size, token_count, self.head_no, self.head_size)
        queries = queries.view(batch_size, token_count, self.head_no, self.head_size)
        values  = values.view(batch_size, token_count, self.head_no, self.head_size)

        # in order to do single matrix multiplication for Q*K - fold heads into the batch dimension
        # print(keys.shape)
        keys = keys.transpose(1, 2) #swap head count with token count first 
        keys = keys.contiguous().view(batch_size * self.head_no, token_count, self.head_size) #reform the tensor pulling head_count into batch
        queries = queries.transpose(1, 2).contiguous().view(batch_size * self.head_no, token_count, self.head_size)
        values = values.transpose(1, 2).contiguous().view(batch_size * self.head_no, token_count, self.head_size)

        #attention mechanism below:
        # wij = qiT * kj
        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = F.softmax(dot, dim=2) 
        dot = dot / self.scale_factor # this is added for training stability

        #yi= sumj(wij * vj)
        out = torch.bmm(dot, values) #.view(b, h, t, s)

        #extra for multi-head: inverse the folding of heads done prior to self-attetion
        out = out.view(batch_size, self.head_no, token_count, self.head_size) #unfold back again the head dimention
        out = out.transpose(1, 2).contiguous().view(batch_size, token_count, self.head_size * self.head_no)
        return self.unifyheads(out)
        #dot = dot * self.scalefactor
        
#------------------------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, heads_no):
        super().__init__()
        #transformer block == self-attention -> norm -> feedForward -> norm

        self.attention = SelfAttention(embedding_size=embedding_size, heads_no=heads_no)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        #output =4 * embedding_size, arbitrary choice to increase size 4x . This seem to be some "heuristic" done in all implementations.
        self.ff = nn.Sequential(
            nn.Linear(in_features=embedding_size, out_features= 4 * embedding_size), 
            nn.GELU(approximate='tanh'),
            nn.Linear(in_features=4 * embedding_size, out_features=embedding_size))

    def forward(self, x):
        #v1 - mine
        # output = self.attention(values=x ,keys=x,queries=x)
        # output_mid = self.norm1(output+x) #+x is the residual connection
        # output = self.ff(output_mid)
        # output = self.norm2(output+output_mid) #+output_min is the residual connection
        #v2 -ln before attention (as in karapathy gpt)
        output = self.norm1(x)
        output = x + self.attention(values=output ,keys=output,queries=output)
        output = output + self.ff(self.norm2(output))
        return output

#------------------------------------------------------------------------------------------

#test run
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    #test input (2 samples, each 9 "words" encoded with some dictonary)
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0]
                    , [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)

    #test input, generate random 2 inputs with tokens up to max vocab-size, and using max embedding size
    x = torch.randint(low=0, high=VOCAB_SIZE, size=(2, EMBEDDING_SIZE), dtype=torch.long).to(device)    

    #TODO