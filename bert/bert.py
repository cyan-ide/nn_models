import torch
from torch import nn
import torch.nn.functional as F

from transformer import TransformerBlock

#hyper-parameters from BERT article
N_LAYERS = 12
EMBEDDING_SIZE = 768
N_ATTN_HEADS = 12
DROPOUT = 0.3

N_SEGMENTS = 3 #segments no for segment embedding
MAX_INPUT_LEN = 512
VOCABULARY_SIZE = 30000


# almost like any classic transfomer model, just that: 1) adds segmentation embedding, 2) has two learning heads depending on pre-training type
# addition = segmentation embedding:
# - authors "allow" input to be explicitly constructed from 2 seperate sentences (optionally, can be also one sentence as special case)
# - segmentation embedding is a mask that tells to which sentences the token belogs to (if single sentence segmentation mask is same value across) (e.g. the positional embedding tells the order to tokens, this is similar principle just tells different information)
# - additionally for this sentences split they include [SEP] token in the input sequence
# - segmentation embedding is related to the fact that authors have 2 types of pre-training (see below)
# addition = learning heads:
# - regular word prediction (here is different to LLMs in way that they mask a word from inside the sentence and predict it, rather than "next" word after the sentence). Reason for this difference is usage of "encoder" without masked attention rather than decoder.
# - next sentence predictiin (this is where segmentation emb is used). Input is two sentences, and output is "next" or "not next". Model decides if second sentence is indeed next or not.
class BERT(nn.Module):
    def __init__(self, embedding_size, heads_no, vocabulary_size, max_input_length, segment_no, dropout, device, max_pool= True, pretraining_type='pred_word'):
        super().__init__()
        self.device = device
        self.max_pool = max_pool

        #embeddings which are used to wrap the regular input        
        self.token_embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_size, padding_idx=0)
        # self.segment_embedding = nn.Embedding(num_embeddings=segment_no, embedding_dim=embedding_size, padding_idx=0)
        self.position_embedding = nn.Embedding(num_embeddings=max_input_length, embedding_dim=embedding_size)

        #transformer blocks
        transformer_blocks = []
        for i in list(range(N_LAYERS)):
            transformer_blocks.append(TransformerBlock(embedding_size=embedding_size, heads_no=heads_no))

        self.transformer_blocks = nn.Sequential(*transformer_blocks)

        #dropout (done after embeddings for stabilising output)
        self.dropout = nn.Dropout(dropout)

        #learning heads for 2 pre-training types (there are 2 in BERT paper)
        self.lm_head_1 = nn.Linear(in_features=embedding_size, out_features=vocabulary_size, bias=False) #(1st pre-training type): predict missing word from vocabulary
        #TEMP remove NSP: as it gives ~1-2% difference, has been proven not to work in later research and its quite a drag to implement
        # self.lm_head_2 = nn.Linear(in_features=embedding_size,  out_features=2, bias=False) #(2nd pre-training type): predict next sentence (two sentenes are given as input and model needs to decide if one follows another, i.e. 2 labels, next or not-next)

    def forward(self, x, y=None): #, x_segment_mask #TEMP remove NSP
        #embeddings
        tokens = self.token_embedding(x)
        #TEMP remove NSP: as it gives ~1-2% difference, has been proven not to work in later research and its quite a drag to implement
        # segments = self.segment_embedding(x_segment_mask)

        batch_size, token_count, embedding_size = tokens.size()  #token_count == amount of input "words"
        positions = torch.arange(token_count,device=self.device) #actual value == sequence of word positions [0...#word_count]
        positions = self.position_embedding(positions)
        positions = positions[None, :, :] #add batch size to tensor (set as 1)
        positions = positions.expand(batch_size, token_count, embedding_size) # reshape to final embedding size, increasing to input batch size 

        # x = tokens + segments + positions #TEMP remove NSP
        x = tokens + positions
        # x = tokens
        # x = positions

        x= self.dropout(x)
        
        #2. go through all transformer blocks
        #out = self.transformerBlock(x)
        x= self.transformer_blocks(x)
        x= self.dropout(x)

        #3. learning head 
        y1 = self.lm_head_1(x) #pre-training #1, predict probs for every token (ie. predict masked token) # [<batch_n>, 128, 30000]

        #TEMP remove NSP
        #pre-training #2, predict if sentences given as B is next or not (ie. 2 probabilities for binary choice)
        # at output we want only 2 predictions so pool over the token_count dimension to reduce size (either via max or mean)
        # if self.max_pool:
        #     x = x.max(dim=1)[0]
        # else: 
        #     x.mean(dim=1) 
        # y2 = self.lm_head_2(x)

        # return F.softmax(y1, dim=2), F.softmax(y2, dim=1) #make into probability
        # return y1, F.softmax(y1, dim=2), y2, F.softmax(y2, dim=1)
        #
        return y1 #, y2 #output un-normalized values (ie. later the crossEntropy uses non-normalized, softmax is needed only for predictions)
        #return y1, y2 #TEMP remove NSP #output un-normalized values (ie. later the crossEntropy uses non-normalized, softmax is needed only for predictions)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = BERT(embedding_size=EMBEDDING_SIZE
                , heads_no=N_ATTN_HEADS
                , vocabulary_size=VOCABULARY_SIZE
                , max_input_length=MAX_INPUT_LEN
                , segment_no=N_SEGMENTS
                , dropout=DROPOUT
                , device= device).to(device)


    #test input (2 samples, each 9 "words" encoded with some dictonary)
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0]
                    , [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    x_seg = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]
                    , [1, 1, 1, 1, 1, 2, 2, 2, 2]]).to(device)

    # x = torch.randint(low=0, high=VOCABULARY_SIZE, size=(2, MAX_INPUT_LEN), dtype=torch.long).to(device)
    # x_seg = torch.randint(low=1, high=2, size=(2, MAX_INPUT_LEN), dtype=torch.long).to(device)
    y1,y2= model(x,x_seg)
    print("--------")
    print(y1.size())
    print(y2.size())
