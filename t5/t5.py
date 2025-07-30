import torch
from torch import nn
import torch.nn.functional as F

from transformer import TransformerEncoderBlock, TransformerDecoderBlock

#hyper-parameters from research paper
N_LAYERS = 12
EMBEDDING_SIZE = 768 #1024 #768 #1024 #768
N_ATTN_HEADS = 12
DROPOUT = 0.3

N_SEGMENTS = 3 #segments no for segment embedding
MAX_INPUT_LEN = 512
VOCABULARY_SIZE = 30000

class T5_Config: #default config as per T5 paper, some settings taken from BERT/GPT-2 to speedup implementation and testing
    max_context_size: int = 1024 # max input size in tokens
    vocabulary_size: int = 50257 # vocabulry size
    embedding_size: int = 768 #1024 #768 #1024 #768 #embedding size
    layer_count: int = 12 # number of transformer encoder/decoder layers
    head_count: int = 12 # number of heads in self-attention for each decoder layer
    optimize_speed: bool = True #all sort of optimizations that might sacrifice a bit of accuracy for speed/memory improvements


class T5(nn.Module):
    def __init__(self, config, dropout, device, max_pool= True, pretraining_type='pred_word'):
        super().__init__()
        self.config = config
        self.device = device
        self.max_pool = max_pool

        #embeddings which are used to wrap the regular input
        self.token_embedding = nn.Embedding(num_embeddings=self.config.vocabulary_size, embedding_dim=self.config.embedding_size, padding_idx=0)
        self.position_embedding = nn.Embedding(num_embeddings=self.config.max_context_size, embedding_dim=self.config.embedding_size)

        #transformer encoder blocks
        transformer_blocks = []
        for i in list(range(self.config.layer_count)):
            transformer_blocks.append(TransformerEncoderBlock(config=self.config))
        self.transformer_encoder_blocks = nn.Sequential(*transformer_blocks)

        #transformer decoder blocks
        transformer_decoder_blocks_list = []
        for i in list(range(self.config.layer_count)):
            transformer_decoder_blocks_list.append(TransformerDecoderBlock(config=self.config))
        self.transformer_decoder_blocks = nn.Sequential(*transformer_decoder_blocks_list)

        #dropout (done after embeddings for stabilising output)
        self.dropout = nn.Dropout(dropout)

        #learning heads for pre-training (MLM/ masked learning model same as BERT/RoBERTa)
        self.lm_head_1 = nn.Linear(in_features=self.config.embedding_size, out_features=self.config.vocabulary_size, bias=False) #(1st pre-training type): predict missing word from vocabulary

    def forward(self, x, y=None): #, x_segment_mask #TEMP remove NSP
        #embeddings
        tokens = self.token_embedding(x)

        batch_size, token_count, embedding_size = tokens.size()  #token_count == amount of input "words"
        positions = torch.arange(token_count,device=self.device) #actual value == sequence of word positions [0...#word_count]
        positions = self.position_embedding(positions)
        positions = positions[None, :, :] #add batch size to tensor (set as 1)
        positions = positions.expand(batch_size, token_count, embedding_size) # reshape to final embedding size, increasing to input batch size 

        x = tokens + positions

        x= self.dropout(x)
        
        #2. go through all transformer blocks
        x= self.transformer_encoder_blocks(x)
        x= self.transformer_decoder_blocks((x,x))
        x= self.dropout(x[0])

        #3. learning head 
        y1 = self.lm_head_1(x) #pre-training #1, predict probs for every token (ie. predict masked token) # [<batch_n>, 128, 30000]

        return y1 #output: un-normalized values (ie. later the crossEntropy uses non-normalized, softmax is needed only for predictions)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    t5_config = T5_Config()
    model = T5(config=t5_config
                , dropout=DROPOUT
                , device= device).to(device)


    #test input (2 samples, each 9 "words" encoded with some dictonary)
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0]
                    , [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)

    y1= model(x)
    print("--------")
    print(y1.size())
