import transformer
from transformer import TransformerDecoderBlock
from torch import nn
import torch.nn.functional as F
import torch


MODEL_DIR = "/mnt/ssd/home/adam/huggingface_models/" #cache for models

#GPT2 implementation following Andrej Karpathy tutorial / code (and some other web sources)
#Base transformer implementation inspired by tutorial of Peter Bloem

class GPT_Config: #default config as per GPT-2 paper, smallest GPT-2 version
    max_context_size: int = 1024 #gpt2 max input size in tokens
    vocabulary_size: int = 50257 #gpt2 vocabulry size
    embedding_size: int = 768 #embedding size
    layer_count: int = 12 #gpt2 number of transformer decoder layers
    head_count: int = 12 #gpt2 number of heads in self-attention for each decoder layer


class GPT_2(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        #declare the entire GPT-2 architecture here!
        #layers names match those of HuggingFace for easier weight loading
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(embedding_dim=self.config.embedding_size, num_embeddings=self.config.vocabulary_size),
            wpe = nn.Embedding(embedding_dim=self.config.embedding_size, num_embeddings=self.config.max_context_size),
            h = nn.ModuleList([TransformerDecoderBlock(config) for _ in range(self.config.layer_count)]),
            ln_f = nn.LayerNorm(self.config.embedding_size),
        ))
        self.lm_head = nn.Linear(self.config.embedding_size, self.config.vocabulary_size, bias=False)

    def forward(self,x):
        batch_size, token_count = x.size()
        assert token_count <= self.config.max_context_size, f"Max context token count is only {self.config.max_context_size}, your input size is: {token_count}"

        #1. encode input into embedding (for each "word" in the input we create an embedding)
        token_emb = self.transformer.wte(x) # token embeddings of shape (batch_size, token_count, embedding_size)

        #2. additionally calculate positional embedding, ie. sequential numbers for position of every "word" in the input
        positions = torch.arange(0, token_count, dtype=torch.long, device=x.device) # shape (token_count)
        position_emb = self.transformer.wpe(positions) # position embeddings of shape (token_count, embedding_size)
        #3. add embeddings
        x = token_emb + position_emb
        #4. Go through all transformer blocks
        for block in self.transformer.h:
            x = block(x)
        #4. Do final layernorm and classifier head that will output probabilities for each vocabulary word (ie. next word)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (batch_size, token_count, vocabulary_size)
        return logits

    @classmethod
    def from_pretrained(cls, model_type): #import weights from huggingface (for testing compatibility with other GPT2 implementation)
        assert model_type in {'gpt2', 'gpt2-medium','gpt2-large','gpt2-xl'} #possible names of model in the huggingface repository
        
        #init local model
        config = GPT_Config()
        model = GPT_2(config)

        #get (my) local model weights
        local_state_dict = model.state_dict()
        local_state_dict_keys = local_state_dict.keys()
        #skip some irrelevant parts of the model
        local_state_dict_keys = [k for k in local_state_dict_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
        
        #get huggingface weigths
        from transformers import GPT2LMHeadModel
        hf_model = GPT2LMHeadModel.from_pretrained(model_type, cache_dir=MODEL_DIR)
        hf_state_dict = hf_model.state_dict()
        hf_state_dict_keys = hf_state_dict.keys()
        #skip irrelevant parts of the model
        hf_state_dict_keys = [k for k in hf_state_dict_keys if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        hf_state_dict_keys = [k for k in hf_state_dict_keys if not k.endswith('.attn.bias')] # same, just the mask (buffer)

        #some layers will need to be transposed as they are stored in the other model in different way
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight'] 


        #test, print hf dict names to see what we have to match
        # for k,v in hf_state_dict.items():
        #     print(k, v.shape)

        # print ("===============================================")
        # #test, print hf dict names to see what we have to match
        # for k,v in local_state_dict.items():
        #     print(k, v.shape)

        #check if both models match in size / layers etc.
        assert len(hf_state_dict_keys) == len(local_state_dict_keys), f"mismatched keys: {len(hf_state_dict_keys)} != {len(local_state_dict_keys)}"

        #do all the copying from hf_state_dict to local_state_dict
        for k in hf_state_dict_keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert hf_state_dict[k].shape[::-1] == local_state_dict[k].shape
                with torch.no_grad():
                    local_state_dict[k].copy_(hf_state_dict[k].t())
            else:
                # vanilla copy over the other parameters
                assert hf_state_dict[k].shape == local_state_dict[k].shape
                with torch.no_grad():
                    local_state_dict[k].copy_(hf_state_dict[k])

        return model



model = GPT_2.from_pretrained('gpt2')


#test model
num_return_sequences = 5
max_length = 30

model.eval()
model.to('cuda')

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to('cuda')

# generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
