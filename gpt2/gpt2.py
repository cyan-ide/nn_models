import transformer
from transformer import TransformerDecoderBlock
from torch import nn
import torch.nn.functional as F
import torch


MODEL_DIR = "<directory_to_store_models>" #cache for models

# GPT2 implementation following Andrej Karpathy tutorial / code (and some other web sources)
# Base transformer implementation inspired by tutorial of Peter Bloem

class GPT_Config: #default config as per GPT-2 paper, smallest GPT-2 version
    max_context_size: int = 1024 #gpt2 max input size in tokens
    vocabulary_size: int = 50257 #gpt2 vocabulry size
    embedding_size: int = 768 #embedding size
    layer_count: int = 12 #gpt2 number of transformer decoder layers
    head_count: int = 12 #gpt2 number of heads in self-attention for each decoder layer
    optimize_speed: bool = True #all sort of optimizations that might sacrifice a bit of accuracy for speed/memory improvements

#gpt2 model 
#predicts next token at every position of the input 
#(for training this is quite efficient, as we get many prediction in one go)
#(for inference this is a bit of a waste as we need only prediction for last token)
#x= (batch_size, tokens==max_tokens)
#y= (batch_size, expected tokens, y[i] = prediction if taking input only up to i-th token) (used during training)
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

        # [optional: optimization / memory] weight sharing scheme, reduces model size in memory
        if self.config.optimize_speed:
            self.transformer.wte.weight = self.lm_head.weight

        # [optional: optimization / performance ] custom init weights
        if self.config.optimize_speed:
            self.apply(self._init_weights) #iterate all layers and apply init_weights to each

    def forward(self,x, y=None):
        batch_size, token_count = x.size()
        assert token_count <= self.config.max_context_size, f"Max context token count is only {self.config.max_context_size}, your input size is: {token_count}"

        #1. encode input into embedding (for each "word" in the input we create an embedding)
        # print(x.size())
        token_emb = self.transformer.wte(x) # token embeddings of shape (batch_size, token_count, embedding_size)
        # print(token_emb.size())
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
        
        loss = None
        if y is not None: #compare prediction to expected value (if available) and calculate loss
            # cross_entropy(out,y) 
            # out = [batch_size * token_count, vocab_size] -> fold logits such that all samples from all batches are considered same
            # y = [batch_size * token_count]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return logits, loss

    #init model weights, values are taken from GPT-2 code by OpenAI
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'CUSTOM_INIT_SCALING'): #add additional scaling
                std *= (2 * self.config.layer_count) ** -0.5

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
