
from transformers import BertTokenizer
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import torch
import torch.nn.functional as F
from datasets import load_dataset

HUGGINGFACE_DATA_CACHE = "/home/adam/huggingface_data_cache/"
MODEL_DIR = "/home/adam/huggingface_models/" #cache for models

MICRO_BATCH_SIZE = 16 #256 #64 #16
GRAD_ACCUM = True
GRAD_ACCUM_STEPS = 21


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print("Using device: {}".format(device))

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
 
print(type(config.hidden_act))
print(config.layer_norm_eps)
print(config.initializer_range)
model = BertForMaskedLM(config).to(device)
print('No of parameters: ', model.num_parameters())

print(model)
print("=============")

sd_hf = model.state_dict()
for k, v in sd_hf.items():
  print(k, v.shape)
print("=============")
#---------------------------------

def tokenize_function(examples,tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", cache_dir=MODEL_DIR)

dataset = load_dataset("bookcorpus/bookcorpus", cache_dir=HUGGINGFACE_DATA_CACHE, split='train[:20000]')
tokenized_dataset = dataset.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

data_loader = torch.utils.data.DataLoader(tokenized_dataset['input_ids'], batch_size=MICRO_BATCH_SIZE, collate_fn=data_collator)
data_iter = iter(data_loader)

# -------------------------------


    # ,learning_rate=
    # ,weight_decay=0.01
    # ,adam_beta1=0.9
    # ,adam_beta2=0.999
    # ,adam_epsilon=1e-8
optimizer = AdamW(model.parameters(), lr=5e-05, betas=(0.9, 0.999), eps=1e-8,weight_decay=0.01)
loss_fn = CrossEntropyLoss()
loss_fn = F.cross_entropy
model.train()
step_no = 0
micro_step_no=0
acc_accum=0
loss=0
batch_time=0
tokens_per_sec=0

if GRAD_ACCUM:
  for epoch in range(1):
    for micro_batch in range(59): ##20k / batch size*grad_accum
      for micro_batch in range(GRAD_ACCUM_STEPS): 
        # for batch in data_loader:
        batch = next(data_iter)
        batch = {k: v.to(device) for k, v in batch.items()}
        # print(batch)
        # outputs = model(**batch)
        outputs = model(batch['input_ids'])
        # print(outputs.logits.shape) #torch.Size([16, 128, 50000])
        # print(batch['labels'].shape) #torch.Size([16, 128])
        # loss = loss_fn(outputs.logits.transpose(1,2), batch['labels'])
        loss = loss_fn(outputs.logits.transpose(1, 2), batch['labels'],ignore_index=-100)
        loss.backward()
        micro_step_no+=1
      optimizer.step()
      optimizer.zero_grad()
      print("MicroStep: {:5d}, Step: {:5d}, batch_acc: {}, loss: {:.6f}, time(ms): {:.2f}, token/sec:{:.2f}".format(micro_step_no,step_no, acc_accum,loss.detach(), batch_time*1000, tokens_per_sec))
      step_no+=1
else:
  for epoch in range(1):
    # for batch in data_loader:
    for micro_batch in range(1250): #20k / batch size
      batch = next(data_iter)
      batch = {k: v.to(device) for k, v in batch.items()}
      # print(batch)
      # outputs = model(**batch)
      outputs = model(batch['input_ids'])
      # print(outputs.logits.shape) #torch.Size([16, 128, 50000])
      # print(batch['labels'].shape) #torch.Size([16, 128])
      # loss = loss_fn(outputs.logits.transpose(1,2), batch['labels'])
      loss = loss_fn(outputs.logits.transpose(1, 2), batch['labels'],ignore_index=-100)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      print("Step: {:5d}, batch_acc: {}, loss: {:.6f}, time(ms): {:.2f}, token/sec:{:.2f}".format(step_no, acc_accum,loss.detach(), batch_time*1000, tokens_per_sec))
      step_no+=1


