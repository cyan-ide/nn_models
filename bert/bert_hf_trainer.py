
HUGGINGFACE_DATA_CACHE = "/home/adam/huggingface_data_cache/"
MODEL_DIR = "/home/adam/huggingface_models/" #cache for models

#-------------------

# Load the tokenizer
from transformers import BertTokenizer, LineByLineTextDataset
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", cache_dir=MODEL_DIR)

sentence = 'A tokenizer is in charge of preparing the inputs for a model.'

encoded_input = tokenizer.tokenize(sentence)
print(encoded_input)

#-------------------

from datasets import load_dataset
from datasets import Dataset

dataset = load_dataset("bookcorpus/bookcorpus", cache_dir=HUGGINGFACE_DATA_CACHE, split='train[:20000]')

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

print('No. of lines: ', len(dataset)) # No of lines in your datset

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
 
model = BertForMaskedLM(config)
print('No of parameters: ', model.num_parameters())


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# --------------------------------

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./hf_output/',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=1
    ,learning_rate=5e-05
    ,weight_decay=0.01
    ,adam_beta1=0.9
    ,adam_beta2=0.999
    ,adam_epsilon=1e-8
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset, 
)

#---------------------------------

trainer.train()
exit()
#trainer.save_model('./hf_output/')

# -------------------------------