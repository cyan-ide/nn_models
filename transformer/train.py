import torch
import torch.nn.functional as F

from datasets import load_dataset
from transformers import BertTokenizer
import tqdm

from transformer import Classifier_Transformer

TRAIN_SIZE = 25000 #10000 #25000
TEST_SIZE = 25000 #10000 #25000

VOCAB_SIZE = 30522
MAX_INPUT_SIZE = 512

EMBEDDING_SIZE=512
ATTENTION_HEADS=8
TRANSFORMER_BLOCK_NO = 6
NUM_CLS = 2 #positive / negative

#training parameters
NUM_EPOCHS = 1 #80
LEARNING_RATE = 0.0001
LEARNING_RATE_WARMUP = 10_000
BATCH_SIZE = 4

def tokenize(label, line):
    return line.split()

def preprocess_function(examples, tokenizer):
    out = tokenizer(examples['text'], padding="max_length", max_length=MAX_INPUT_SIZE, truncation=True)
    # out['label_tensor'] = [torch.tensor([int(not(x)),x]) for x in examples['label']]
    out['label_tensor'] = [[int(not(x)),x] for x in examples['label']]
    return out

if __name__ == "__main__":
    #test simple transformer model using IMDB data
    #my test run got ~80% accuracy, using full train / test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    #test input (2 samples, each 9 "words" encoded with some dictonary)
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0]
                    , [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)


    #datset
    imdb_train,imdb_test = load_dataset("imdb", split=['train', 'test'])
    imdb_train = imdb_train.shuffle().select(range(TRAIN_SIZE)) 
    imdb_test = imdb_test.shuffle().select(range(TEST_SIZE)) 

    #use bert to tokenize samples
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_iter = imdb_train.map(preprocess_function,fn_kwargs= {'tokenizer':tokenizer},batched=True)
    test_iter = imdb_test.map(preprocess_function,fn_kwargs= {'tokenizer':tokenizer},batched=True)
    
    #check dataset stats
    tr_pos_lbl = sum(train_iter['label'])
    tst_pos_lbl = sum(test_iter['label'])
    print("Train set size: {} / pos: {} neg: {}".format(len(train_iter),tr_pos_lbl,TRAIN_SIZE-tr_pos_lbl ))
    print("Test set size: {} / pos: {} neg: {}".format(len(test_iter),tr_pos_lbl,TEST_SIZE-tst_pos_lbl ))

    print("Transformers robots in disguise!")

    model = Classifier_Transformer(embedding_size=EMBEDDING_SIZE
                            , heads_no=ATTENTION_HEADS
                            , transformer_block_no = TRANSFORMER_BLOCK_NO
                            , vocabulary_size= VOCAB_SIZE
                            , max_input_length=MAX_INPUT_SIZE
                            , num_classes = NUM_CLS
                            , device=device).to(device)

    optimizer = torch.optim.Adam(lr=LEARNING_RATE, params=model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (LEARNING_RATE_WARMUP / BATCH_SIZE), 1.0))

    # training loop
    seen = 0
    for e in range(NUM_EPOCHS):
        print("EPOCH: {}".format(e))
        #do train pass
        model.train(True)
        print("TRAINING:")
        for batch in tqdm.tqdm(train_iter.batch(batch_size=2)): #pass all batches
            optimizer.zero_grad()
            x = torch.tensor(batch['input_ids']).to(device) #batch['text']
            label = torch.tensor(batch['label']).to(device)
            out = model(x)
            loss = F.nll_loss(out, label) #calculate loss
            loss.backward() #backprop
            optimizer.step()
            lr_scheduler.step()

        #do validation pass
        print("VALIDATION:")
        with torch.no_grad():
            model.train(False)
            total_samples, correct_preds= 0.0, 0.0
            #do test pass
            for batch in tqdm.tqdm(test_iter.batch(batch_size=2)):
                x = torch.tensor(batch['input_ids']).to(device)
                label = torch.tensor(batch['label']).to(device)
                out = model(x)
                out = out.argmax(dim=1) #return prob in [p1,p2], argmax return index which pred is bigger, at idx= 0 or 1 (p1 or p2)
                total_samples += float(x.size(0)) #no. samples in the batch
                correct_preds += float((label == out).sum().item()) #add for all items in the batch matching predictions
            acc = correct_preds / total_samples
            print("VALID ACCURACY: {}".format(acc))
    