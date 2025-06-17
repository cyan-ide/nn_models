"""
download and tokenize and save to shards the imdb dataset
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm
from transformers import BertTokenizer, BertTokenizerFast
from typing import List, Tuple, Dict
import random as rd
import re
from math import ceil

# ------------------------------------------
local_dir = "edu_fineweb10B"
shard_size = int(259740*385) #100M tokens #for testing ~1M token, our samples are 385 however, so 2,597*385=999,845;  #int(1e8) # 100M tokens per shard, total of 100 shards

# create the cache the local directory if it doesn't exist yet
SHARD_DATA_CACHE_DIR = "/mnt/data/backup/adam/llm_train_data/bookcorpus_tokenized/" #os.path.join(os.path.dirname(__file__), local_dir)
SHARD_DATA_CACHE_DIR = "/mnt/data/backup/adam/llm_train_data/imdb_tokenized_tst/"
HUGGINGFACE_DATA_CACHE = "/mnt/data/backup/adam/huggingface_data_cache/"
MODEL_DIR = "/mnt/ssd/home/adam/huggingface_models/" #cache for models
#os.makedirs(DATA_CACHE_DIR, exist_ok=True)


SEQ_LENGTH=128
SAMPLE_LENGTH=SEQ_LENGTH+SEQ_LENGTH+SEQ_LENGTH+1 # =385 = "masked sentence" + "mask_labels" +"segment_mask" + is_next_label
MAX_SENTENCE_WORDS= SEQ_LENGTH//2 # Maximum number of words in one sentence (ie. seq len / 2 as we have max 2 sentences)
MASK_TO_TOKEN_RATIO=0.15 #Proportion of tokens to mask in each sentence. = 15% according to paper
DELIMITERS=".,;:!?" #Punctuation marks for sentence splitting.
LOWER_CASE=True #Flag indicating whether to convert sentences to lowercase.

MASK_TOKEN = '[MASK]'
PAD_TOKEN = '[PAD]'
CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
PAD_IDX = 0



# download the dataset
imdb_dataset = load_dataset("nocode-ai/imdb-movie-reviews", split="train", cache_dir=HUGGINGFACE_DATA_CACHE)


tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased", cache_dir=MODEL_DIR)
# eot = tokenizer._special_tokens['<|endoftext|>'] # end of text token

#get tokenizer vocabulary (in bert pre-training there is 10% chance masked word will be raplced by random)
vocab = tokenizer.get_vocab()
BERT_VOCABULARY = [
    word for word in vocab.keys()
    if not (re.compile(r'\[unused\d+\]').match(word)
            or word in ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
            or not re.compile(r'^[a-zA-Z]+$').match(word))
]


def book2tokens(doc):
    samples = []
    all_tokens_np = np.zeros(0) #np.empty(dtype=np.uint16)
    sentences = split_sentences(doc, DELIMITERS, MAX_SENTENCE_WORDS)

    last_sentence = len(sentences) - 1
    for i, sent_A in enumerate(sentences):
        if rd.random() <= 0.5 and i != last_sentence:
            is_next = 1
            sent_B = sentences[i + 1]
        else:
            is_next = 0
            sent_B = read_random_sentence()

        sent_A, sent_B = custom_std(sent_A, LOWER_CASE), custom_std(sent_B, LOWER_CASE)

        sent_A, label_A = mask_sentence(sent_A)
        sent_B, label_B = mask_sentence(sent_B)

        bert_label = ([PAD_TOKEN] + label_A + [PAD_TOKEN] + label_B) + [PAD_TOKEN]

        sent_A = [CLS_TOKEN] + sent_A + [SEP_TOKEN]
        sent_B = sent_B + [SEP_TOKEN]

        segment_label = [1 for _ in range(len(sent_A))] + [2 for _ in range(len(sent_B))]

        sequence = sent_A + sent_B

        padding = [PAD_TOKEN for _ in range(SEQ_LENGTH - len(sequence))]
        sequence.extend(padding), bert_label.extend(padding), segment_label.extend([PAD_IDX] * len(padding))

        bert_input = tokenizer.convert_tokens_to_ids(sequence)
        bert_label = tokenizer.convert_tokens_to_ids(bert_label)

        assert len(bert_input) == len(bert_label) == len(segment_label)

        tokens_np = np.array(bert_input+bert_label+segment_label+[is_next]) #len = 128+128+128+is_next = 385x int
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
        tokens_np_uint16 = tokens_np.astype(np.uint16)
        if len(all_tokens_np)==0:
            all_tokens_np = tokens_np_uint16
        else:
            all_tokens_np=np.concatenate((all_tokens_np, tokens_np_uint16))
        samples.append({"bert_input": bert_input,
                        "bert_label": bert_label,
                        "segment_label": segment_label,
                        "is_next": is_next})
    return all_tokens_np

def mask_sentence(sent: str) -> Tuple[List[str], List[str]]:
        """
        Masks tokens in a sentence.

        Args:
            sent (str): The input sentence.

        Returns:
            Tuple[List[str], List[str]]: Tuple containing masked tokens and target sequence.
        """
        tokens = tokenizer.tokenize(sent)[:MAX_SENTENCE_WORDS - 2]
        num_masked = ceil(MASK_TO_TOKEN_RATIO * len(tokens))
        masked_indices = rd.sample(range(len(tokens)), num_masked)

        target_sequence = [PAD_TOKEN] * len(tokens)
        for idx in masked_indices:
            target_sequence[idx] = tokens[idx]

            p = rd.random()
            p=0.75 #temp
            # 80% of times put [mask] token, 10% put random word, 10% no change
            if p < 0.8:
                tokens[idx] = MASK_TOKEN

                next_idx = idx + 1 #if word is split into more than 1 token, mask all (not exactly how its described in paper)
                while next_idx < len(tokens) and tokens[next_idx].startswith("##"):
                    target_sequence[next_idx] = tokens[next_idx]
                    tokens[next_idx] = MASK_TOKEN
                    next_idx += 1
            elif p <= 0.9:
                tokens[idx] = rd.choice(BERT_VOCABULARY)
            else:
                pass
        return tokens, target_sequence

def read_random_sentence() -> str:
    """
    Reads a random sentence from the dataset.

    Returns:
        str: A randomly selected sentence.
    """
    idx = rd.randint(0, len(imdb_dataset) - 1)
    doc = imdb_dataset[idx]['review'].strip()
    # with open(self.filenames[idx], "r") as file:
    #     doc = file.read().strip()

    sentences = split_sentences(doc, DELIMITERS, MAX_SENTENCE_WORDS)
    return sentences[rd.randint(0, len(sentences) - 1)]

def split_sentences(text: str, delimiters: str, max_words: int) -> List[str]:
    """
    Splits text into sentences based on various strategies.

    Args:
        text (str): The input text to be split.
        delimiters (str, optional): Punctuation marks to split sentences.
        max_words (int, optional): The maximum number of words per split.

    Returns:
        List[str]: List of split sentences.
    """
    def split_text_by_punctuation(text: str, delimiters: str, max_words: int) -> List[str]:
        """
        Splits text into sentences based on specified punctuation marks and a maximum number of words per split.

        Args:
            text (str): The input text to be split.
            delimiters (str): Punctuation marks to split sentences.
            max_words (int): The maximum number of words per split.

        Returns:
            List[str]: List of split sentences.
        """
        sentences = []

        for sentence in re.split(r'(?<=[{}])'.format(re.escape(delimiters)), text):
            sentence = sentence.strip()
            if sentence:
                sentences.extend(split_text_by_maximum_word_count(sentence, max_words))

        return sentences


    def split_text_by_maximum_word_count(text: str, max_words: int) -> List[str]:
        """
        Splits text into smaller strings, each with a predetermined word count.

        Args:
            text (str): The input text to be split.
            max_words (int): The maximum number of words per split.

        Returns:
            List[str]: List of split sentences.
        """
        words = text.split()
        result_sentences = []

        while words:
            result_sentences.append(" ".join(words[:max_words]))
            words = words[max_words:]

        return result_sentences

    if rd.random() < 0.75:
        return split_text_by_punctuation(text, delimiters, max_words)
    else:
        return split_text_by_maximum_word_count(text, max_words)

def custom_std(sentence: str, lower_case: bool = False) -> str:
    """ Remove HTML line-break tags and lowercase the sentence."""
    sentence = re.sub("<br />", " ", sentence).strip()
    if LOWER_CASE:
        sentence = sentence.lower()
    return sentence


def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(tokenizer.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


#imdb dataset size = 50,000 samples
shard_index = 0
# preallocate buffer to hold current shard
all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
token_count = 0
progress_bar = None
sample_count = 0
for sample in imdb_dataset:
    tokens = book2tokens(sample['review'])
    # is there enough space in the current shard for the new tokens?
    if token_count + len(tokens) < shard_size:
        # simply append tokens to current shard
        all_tokens_np[token_count:token_count+len(tokens)] = tokens
        token_count += len(tokens)
        # update progress bar
        if progress_bar is None:
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(len(tokens))
    else:
        # write the current shard and start a new one
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(SHARD_DATA_CACHE_DIR, f"imdb_{split}_{shard_index:06d}")
        # split the document into whatever fits in this shard; the remainder goes to next one
        remainder = shard_size - token_count
        progress_bar.update(remainder)
        all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
        write_datafile(filename, all_tokens_np)
        shard_index += 1
        progress_bar = None
        # populate the next shard with the leftovers of the current doc
        all_tokens_np= np.empty((shard_size,), dtype=np.uint16) #empty prev array
        all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
        token_count = len(tokens)-remainder
        # update progress bar
        if progress_bar is None:
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(token_count)
        
    if sample_count%10000==0:
        print("sample_count: {}".format(sample_count))
    sample_count +=1


# write any remaining tokens as the last shard
if token_count != 0:
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(SHARD_DATA_CACHE_DIR, f"imdb_{split}_{shard_index:06d}")
    write_datafile(filename, all_tokens_np[:token_count])


