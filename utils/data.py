import torch
import os
import numpy as np
import torchtext.data as data
from pickle import load
from utils.vocab import Vocab
from vars import *
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence

# function to sort batch according to sequence length
def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X, y, lengths 

class Articles(torch.utils.data.Dataset):
    def __init__(self, test=False, data_dir="data", vocab_path='data/vocab'):
        super(Articles, self).__init__()
        '''Initialization'''
        self.vocab = Vocab(vocab_path, voc_size)
        self.tokenizer = data.get_tokenizer('basic_english')
        self.max_len_story = MAX_LEN_STORY
        self.max_len_highlight = MAX_LEN_HIGHLIGHT

        is_test = {
            False: os.path.join(data_dir, "trainchunk.pkl"),
            True: os.path.join(data_dir, "testchunk.pkl")
        }
        self.data_path = is_test.get(test, "Wrong set name.")

        with open(self.data_path, 'rb') as f:
            self.data = load(f)

    def __len__(self):
        '''return the number of articles'''
        return len(self.data)

    def __getitem__(self, idx):
        '''generates one sample of data'''
        X, y = self.data[idx]['story'], self.data[idx]['highlights']
        X_tokenized, y_tokenized = list(map(lambda x: self.tokenize(x), [X, y]))
        # X_tokenized, y_tokenized = list(map(lambda x: self.words_to_index(x), [X_tokenized, y_tokenized]))
        
        if len(X_tokenized) <= MAX_LEN_STORY:
            X_len = len(X_tokenized)
        else:
            X_len = MAX_LEN_STORY
        y_len = len(y_tokenized)
        X_padded = self.padding(X_tokenized)
        y_padded = self.padding(y_tokenized, sequence_type="highlight")
        return X_padded, y_padded, X_len, y_len

    def tokenize(self, sequence):
        '''tokenize a sequence'''
        tokenized_sequence = []
        tokenized_sequence.extend([token for token in self.tokenizer(sequence)])
        tokenized_sequence.append(STOP_TOKEN)
        return tokenized_sequence

    def words_to_index(self, tokenized_sequence):
        '''return list of index of tokens in the sequence'''
        return self.vocab.sequence_2_id(tokenized_sequence)

    def padding(self, sequence, sequence_type="story"):
        '''pad the sequence with the corresponding length'''
        if sequence_type == "story":
            max_len = self.max_len_story
        else:
            max_len = self.max_len_highlight
        if len(sequence) > max_len:
            sequence = sequence[:max_len]
        else:
            sequence += [PAD_TOKEN] * (max_len - len(sequence))
        return sequence


class Batcher:
    def __init__(self, story, highlight, vocab):
        self.vocab = vocab
        self.story = story
        self.highlight = highlight

        self.vocab_extended = deepcopy(vocab)
        self.vocab_extended.extend_vocab(story)

    def get_batch(self, get_vocab_extended=False):
        story = self.vocab.batch_tokens_to_id(self.story)
        highlight = self.vocab.batch_tokens_to_id(self.highlight)
        extra_zero = torch.zeros(batch_size, self.vocab_extended.vocab_size - voc_size + 1).clamp(min=1e-8)
        story_extended = self.vocab_extended.batch_tokens_to_id(self.story)
        highlight_extended = self.vocab_extended.batch_tokens_to_id(self.highlight)
        vocab_extended = self.vocab_extended if get_vocab_extended else None
        return story, highlight, extra_zero, story_extended, highlight_extended, vocab_extended

