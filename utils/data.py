import torch
import os
import numpy as np
import torchtext.data as data
from pickle import load
from utils.vocab import Vocab
from vars import *


class Articles(torch.utils.data.Dataset):
    def __init__(self, test=False, data_dir="data", vocab_path='data/vocab'):
        super(Articles, self).__init__()
        '''Initialization'''
        self.vocab = Vocab('data/vocab', voc_size)
        self.tokenizer = data.get_tokenizer('basic_english')
        self.max_len_story = MAX_LEN_STORY
        self.nax_len_highlight = MAX_LEN_HIGHLIGHT

        is_test = {
            False: os.path.join(data_dir, "train_small.pkl"),
            True: os.path.join(data_dir, "test_small.pkl")
        }
        self.data_path = is_test.get(test, "Wrong set name.")

        with open(self.data_path, 'rb') as f:
            self.data = load(f)

    def __len__(self):
        '''return the number of articles'''
        return len(self.data)

    def __getitem__(self, idx):
        '''generates one sample of data'''
        X, y = self.data[idx]['story'], self.data[idx]['highlights'][0]
        X_tokenized, y_tokenized = list(map(lambda x: self.tokenize(x), [X, y]))
        X_tokenized, y_tokenized = list(map(lambda x: self.words_to_index(x), [X_tokenized, y_tokenized]))
        X_padded = self.padding(X_tokenized)
        y_padded = self.padding(y_tokenized, sequence_type="highlight")
        return X_padded, y_padded

    def tokenize(self, sequence):
        '''tokenize a sequence'''
        tokenized_sequence = [START_TOKEN]
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
            max_len = self.nax_len_highlight
        len_sequence = min(max_len, len(sequence))
        data_pad = np.pad(sequence, (0, max(0, max_len - len_sequence)), 'constant',
                          constant_values=(self.vocab.word_2_id(PAD_TOKEN)))[:max_len]
        return data_pad