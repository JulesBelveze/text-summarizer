import numpy as np
import torch
from typing import List
from vars import *


class Vocab(object):
    def __init__(self, vocab_file, vocab_size):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = vocab_size

        token_counter = 0
        for token in [PAD_TOKEN, UNKNOWN_TOKEN, START_TOKEN, STOP_TOKEN]:
            self.word_to_id[token] = token_counter
            self.id_to_word[token_counter] = token
            token_counter += 1

        with open(vocab_file, 'r') as f:
            for line in f:
                try:
                    token, _ = line.split()
                    self.word_to_id[token] = token_counter
                    self.id_to_word[token_counter] = token
                    token_counter += 1
                    if token_counter >= self.vocab_size:
                        break
                except ValueError as e:
                    print("Error in line {}: {}".format(token_counter - 4, e))
                    pass

    def word_2_id(self, word: str) -> int:
        if word == SENTENCE_END:
            return self.word_to_id["."]
        elif word not in self.word_to_id:
            return self.word_to_id[UNKNOWN_TOKEN]
        return self.word_to_id[word]

    def id_2_word(self, id: int) -> str:
        if id not in self.id_to_word:
            return UNKNOWN_TOKEN
        return self.id_to_word[id]

    def sequence_2_id(self, sequence: List[str]) -> List[int]:
        return [self.word_2_id(token) for token in sequence]

    def ids_to_sequence(self, ids):
        return [self.id_2_word(id) for id in ids]

    def extend_vocab(self, batch_stories):
        counter = self.vocab_size
        for story in batch_stories:
            for token in story:
                if not token in self.word_to_id:
                    counter += 1
                    self.word_to_id[token] = counter
                    self.id_to_word[counter] = token
        self.vocab_size = counter

    def batch_tokens_to_id(self, batch_seq):
        list_ids = []
        for seq in batch_seq:
            if len([self.word_2_id(token) for token in seq if token != SENTENCE_START]) == 0:
                print(seq)
            list_ids.append([self.word_2_id(token) for token in seq if token != SENTENCE_START])
        array_ids = np.array(list_ids)
        return torch.LongTensor(np.transpose(array_ids))
