import numpy as np
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
        if word not in self.word_to_id:
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

    def one_hot_encode(self, token_id: int) -> List[int]:
        one_hot = [0] * self.vocab_size
        one_hot[int(token_id)] = 1
        return one_hot

    def one_hot_encode_sequence(self, sequence):
        # Encode each word in the sentence
        encoding = np.array([self.one_hot_encode(token_id) for token_id in sequence])

        # Reshape encoding s.t. it has shape (num words, vocab size)
        encoding = encoding.reshape(encoding.shape[0], encoding.shape[1])
        return encoding

    def one_hot_encode_tensor(self, tensor):
        encoding = np.array([self.one_hot_encode_sequence(sequence) for sequence in tensor])
        return torch.LongTensor(encoding)
