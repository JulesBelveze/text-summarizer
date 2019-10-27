PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
START_TOKEN = '[START]'
STOP_TOKEN = '[STOP]'


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
                token, _ = line.split()
                self.word_to_id[token] = token_counter
                self.id_to_word[token_counter] = token
                token_counter += 1

                if token_counter >= self.vocab_size:
                    break

    def word_2_id(self, word: str) -> int:
        if word not in self.word_to_id:
            return self.word_to_id[UNKNOWN_TOKEN]
        return self.word_to_id[word]

    def id_2_word(self, id: int) -> str:
        if id not in self.id_to_word:
            return UNKNOWN_TOKEN
        return self.id_to_word[id]


