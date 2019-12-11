import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embedding_dim = 128
MAX_LEN_STORY = 300
hidden_dim = 128
voc_size = 20000
MAX_LEN_HIGHLIGHT = 100

NUM_EPOCHS=100
batch_size=16
lr = 0.15

## tokens variable
PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
START_TOKEN = '[START]'
STOP_TOKEN = '[STOP]'
SENTENCE_START = '<s>' # start sentence in highlight
SENTENCE_END = '</s>' # end sentence in highlight
