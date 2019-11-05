import torch
from torch.utils.data import DataLoader
from model import Encoder, Decoder, Seq2seq, Seq2seqAttention, AttnDecoder
from utils.data import Articles
from utils.vocab import Vocab
from train import train
from vars import *


if __name__ == "__main__":
    train_set = Articles(test=False)
    test_set = Articles(test=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)

    encoder = Encoder()
    attention_decoder = AttnDecoder()
    model = Seq2seqAttention(encoder, attention_decoder)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-3)
    loss_function = torch.nn.NLLLoss()

    train(train_loader, test_loader, loss_function, model, optimizer)
