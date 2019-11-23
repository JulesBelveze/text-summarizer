import torch
import argparse
from torch.utils.data import DataLoader
from model import Encoder, Decoder, Seq2seq, Seq2seqAttention, AttnDecoder
from utils.data import Articles
from utils.utils import load_ckp
from train import train
from vars import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do-train', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--ckpt", default=None)
    return parser.parse_args()


def run(do_train, ckpt):
    train_set = Articles(test=False)
    test_set = Articles(test=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)

    encoder = Encoder()
    attention_decoder = AttnDecoder()
    model = Seq2seqAttention(encoder, attention_decoder)
    model.to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    loss_function = torch.nn.NLLLoss()

    if ckpt:
        model, optimizer, epoch = load_ckp(checkpoint_path=ckpt, model=model, optimizer=optimizer)
    else:
        epoch = 0

    if do_train:
        train(train_loader, test_loader, loss_function, model, optimizer, epoch, num_epochs=epoch+800)




if __name__ == "__main__":
    args = parse_args()
    run(**vars(args))
