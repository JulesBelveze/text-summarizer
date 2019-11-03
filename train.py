import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar
from utils.vocab import Vocab
from vars import *


def train_epoch(train_iter, criterion, model, optimizer, vocab):
    epoch_loss = 0
    model.train()
    for i, batch in enumerate(train_iter):

        optimizer.zero_grad()
        story, highlight = batch
        story = story.to(device)
        highlight = highlight.to(device)
        output = model(story, highlight)

        batch_loss = 0
        for sequence, target in zip(output, highlight):
            batch_loss += criterion(sequence, target)
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss
    return epoch_loss / len(train_iter)


def train(train_iter, criterion, model, optimizer, num_epochs=NUM_EPOCHS):
    train_loss = []
    vocab = Vocab('data/vocab', voc_size)
    model.to(device)
    for epoch in range(num_epochs):
        start = time.time()
        epoch_loss = train_epoch(train_iter, criterion, model, optimizer, vocab)
        train_loss.append(epoch_loss)
        elapsed_time = time.time() - start
        print("Epoch: {} | loss: {} | time: {}".format(epoch, epoch_loss, elapsed_time))

    plt.plot(range(num_epochs), train_loss)
    return train_loss
