import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar
from utils.vocab import Vocab
from vars import *


def train_epoch(train_iter, test_iter, criterion, model, optimizer, vocab):
    epoch_loss_train, epoch_loss_eval = 0, 0

    # training network
    model.train()
    bar = Bar("\tprocessing train batches: ", max=len(train_iter))
    for i, batch in enumerate(train_iter):
        batch_loss_train = 0
        optimizer.zero_grad()

        story, highlight = batch
        story = story.to(device)
        highlight = highlight.to(device)

        output = model(story, highlight)
        for sequence, target in zip(output, highlight):
            batch_loss_train += criterion(sequence, target)
        batch_loss_train.backward()
        epoch_loss_train += batch_loss_train
        optimizer.step()
        bar.next()
    bar.finish()

    # evaluating network
    model.eval()
    for i, batch in enumerate(test_iter):
        batch_loss_eval = 0

        story, highlight = batch
        story = story.to(device)
        highlight = highlight.to(device)

        output = model(story, highlight)
        for sequence, target in zip(output, highlight):
            batch_loss_eval += criterion(sequence, target)
        epoch_loss_eval += batch_loss_eval

    return epoch_loss_train / len(train_iter), epoch_loss_eval / len(test_iter)


def train(train_iter, test_iter, criterion, model, optimizer, num_epochs=NUM_EPOCHS):
    train_loss, eval_loss = [], []
    vocab = Vocab('data/vocab', voc_size)
    model.to(device)

    # training loop
    for epoch in range(num_epochs):
        print("-------------------------------- Epoch nb {} --------------------------------".format(epoch))
        start = time.time()
        epoch_loss_train, epoch_loss_eval = train_epoch(train_iter, test_iter, criterion, model, optimizer, vocab)
        train_loss.append(epoch_loss_train.item())
        eval_loss.append(epoch_loss_eval.item())
        elapsed_time = time.time() - start
        print("\tloss train: {} | loss eval: {} | time: {}".format(epoch_loss_train, epoch_loss_eval, elapsed_time))

    plt.figure()
    plt.plot(range(num_epochs), train_loss)
    plt.plot(range(num_epochs), eval_loss)
    plt.show()
    return train_loss
