import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar
from utils.vocab import Vocab
from utils.utils import save_checkpoint
from vars import *
from copy import deepcopy


def train_epoch(train_iter, test_iter, criterion, model, optimizer, vocab):
    epoch_loss_train, epoch_loss_eval, epoch_acc_train, epoch_acc_eval = 0, 0, 0, 0

    # training network
    model.train()
    bar = Bar("\tprocessing train batches: ", max=len(train_iter))
    for i, batch in enumerate(train_iter):
        batch_loss_train, batch_acc_train = 0, 0
        optimizer.zero_grad()

        story, highlight = batch
        vocab_extended = deepcopy(vocab)
        vocab_extended.extend_vocab(story)

        story_extended = vocab_extended.batch_tokens_to_id(story)
        highlight_extended = vocab_extended.batch_tokens_to_id(highlight)
        extra_zeros = torch.zeros(batch_size, vocab_extended.vocab_size - voc_size + 1).clamp(min=1e-8)
        story = vocab.batch_tokens_to_id(story)
        highlight = vocab.batch_tokens_to_id(highlight)


        story = story.to(device)
        highlight = highlight.to(device)

        story_extended = story_extended.to(device)
        highlight_extended = highlight_extended.to(device)
        extra_zeros = extra_zeros.to(device)

        output = model(story, highlight, story_extended, extra_zeros)
        for predicted, target in zip(output, highlight_extended):
            predicted = predicted.to(device)
            target = target.to(device)
            batch_loss_train += criterion(predicted, target)

            batch_acc_train += (target == predicted.argmax(dim=1)).sum().item() / MAX_LEN_HIGHLIGHT

        # propagating loss
        batch_loss_train.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        epoch_loss_train += batch_loss_train.item()
        optimizer.step()

        # computing accuracy
        batch_acc_train /= batch_size
        epoch_acc_train += batch_acc_train
        bar.next()
    bar.finish()
    # showing last output
    target_sequence = " ".join(vocab_extended.ids_to_sequence(target.tolist()))
    predicted_tokens = predicted.argmax(dim=1).tolist()
    predicted_sequence = " ".join(vocab_extended.ids_to_sequence(predicted_tokens))
    print("Targeted sentence: {}".format(target_sequence))
    print("Predicted sentence: {}\n".format(predicted_sequence))

    # evaluating network
    with torch.no_grad():
        for i, batch in enumerate(test_iter):
            batch_loss_eval, batch_acc_eval = 0, 0

            story, highlight = batch
            vocab_extended = deepcopy(vocab)
            vocab_extended.extend_vocab(story)
            story_extended = vocab_extended.batch_tokens_to_id(story)
            extra_zeros = torch.zeros(batch_size, vocab_extended.vocab_size - voc_size + 1).clamp(min=1e-8)
            story = vocab.batch_tokens_to_id(story)
            highlight = vocab.batch_tokens_to_id(highlight)


            story = story.to(device)
            highlight = highlight.to(device)

            story_extended = story_extended.to(device)
            highlight_extended = highlight_extended.to(device)
            extra_zeros = extra_zeros.to(device)

            output = model(story, highlight, story_extended, extra_zeros)
            for predicted, target in zip(output, highlight):
                batch_loss_eval += criterion(predicted, target).item()
                batch_acc_eval += (target == predicted.argmax(dim=1)).sum().item() / MAX_LEN_HIGHLIGHT

            # computing epoch loss
            epoch_loss_eval += batch_loss_eval

            # computing accuracy
            batch_acc_eval /= batch_size
            epoch_acc_eval += batch_acc_eval

        # showing last output
        target_sequence = " ".join(vocab_extended.ids_to_sequence(target.tolist()))
        predicted_tokens = predicted.argmax(dim=1).tolist()
        predicted_sequence = " ".join(vocab_extended.ids_to_sequence(predicted_tokens))
        print("Targeted sentence: {}".format(target_sequence))
        print("Predicted sentence: {}".format(predicted_sequence))

    return epoch_loss_train / len(train_iter), epoch_acc_train / len(train_iter), epoch_loss_eval / len(
        test_iter), epoch_acc_eval / len(test_iter)


def train(train_iter, test_iter, criterion, model, optimizer, epoch_start=0, num_epochs=NUM_EPOCHS):
    train_loss, eval_loss, best_loss, epochs = [], [], 0, []
    vocab = Vocab('data/vocab', voc_size)

    # training loop
    for epoch in range(epoch_start, num_epochs):
        epochs.append(epoch)
        start = time.time()
        epoch_loss_train, epoch_acc_train, epoch_loss_eval, epoch_acc_eval = train_epoch(train_iter, test_iter,
                                                                                         criterion, model, optimizer,
                                                                                         vocab)
        elapsed_time = time.time() - start

        print("-------------------------------- Epoch nb {} --------------------------------".format(epoch))
        print("\tloss train: {} | loss eval: {} | acc train: {} | acc eval: {} | time: {}".format(epoch_loss_train,
                                                                                                  epoch_loss_eval,
                                                                                                  epoch_acc_train,
                                                                                                  epoch_acc_eval,
                                                                                                  elapsed_time))

        # saving model
        is_best = epoch_loss_eval > best_loss
        best_loss = epoch_loss_eval if is_best else best_loss
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best)

        with open('losses.txt', 'a+') as f:
            f.write("{}\t{}\t{}\n".format(epoch_loss_train, epoch_loss_eval, epoch))

    return train_loss
