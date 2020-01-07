import torch
from torch.autograd import Variable
from vars import *
from utils.vocab import Vocab
from utils.data import Batcher

voc = Vocab("data/vocab", voc_size)
stop_id, unk_id, pad_id = voc.word_2_id(STOP_TOKEN), voc.word_2_id(UNKNOWN_TOKEN), voc.word_2_id((PAD_TOKEN))

def eval(test_iter, model, criterion):
    epoch_loss_eval, epoch_acc_eval = 0, 0
    with torch.no_grad():
        for i, batch in enumerate(test_iter):
            batch_loss_eval, batch_acc_eval = 0, 0
            story, highlight = batch
            batcher = Batcher(story, highlight, voc)
            story, highlight, extra_zeros, story_extended, _, vocab_extended = batcher.get_batch(
                get_vocab_extended=True)

            story = story.to(device)
            highlight = highlight.to(device)
            story_extended = story_extended.to(device)
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

        print("Loss: {}".format(epoch_loss_eval))
        print("Acc: {}".format(epoch_acc_eval))


def get_batch_prediction(stories, output, target, vocab=voc):
    clean_output, clean_target, clean_stories = [],[], []
    for pred, target, story in zip(output, target, stories):
        target = vocab.ids_to_sequence(target.tolist())
        story = vocab.ids_to_sequence(story.tolist())
        try:
            target = target[:target.index(STOP_TOKEN)]
            clean_output.append(get_sentence_prediction(pred))
            clean_target.append(" ".join(target))
            clean_stories.append(" ".join(story))
        except ValueError:
            target = target[:min(len(target), MAX_LEN_HIGHLIGHT)]
            clean_output.append(get_sentence_prediction(pred))
            clean_target.append(" ".join(target))
            clean_stories.append(" ".join(story))

    # print the first input and its corresponding output
    print("STORIES: {} \n".format(clean_stories[0]))
    print("TARGET SEQUENCE: {} \n".format(clean_target[0]))
    print("PREDICTED SEQUENCE: {} \n".format(clean_output[0]))
    return clean_output, clean_target

def get_sentence_prediction(sentence):
    predicted_sentence = []
    pred = torch.topk(sentence, k=3, dim=1)[1]
    for token in pred:
        token = get_right_token(token)
        predicted_sentence.append(voc.id_2_word(token))

    try:
        return " ".join(predicted_sentence[:predicted_sentence.index(STOP_TOKEN)])
    except ValueError:
        return " ".join(predicted_sentence)


def get_right_token(token):
    for t in token.tolist():
        if t not in [unk_id, pad_id]:
            return t
