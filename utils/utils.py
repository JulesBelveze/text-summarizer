import os
import torch
import shutil
import numpy as np
from pyrouge import Rouge155
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from eval import get_batch_prediction
from vars import device


def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pt')


def load_ckp(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


def get_random_sentences(dataset, num):
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)

    idx = indices[:num]
    sampler = SubsetRandomSampler(idx)
    loader = DataLoader(dataset, sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    stories, highlights = dataiter.next()
    return stories.to(device), highlights.to(device)


def get_rouge_score(system_dir='predicted_summaries', model_dir='targeted_summaries'):
    r = Rouge155()
    r.system_dir = system_dir
    r.model_dir = model_dir
    r.system_filename_pattern = 'summary.(\d+).txt'
    r.model_filename_pattern = 'summary.[A-Z].#ID#.txt'

    output = r.convert_and_evaluate()
    print(output)
    output_dict = r.output_to_dict(output)
    print(output_dict)


def get_rouge_files(model, data_iter, system_dir='predicted_summaries', model_dir='targeted_summaries'):
    counter = 1
    for story, highlight in data_iter:
        prediction = model(story.to(device), highlight.to(device))
        clean_outputs, clean_targets = get_batch_prediction(prediction, highlight)

        for output, target in zip(clean_outputs, clean_targets):
            try:
                with open(os.path.join(system_dir,"summary.{}.txt".format(counter)), 'w+') as f:
                    f.write(output)
                with open(os.path.join(model_dir, "summary.A.{}.txt".format(counter)), 'w+') as f:
                    f.write(target)
                counter += 1
            except:
                pass
