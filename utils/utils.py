import torch
import shutil
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


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
    return stories, highlights
