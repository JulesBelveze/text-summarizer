import torch
import os
from pickle import load


class DataLoader(torch.utils.data.Dataset):
    def __init__(self, set: str, data_dir="data"):
        '''Initialization'''

        switcher = {
            "train": os.path.join(data_dir, "train.pkl"),
            "test": os.path.join(data_dir, "test.pkl"),
            "validation": os.path.join(data_dir, "validation.pkl")
        }
        self.data_path = switcher.get(set, "Wrong set name.")

        with open(self.data_path, 'rb') as f:
            self.data = load(f)

    def __len__(self):
        '''return the number of articles'''
        return len(self.data)

    def __getitem__(self, idx):
        '''generates one sample of data'''
        X, y = self.data[idx]['story'], self.data[idx]['highlights']
        return X, y
