import pickle
from torch.utils.data import Dataset
import torch
import random
import numpy as np

def load_weibo_pkl():
    train_file = open('data/weibo/train.pkl', 'rb')
    train = pickle.load(train_file)
    val_file = open('data/weibo/valid.pkl', 'rb')
    val = pickle.load(val_file)
    test_file = open('data/weibo/test.pkl', 'rb')
    test = pickle.load(test_file)
    return train, val, test

train, valid, test = load_weibo_pkl()
dataset = []

dataset.extend(train)
dataset.extend(valid)
dataset.extend(test)


class WEIBO_Dataset(Dataset):
    def __init__(self):
        self.indices = dataset
        self.num_features = 768
        self.num_classes = 2

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx]