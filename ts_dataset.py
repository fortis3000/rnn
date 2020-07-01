"""Custom timeseries datasets using PyTorch"""

import torch
import torch.nn as nn
from torch.autograd import Variable


class TimeSeriesDataSet(torch.utils.data.Dataset):
    """Torch Dataset for timeseries data packed into pandas.DataFrame
    
    Shape of hidden_size is 1x1
    
    """

    def __init__(
        self,
        x,
        y,
        transform=None,
        target_transform=None,
        normalize=False,
        device="cpu",
    ):
        self.x = torch.FloatTensor(x).unsqueeze(2).to(device)
        self.y = torch.FloatTensor(y).unsqueeze(2).to(device)
        self.transform = transform
        self.target_transform = target_transform
        self.X_shape = self.x.shape
        self.Y_shape = self.y.shape

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        item, target = self.x[idx], self.y[idx]

        return item, target


class ToDevice(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        return sample.to(self.device)

