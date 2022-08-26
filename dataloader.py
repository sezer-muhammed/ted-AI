import pandas as pd
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import os
from torch.utils.data import Dataset, DataLoader

import warnings

warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, npy_file: str, device):

        self.device = device
        self.all_Data = np.load(npy_file, allow_pickle=True)
        self.all_Data = torch.from_numpy(self.all_Data).float().to(self.device)

        print(self.all_Data.shape)

    def __len__(self):
        return self.all_Data.shape[0] - 2000

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = idx + 2000
        # 0 is label, 1 and 2 are input
        past_1sec = self.all_Data[idx - 2000, :]
        past_750msec = self.all_Data[idx - 1500, :]
        past_500msec = self.all_Data[idx - 1000, :]
        past_250msec = self.all_Data[idx - 500, :]
        now = self.all_Data[idx, 1:]

        label = torch.unsqueeze(self.all_Data[idx, 0], dim = 0)

        inputs = torch.concat((past_1sec, past_750msec, past_500msec, past_250msec, now), dim = -1)


        return inputs, label
