import pandas as pd
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import os
from torch.utils.data import Dataset, DataLoader
from dataloader import FaceLandmarksDataset

import warnings

warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = FaceLandmarksDataset("data.npy", device = device)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

class FingerForcer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(14, 100)
        self.fc2_5 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2_5(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


forcer = FingerForcer()

print(forcer)

forcer.to(device)

forcer.load_state_dict(torch.load("forcer_net_v2_sigmoid.pth"))
forcer.eval()

total_data = []

import time


start = time.time()
for i, (inputs, labels) in enumerate(dataloader):
    outputs = forcer(inputs)

    saba = labels.cpu().detach().numpy()
    output = outputs.cpu().detach().numpy()

    total_data.append([saba[0][0], output[0][0]])
    print(saba, output)
    if i == 58_001:
        break

finish = time.time()
print(f"inference Hz: {6000100/(finish-start)}")
data_frame = pd.DataFrame(total_data, columns=["saba", "model output"])
data_frame.to_excel("saba_vs_model.xlsx")