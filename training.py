from cProfile import label
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

dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)


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

criterion = nn.MSELoss()
optimizer = optim.Adam(forcer.parameters(), lr=0.005) 


forcer.load_state_dict(torch.load("forcer_net_v2_sigmoid.pth"))
forcer.train()
min_loss = 0.16 * 20

for epoch in range(100):  # loop over the dataset multiple times


    total_loss = 0
    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = forcer(inputs)
        loss = criterion(outputs, labels)
        print(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # print statistics

        if i % 20 == 19:
            print(f"{total_loss / 20}, epoch {epoch}, iter {i+1}")
            if total_loss < min_loss and epoch > 5:
                min_loss = total_loss
                torch.save(forcer.state_dict(), "forcer_net_v2_sigmoid.pth")
                print("saved")
            total_loss = 0