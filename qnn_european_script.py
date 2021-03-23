import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit.visualization import *

from QuantumNet import QuantumNet

from sklearn.model_selection import train_test_split


# ----------------------------------
# Prepare data from European dataset
# ----------------------------------
dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = str(dir_path) + '/data/european/creditcard.csv'

df = pd.read_csv(file_path)
y = df['Class']
X = df.drop('Class', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

X_train = torch.from_numpy(X_train.to_numpy()).float()
X_test =  torch.from_numpy(X_test.to_numpy()).float()
y_train = torch.from_numpy(y_train.to_numpy())
y_test =  torch.from_numpy(y_test.to_numpy())

train_ds = torch.utils.data.TensorDataset(X_train, y_train)
test_ds = dataset = torch.utils.data.TensorDataset(X_test, y_test)


train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=True)
print(train_loader)

# -------------------------------------------
# Here is where we actually train the network
# -------------------------------------------
model = QuantumNet(2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.BCEWithLogitsLoss()

epochs = 1
loss_list = []

model.train()
for epoch in range(epochs):
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # Forward pass
        output = model(data).squeeze(0)
        # Calculating loss
        loss = loss_func(output, target.float())
        # Backward pass
        loss.backward()
        # Optimize the weights
        optimizer.step()
        
        total_loss.append(loss.item())
    loss_list.append(sum(total_loss)/len(total_loss))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
        100. * (epoch + 1) / epochs, loss_list[-1]))


# Concentrating on the first 100 samples
# n_samples = 100

# X_train = datasets.MNIST(root='./data', train=True, download=True,
#                          transform=transforms.Compose([transforms.ToTensor()]))