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

# Focusing on the first 100 training samples
X_train = torch.from_numpy(X_train.to_numpy()[:100]).float()
y_train = torch.from_numpy(y_train.to_numpy()[:100])
# Focusing on the first 50 test samples
X_test =  torch.from_numpy(X_test.to_numpy()[:50]).float()
y_test =  torch.from_numpy(y_test.to_numpy()[:50])

train_ds = torch.utils.data.TensorDataset(X_train, y_train)
test_ds = dataset = torch.utils.data.TensorDataset(X_test, y_test)
print("X_train shape: " + str(X_train.shape))
print("X_test shape: " + str(X_test.shape))
# exit()


train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=True)


# -------------------------------------------
# Here is where we actually train the network
# -------------------------------------------
model = QuantumNet(2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.BCEWithLogitsLoss()

epochs = 20
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


