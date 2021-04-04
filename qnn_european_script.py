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
import sklearn


# ----------------------------------
# Prepare data from European dataset
# ----------------------------------
dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = str(dir_path) + '/data/european/creditcard.csv'

df = pd.read_csv(file_path)
df = sklearn.utils.shuffle(df)
y = df['Class']
X = df.drop('Class', axis=1)

X_0 = X[y == 0][:1000]
X_1 = X[y == 1]
y_0 = y[y == 0][:1000]
y_1 = y[y == 1]


X = pd.concat([X_0, X_1])
# print("X_0 shape: {}".format(X_0.shape))
# print("X_1 shape: {}".format(X_1.shape))
# print("X shape: {}".format(X.shape))
y = pd.concat([y_0, y_1])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Turn everything to tensors so that Pythorch knows what to do
X_train = torch.from_numpy(X_train.to_numpy()).float()
y_train = torch.from_numpy(y_train.to_numpy())
X_test =  torch.from_numpy(X_test.to_numpy()).float()
y_test =  torch.from_numpy(y_test.to_numpy())

train_ds = torch.utils.data.TensorDataset(X_train, y_train)
test_ds = dataset = torch.utils.data.TensorDataset(X_test, y_test)
print("X_train shape: " + str(X_train.shape))
print("X_test shape: " + str(X_test.shape))
# exit()

batch_size   = 5
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_ds , batch_size=batch_size, shuffle=True)


# -------------------------------------------
# Here is where we actually train the network
# -------------------------------------------
model = QuantumNet(10)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1000/492.0]))

epochs = 30
loss_list = []

model.train()
for epoch in range(epochs):
    total_loss = []
    correct = 0
    count_0 = 0
    total_size = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # if batch_idx > 20:
        #     exit()
        optimizer.zero_grad()
        # Forward pass
        output = model(data).squeeze(0).squeeze(1)
        pred = (output >= 0).float()
        for x in pred.tolist():
            if (x != 0.):
                count_0 += 1
        # print(target.view_as(pred).float())
        correct += pred.eq(target.view_as(pred).float()).sum().item()
        loss = loss_func(output, target.float())
        total_size += list(output.size())[0]
        # Backward pass
        loss.backward()
        # Optimize the weights
        optimizer.step()
        
        total_loss.append(loss.item())
    loss_list.append(sum(total_loss)/(batch_size * len(total_loss)))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
        100. * (epoch + 1) / epochs, loss_list[-1]))
    print('Accuracy: {:.1f}%'.format(
        correct / total_size* 100))
    print('Proportion of 0s was: {}%'.format(100* count_0 / total_size))

# ---------------------
# Save training results
# ---------------------
plt.plot(loss_list)
plt.title('Hybrid NN Training Convergence')
plt.xlabel('Epochs')
plt.ylabel('BCE')
plt.savefig('temp.pdf')

# ---------------------
# Evaluate the training
# ---------------------
model.eval()
with torch.no_grad():
    total_loss = []
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data).squeeze(0).squeeze(1)
        pred = (output >= 0).float()
        correct += pred.eq(target.view_as(pred).float()).sum().item()
        
        loss = loss_func(output, target.float())
        total_loss.append(loss.item())
        
    print('Performance on training data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
        sum(total_loss) / (batch_size * len(total_loss)),
        correct / (batch_size * len(train_loader)) * 100)
        )

# ------------------
# Evaluate the model
# ------------------
model.eval()
with torch.no_grad():
    total_loss = []
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data).squeeze(0).squeeze(1)
        
        pred = (output >= 0).float()
        correct += pred.eq(target.view_as(pred).float()).sum().item()
        
        loss = loss_func(output, target.float())
        total_loss.append(loss.item())
        
    print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
        sum(total_loss) / (batch_size * len(total_loss)),
        correct / (batch_size * len(test_loader)) * 100)
        )


exit()
# --------------------
# Get some predictions
# --------------------
n_samples_show = 100
count = 0
# fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

model.eval()
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        if count == n_samples_show:
            break
        output = model(data)
         
        pred = (output >= 0).int()
        act = target[0].numpy().squeeze()
        # print("Predicted {}, actually was {}".format(pred.item(), act))

        # axes[count].imshow(data[0].numpy().squeeze(), cmap='gray')

        # axes[count].set_xticks([])
        # axes[count].set_yticks([])
        # axes[count].set_title('Predicted {}'.format(pred.item()))

        
        count += 1