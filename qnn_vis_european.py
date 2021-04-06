import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.model_selection import train_test_split
import sklearn
import csv

dir_path = os.path.dirname(os.path.realpath(__file__))
loss_path = str(dir_path) + '/res/european/qnn/loss.csv'
vis_path = str(dir_path) + '/res/european/qnn/vis/'

with open(loss_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        plt.figure()
        plt.plot([float(i) for i in row[1:]])
        plt.title('Hybrid NN Training Convergence')
        plt.xlabel('Epochs')
        plt.ylabel('BCE')
        plt.savefig(vis_path + 'loss_' + str(row[0]) + '.pdf')
        plt.close()


acc_path = str(dir_path) + '/res/european/qnn/accuracy.csv'
with open(acc_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        plt.figure()
        plt.plot([float(i) for i in row[1:]])
        plt.title('Hybrid NN Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Percentage of correct guesses')
        plt.savefig(vis_path + 'acc_' + str(row[0]) + '.pdf')
        plt.close()
