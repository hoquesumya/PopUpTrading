import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from data_process import getProcessedData
from sklearn.preprocessing import MinMaxScaler
from keras.src.models import Sequential

data = getProcessedData()

training_set = data.iloc[:,1:2].values

scaler = MinMaxScaler(feature_range=(0,1))
scaled_training_set = scaler.fit_transform(training_set)
len_training_set = len(scaled_training_set)

X_train = []
y_train = []
for i in range(10, len_training_set):
    X_train.append(scaled_training_set[i-10:i, 0])
    y_train.append(scaled_training_set[i, 0])
X_train = np.array(X_train)
y_train = np.array(y_train)


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))







