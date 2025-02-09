import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from data_process import getProcessedTrainingData, getProcessedTestingData
from sklearn.preprocessing import MinMaxScaler
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

dataset_train = getProcessedTrainingData()
training_set = dataset_train.iloc[:, 1:2].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_training_set = scaler.fit_transform(training_set)
len_training_set = len(scaled_training_set)

X_train, y_train = [], []
for i in range(10, len_training_set):
    X_train.append(scaled_training_set[i-10:i, 0])
    y_train.append(scaled_training_set[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

dataset_test = getProcessedTestingData()
actual_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 10:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)
len_testing_set = len(dataset_test)

X_test = []
for i in range(10, len_testing_set + 10):
    X_test.append(inputs[i-10:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

plt.plot(predicted_stock_price, color='blue', label='Predicted')
plt.plot(actual_stock_price, color='red', label='Actual')
plt.legend()
plt.show()