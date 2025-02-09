import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_process import getProcessedTrainingData, getProcessedTestingData
from sklearn.preprocessing import MinMaxScaler
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Dropout

# Load Training Data
dataset_train = getProcessedTrainingData()
training_set = dataset_train.iloc[:, 1:2].values  # Use only the second column (e.g., 'Open' price)

# Normalize Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_training_set = scaler.fit_transform(training_set)

# Create Training Sequences
X_train, y_train = [], []
for i in range(10, len(scaled_training_set)):
    X_train.append(scaled_training_set[i-10:i, 0])
    y_train.append(scaled_training_set[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # Reshape for LSTM

# Build LSTM Model
regressor = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])

# Compile & Train the Model
regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(X_train, y_train, epochs=50, batch_size=32)

# Load Testing Data
dataset_test = getProcessedTestingData()
actual_stock_price = dataset_test.iloc[:, 1:2].values  # Extract actual values

# Prepare Input Data for Testing
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 10:].values.reshape(-1, 1)
inputs = scaler.transform(inputs)

# Create Testing Sequences
X_test = np.array([inputs[i-10:i, 0] for i in range(10, len(inputs))])
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make Predictions
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Forecast Future Prices
future_predictions = []
last_known_values = X_test[-1].copy()  # Start from the last known values

for _ in range(12):  # Predict next 12 months
    next_prediction = regressor.predict(last_known_values.reshape(1, 10, 1))
    next_prediction_value = next_prediction.flatten()[0]  # Convert 2D array to scalar

    future_predictions.append(next_prediction_value)  # Append scalar value

    # Update `last_known_values` with the new prediction
    last_known_values = np.append(last_known_values[1:], [[next_prediction_value]], axis=0)

# Convert predictions back to original scale
future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions)

# Plot Results
plt.figure(figsize=(10, 5))
plt.plot(actual_stock_price.flatten(), label="Actual Stock Price", color='blue')
plt.plot(predicted_stock_price.flatten(), label="Predicted Stock Price", color='red', linestyle='dashed')
plt.plot(range(len(actual_stock_price), len(actual_stock_price) + 12), future_predictions.flatten(), label="Future Predictions", color='green', linestyle='dotted')

plt.title("Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
