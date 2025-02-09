import os
import numpy as np
import pandas as pd
import requests
import io
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from data_process import getProcessedTrainingData, getProcessedTestingData
from sklearn.preprocessing import MinMaxScaler
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Dropout

app = FastAPI()

load_dotenv()
api_key = os.getenv("API_KEY_ALPHAVANTAGE")
user_symbol = "SPY"
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={user_symbol}&datatype=csv&apikey={api_key}"

# Initialize Model and Scaler
scaler = MinMaxScaler(feature_range=(0, 1))
regressor = None

# Train the Model Once
def train_model():
    global regressor, scaler
    
    dataset_train = getProcessedTrainingData()
    training_set = dataset_train.iloc[:, 1:2].values

    scaled_training_set = scaler.fit_transform(training_set)

    X_train, y_train = [], []
    for i in range(10, len(scaled_training_set)):
        X_train.append(scaled_training_set[i - 10 : i, 0])
        y_train.append(scaled_training_set[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

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

    regressor.compile(optimizer="adam", loss="mean_squared_error")
    regressor.fit(X_train, y_train, epochs=50, batch_size=32)

# Train model at startup
train_model()

# Function to fetch stock data
@app.get("/Fetch")
def FetchStockData():
    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Error fetching stock data")

        df = pd.read_csv(io.StringIO(response.text))
        return df.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stock data: {str(e)}")

# Function to predict stock prices
def getFuturePredictions():
    dataset_test = getProcessedTestingData()
    dataset_total = pd.concat((getProcessedTrainingData()["Open"], dataset_test["Open"]), axis=0)

    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 10:].values.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(10, len(dataset_test) + 10):
        X_test.append(inputs[i-10:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    future_predictions = []
    last_known_values = X_test[-1].copy()

    for _ in range(12):
        next_prediction = regressor.predict(last_known_values.reshape(1, 10, 1))  # Shape (1,1)
        next_prediction = next_prediction.flatten()[0]  # Convert to scalar

        future_predictions.append(next_prediction)

        last_known_values = np.append(last_known_values[1:], [[next_prediction]], axis=0)  # Ensure correct shape

    
    future_predictions = np.asarray(future_predictions, dtype="object")
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).tolist()


@app.get("/Prediction")
def FetchPrediction():
    try:
        prediction = getFuturePredictions()
        return prediction

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

print(getFuturePredictions())