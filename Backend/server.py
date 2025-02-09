import requests
from dotenv import load_dotenv
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from predict_stock import getFuturePredictions

app = FastAPI()

load_dotenv()

api_key = os.getenv("API_KEY_ALPHAVANTAGE")
user_symbol = "SPY"
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={user_symbol}&datatype=json&apikey={api_key}"


@app.get("/")
def Welcome():
    return {"message": "Hello, welcome to the stock API!"}


@app.get("/Fetch")
def FetchStockData():
    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Error fetching stock data")

        
        # Convert DataFrame to JSON format
        return response.json()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stock data: {str(e)}")


@app.get("/FetchTicker")
def FetchTicker():
    try:
        info = FetchStockData()
        if not info:
            raise HTTPException(status_code=400, detail="No data available")

        ticker_name = user_symbol  # Since AlphaVantage CSV does not return metadata
        return {"symbol": ticker_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch ticker: {str(e)}")


@app.get("/FetchData")
def FetchData():
    try:
        info = FetchStockData()
        if not info:
            raise HTTPException(status_code=400, detail="No data available")

        return info  

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stock data: {str(e)}")


@app.get("/Prediction")
async def FetchPrediction():
    try:
        prediction = await getFuturePredictions()
        return {"prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
