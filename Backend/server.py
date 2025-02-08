import requests
from dotenv import load_dotenv
import os
from fastapi import FastAPI

app = FastAPI()


load_dotenv()

api_key = os.getenv('API_KEY_ALPHAVANTAGE')
user_symbol = "SPY"
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={user_symbol}&datatype=csv&apikey={api_key}'


@app.get("/")
def Welcome():
    return "Hello"

@app.get("/Fetch")
def FetchStockData():
    stock_data = requests.get(url)
    return stock_data.json()

@app.get("/FetchTicker")
def FetchTicker():
    info = FetchStockData()
    ticker_name = info['Meta Data']

    return ticker_name["2. Symbol"]

@app.get("/FetchData")
def FetchData():
    info = FetchStockData()
    ticker_data = info["Monthly Time Series"]
    
    return ticker_data






