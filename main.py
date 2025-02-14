from fastapi import FastAPI
import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import io
import base64
import matplotlib.pyplot as plt

app = FastAPI()

model = load_model('stock_model.h5')
scaler = joblib.load('scaler.pkl')

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def home():
    return {'message':'Stock Prediction API'}

@app.get('/predict')
def predict_stock(stock_symbol:str,days:int=30):
    df = yf.download(stock_symbol,period='60d')
    data = df[['Close']].values
    data_scaled = scaler.transform(data)
    x_input = np.array(data_scaled[-60:])
    x_input = np.reshape(x_input,(1,60,1))
    predictions=[]
    for _ in range(days):
        pred = model.predict(x_input)
        predictions.append(pred[0][0])
        x_input = np.roll(x_input,-1,axis=1)
        x_input[0,-1,0]=pred[0][0]
    predicted_price = scaler.inverse_transform(np.array(predictions).reshape(-1,1)).flatten()
    return{"prediction":predicted_price.tolist()}

@app.get('/plot')
def plot_stock(stock_symbol:str,days:int=30):
    df = yf.download(stock_symbol,period='60d')
    data = df[['Close']].values
    data_scaled = scaler.transform(data)
    x_input = np.array(data_scaled[-60:])
    x_input = np.reshape(x_input,(1,60,1))
    predictions=[]
    for _ in range(days):
        pred = model.predict(x_input)
        predictions.append(pred[0][0])
        x_input = np.roll(x_input,-1,axis=1)
        x_input[0,-1,0]=pred[0][0]
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1,1)).flatten()
    fig = plt.figure(figsize=(12,6))
    plt.plot(predicted_prices,label="Predicted Price", color = 'red')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.legend()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer,format='png')
    plt.close(fig)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return{'image_base64':img_base64}
