from fastapi import FastAPI
from fastapi.responses import Response
import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import io
import base64
import matplotlib.pyplot as plt
import time


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
    try:
        time.sleep(2)
        df = yf.download(stock_symbol,period='60d')
        if df.empty:
            return{'error:':'Invalid stock symbol or no data available'}
        data = df[['Close']].values
        data_scaled = scaler.fit_transform(data)
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
    except Exception as e:
        return {"error:": str(e)}

@app.get('/plot')
def plot_stock(stock_symbol:str,days:int=10):
    try:
        time.sleep(2)
        df = yf.download(stock_symbol,period='60d')
        data = df[['Close']].values
        scaled_data = scaler.fit_transform(data)

        train_size = int(len(scaled_data)*0.8)
        train_data,test_data = scaled_data[:train_size], scaled_data[train_size:]
        def create_sequences(dataset,seq_length):
            x,y = [], []
            for i in range(len(dataset)-seq_length):
                x.append(dataset[i:i+seq_length])
                y.append(dataset[i+seq_length])
            return np.array(x),np.array(y)

        sequence_length = 60
        x_train,y_train = create_sequences(train_data,sequence_length)
        x_test,y_test = create_sequences(test_data,sequence_length)

        x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
        x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
        if df.empty:
            return{'error':"Invalid stock symbol or no data available"}
            
        prediction = model.predict(x_test)
        predicted_prices = scaler.inverse_transform(prediction)
        actual_prices = scaler.inverse_transform(y_test.reshape(-1,1))
        
        fig = plt.figure(figsize=(12,6))
        plt.figure(figsize=(12,6))
        plt.plot(actual_prices,label = "Actual Price",color='blue')
        plt.plot(predicted_prices,label="Predicted Price",color = 'red')
        plt.title(f"{stock_symbol} Stock Price Prediction(LMST)")
        plt.xlabel("Days")
        plt.ylabel("Stock Price")
        plt.lengend()
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer,format='png')
        plt.close(fig)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        return {"image": f"data:image/png;base64,{img_base64}"}
    except Exception as e:
        return{"error":str(e)}

