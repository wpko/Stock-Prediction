import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model
from keras.layers import LSTM,Dense,Dropout
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import pydantic
import uvicorn
import io
import base64
import time
from tensorflow.keras.callbacks import EarlyStopping

model = load_model('stock_model.h5')
scaler = joblib.load('scaler.pkl')

early_stopping = EarlyStopping(monitor = 'val_loss',patience=5,restore_best_weights=True)

stock_symbol = "AAPL"
start_date = "2015-01-01"
end_date = "2024-12-31"

time.sleep(2)
df = yf.download(stock_symbol,start=start_date,end=end_date)

data = df[['Close']].values

scaled_data = scaler.fit_transform(data)
train_size = int(len(scaled_data)*0.8)
train_data,test_data = scaled_data[:train_size],scaled_data[train_size:]

def create_sequences(dataset,seq_length):
    x,y = [],[]
    for i in range(len(dataset)-seq_length):
        x.append(dataset[i:i+seq_length])
        y.append(dataset[i+seq_length],)
    return np.array(x),np.array(y)
    
sequence_length = 60

x_train,y_train = create_sequences(train_data,sequence_length)
x_test,y_test = create_sequences(test_data,sequence_length)

x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

print(f"Training Data Shape: {x_train.shape},{y_train.shape}")
print(f"Testing Data Shape:{x_test.shape},{y_test.shape}")

prediction = model.predict(x_test)
predicted_prices = scaler.inverse_transform(prediction)
actual_prices = scaler.inverse_transform(y_test.reshape(-1,1))
rmse = np.sqrt(np.mean((predicted_prices-actual_prices)**2))
print(f"actual prices shape:{actual_prices.shape}")
print(f"predicted prices shape:{predicted_prices.shape}")
print(f"Root Mean Squared Error:{rmse}")
print("Actual Prices:", actual_prices[:10])
print("Predicted prices:", predicted_prices[:10])

app = FastAPI()
class StockRequest(BaseModel):
    stock_symbol: str

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['https://wpko.github.io'],
    allow_methods = ['POST','GET'],
    allow_headers = ['*'],
)

@app.get('/')
def home():
    return {'message':'Stock Prediction API'}

@app.post("/plot/")
async def plot_stock_base64(request:StockRequest,days:int=30):
    stock_data = request.dict()
    stock_symbol = stock_data["stock_symbol"]
    fig,ax = plt.subplots(figsize=(12,6))
    ax.plot(actual_prices,label = "Actual Price",color='blue')
    ax.plot(predicted_prices,label="Predicted Price",color = 'red')
    ax.set_title(f"{stock_symbol} Stock Price Prediction(LMST)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Stock Price")
    ax.legend()
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer,format='png')
    plt.close(fig)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    return {"image":img_base64}
