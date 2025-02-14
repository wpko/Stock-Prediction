import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib

stock_symbol = "AAPL"
start_date = "2015-01-01"
end_date = "2024-12-31"

df = yf.download(stock_symbol,start = start_date, end = end_date)
data = df[['Close']].values

scaler = MinMaxScaler(feature_range=(0,1))
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

model = Sequential([
  LSTM(50,return_sequences = True, input_shape = (sequence_length,1)),
  Dropout(0.2),
  LSTM(50, return_sequences=False),
  Dropout(0.2),
  Dense(25),
  Dense(1)
])

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train,y_train,epochs = 20, batch_size=32,validation_data = (x_test,y_test))
model.save("stock_model.h5")
joblib.dump(scaler,'scaler.pkl')
print("Model & Scaler Saved!")
