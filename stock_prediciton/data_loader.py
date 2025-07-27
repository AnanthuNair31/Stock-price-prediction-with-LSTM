import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


def fetch_data(ticker, start='2015-01-01', end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df

def normalize_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

def create_sequences(data, seq_length=60):
    X,y = [],[]
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i,3])
    return np.array(X), np.array(y)
