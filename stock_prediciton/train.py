from model import build_model
from data_loader import fetch_data, normalize_data, create_sequences
from indicators import add_moving_average, add_rsi, add_macd
import numpy as np
from datetime import datetime
from keras.models import load_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

def train_model(save_path='model_trained.keras', save_date_path='last_trained.txt'):

    stocks = ['AAPL', 'GOOG', 'IBM', 'RELIANCE.NS', 'INFY.NS', 'HDFCBANK.NS']
    all_X, all_y = [], []

    for ticker in stocks:
        df = fetch_data(ticker)

        df = add_moving_average(df)
        df = add_rsi(df)
        df = add_macd(df)
        df.dropna(inplace=True)

        features = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA', 'RSI', 'MACD', 'Signal_Line']]
        data, _ = normalize_data(features)
        X, y = create_sequences(data)

        all_X.append(X)
        all_y.append(y)

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    split = int(0.8 * len(X_all))
    X_train, X_val = X_all[:split], X_all[split:]
    y_train, y_val = y_all[:split], y_all[split:]

    model = build_model((X_train.shape[1], X_train.shape[2]))
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, batch_size=32)


    model.save(save_path)


    with open(save_date_path, 'w') as f:
        f.write(datetime.today().strftime('%Y-%m-%d'))


    preds = model.predict(X_val)
    plt.figure(figsize=(12, 6))
    plt.plot(y_val, label='Actual')
    plt.plot(preds, label='Predicted')
    plt.legend()
    plt.title('Validation Predictions vs Actual')
    plt.show()


if __name__ == "__main__":
    train_model()
