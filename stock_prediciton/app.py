import streamlit as st
from keras.models import load_model
from data_loader import fetch_data, normalize_data, create_sequences
from indicators import add_moving_average, add_rsi, add_macd
import matplotlib.pyplot as plt
import numpy as np
import os


def denormalize_prediction(pred, scaler, target_index=3, n_features=9):
    padded = np.zeros((1, n_features))
    padded[0][target_index] = pred
    inv = scaler.inverse_transform(padded)
    return inv[0][target_index]

st.title("Stock Price Trend Prediction")
stock = st.text_input("Enter Stock Ticker (e.g., AAPL):", value="AAPL")
predict_btn = st.button("Predict")

if predict_btn:
    try:
        df = fetch_data(stock)

        if df.empty:
            st.error("No data found. Please check the stock ticker and try again.")
        else:

            df = add_moving_average(df)
            df = add_rsi(df)
            df = add_macd(df)
            df.dropna(inplace=True)


            st.subheader("Stock Chart with MA & RSI")
            fig, ax = plt.subplots()
            ax.plot(df['Close'], label='Close Price', linestyle='-', color='blue')
            ax.plot(df['MA'], label='Moving Average', linestyle='-', color='orange')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)


            st.subheader("RSI Indicator (Relative Strength Index)")
            fig_rsi, ax_rsi = plt.subplots(figsize=(10, 3))
            ax_rsi.plot(df['RSI'], label='RSI', color='purple')
            ax_rsi.axhline(70, color='red', linestyle='--', label='Overbought (70)')
            ax_rsi.axhline(30, color='green', linestyle='--', label='Oversold (30)')
            ax_rsi.set_ylabel('RSI')
            ax_rsi.set_title("RSI Indicator")
            ax_rsi.legend()
            st.pyplot(fig_rsi)


            features = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA', 'RSI', 'MACD', 'Signal_Line']]
            data_scaled, scaler = normalize_data(features)
            X, _ = create_sequences(data_scaled)


            model_path = os.path.join(os.path.dirname(__file__), 'model_trained.keras')
            model = load_model(model_path)


            prediction = model.predict(np.expand_dims(X[-1], axis=0))
            future_price = denormalize_prediction(prediction[0][0], scaler, target_index=3, n_features=9)
            currency = "₹" if stock.endswith((".NS",)) else "$"

            st.write(f"### Predicted Next Day Close: **{currency}{future_price:,.2f}**")



    except Exception as e:
        st.error("Error: Unable to fetch or process the stock data.")
        st.exception(e)
