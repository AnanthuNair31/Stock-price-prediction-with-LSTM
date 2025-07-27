def add_moving_average(df,window=14):
    df['MA']=df['Close'].rolling(window=window).mean()
    return df

def add_rsi(df,window=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0,0)
    loss = -delta.where(delta < 0,0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    df['RSI']= 100-(100/(1+rs))
    return df

def add_macd(df, fast=12, slow=26, signal=9):
    df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['Signal_Line'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    return df
