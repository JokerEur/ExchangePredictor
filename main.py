import ccxt
import pandas as pd
import xgboost as xgb
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1d'
limit = 1000
ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

df['target'] = df['close'].shift(-1)
df.dropna(inplace=True)

df['sma20'] = df['close'].rolling(window=20).mean()
df['sma50'] = df['close'].rolling(window=50).mean()
df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
df['macd'] = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9).macd()
df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
df.dropna(inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'], axis=1), df['target'], test_size=0.2, shuffle=False)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
param = {'max_depth': 5, 'eta': 0.1, 'objective': 'reg:squarederror'}
num_round = 100
bst = xgb.train(param, dtrain, num_round)

y_pred_xgb = bst.predict(dtest)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f"MSE (XGBoost): {mse_xgb:.4f}")

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation=None))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

y_pred_tf = model.predict(X_test).flatten()
mse_tf = mean_squared_error(y_test, y_pred_tf)
print(f"MSE (TensorFlow): {mse_tf:.4f}")
