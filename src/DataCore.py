import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import ccxt
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# Evaluate metrics
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

# XGBoost
from xgboost import XGBRegressor
from xgboost import plot_importance, plot_tree


# To avoid warning messages
import warnings
warnings.filterwarnings("ignore")
color_pal = sns.color_palette()

def get_data_from_exchange(symbol='BTC/USD',timeframe='1d'):
    exchange = ccxt.bitfinex()
    limit = 3500
    symbol = input("Specify cryptocurrency (e.g BTC/USD , ETH/USDT) or use default (BTC/USD): ")
    if not symbol:
        symbol = 'BTC/USD'

    data = []

    print(f"Obtaining data from exchange: ")
    for i in tqdm(range(30)):
        start_timestamp = exchange.parse8601('2014-01-01T00:00:00Z')
        if i > 0:
            start_timestamp = data[-1][0] + 1  
        page_data = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit, since=start_timestamp)
        data += page_data
    return data

def to_pands_df(data)->pd.core.frame.DataFrame:
    df = pd.DataFrame(data, columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['DateTime'] = pd.to_datetime(df['DateTime'], unit = 'ms')
    df.set_index('DateTime', inplace=True)
    return df

