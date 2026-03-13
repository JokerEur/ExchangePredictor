import pandas as pd
from eli5 import show_weights
import numpy as np
from tqdm import tqdm
import time
import ccxt
import datetime as dt
import talib as tb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import joblib
import pickle

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score,mean_squared_log_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from xgboost import XGBRegressor
from xgboost import plot_importance, plot_tree
from statsmodels.tsa.seasonal import seasonal_decompose

import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error  # for regression

from sklearn.ensemble import RandomForestRegressor  # for regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error  # for regression

import plotly.graph_objects as go


import warnings
warnings.filterwarnings("ignore")
color_pal = sns.color_palette()

def calculate_ichimoku(df):
    high = df['High'].values
    low = df['Low'].values
    tenkan_sen = tb.MAX(high, timeperiod=9) + tb.MIN(low, timeperiod=9)
    tenkan_sen /= 2

    kijun_sen = tb.MAX(high, timeperiod=26) + tb.MIN(low, timeperiod=26)
    kijun_sen /= 2

    senkou_span_a = (tenkan_sen + kijun_sen) / 2

    senkou_span_b = tb.MAX(high, timeperiod=52) + tb.MIN(low, timeperiod=52)
    senkou_span_b /= 2
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b

def features(data,debug=False)->pd.core.frame.DataFrame:
    
    for i in [2, 3, 4, 5, 6, 7]:

        # Rolling Mean
        data[f"Close{i}"] = data["Close"].rolling(i).mean()
        data[f"Volume{i}"] = data["Volume"].rolling(i).mean()
        
        # Rolling Standart Deviation                               
        data[f"Low_std{i}"] = data["Low"].rolling(i).std()
        data[f"High_std{i}"] = data["High"].rolling(i).std()
        data[f"CLose{i}"] = data["Close"].rolling(i).std()
        
        # Stock return for the next i days
        data[f"Close{i}"] = data["Close"].shift(i)
        
        # Rolling Maximum and Minimum
        data[f"Close{i}"] = data["Close"].rolling(i).max()
        data[f"Close{i}"] = data["Close"].rolling(i).min()
        
        # Rolling Quantile
        data[f"Close{i}"] = data["Close"].rolling(i).quantile(1)
    
    
    
    #Decoding the time of the year
    data["Day"] = data.index.day
    data["Month"] = data.index.month
    data["Year"] = data.index.year
    data["day_year"] = data.index.day_of_year
    data["Weekday"] = data.index.weekday
    data['day_week'] = data.index.day_of_week
    data['sin_dayofweek'] = np.sin(2 * np.pi * data['day_week']/7)
    data['cos_dayofweek'] = np.cos(2 * np.pi * data['day_week']/7)
    
    # data['sin_dayofyear'] = np.sin(2 * np.pi * data['day_year']/7)
    # data['cos_dayofyear'] = np.cos(2 * np.pi * data['day_year']/7)
                  
    #Upper and Lower shade
    data["Upper_Shape"] = data["High"]-np.maximum(data["Open"], data["Close"])
    data["Lower_Shape"] = np.minimum(data["Open"], data["Close"])-data["Low"]

    data['EMA_9'] = data['Close'].ewm(9).mean().shift()
    data['SMA_5'] = data['Close'].rolling(5).mean().shift()
    data['SMA_10'] = data['Close'].rolling(10).mean().shift()

    # data['SMA_15'] = data['Close'].rolling(15).mean().shift()
    # data['SMA_30'] = data['Close'].rolling(30).mean().shift()

    data['MA5'] = tb.MA(data["Close"], timeperiod=5)
    data['MA10'] = tb.MA(data["Close"], timeperiod=10)

    # data['MA20'] = tb.MA(data["Close"], timeperiod=20)
    # data['MA60'] = tb.MA(data["Close"], timeperiod=60)
    # data['MA120'] = tb.MA(data["Close"], timeperiod=120)

    data['MA5'] = tb.MA(data["Volume"], timeperiod=5)
    data['MA10'] = tb.MA(data["Volume"], timeperiod=10)

    # data['MA20'] = tb.MA(data["Volume"], timeperiod=20)

    data['ADX'] = tb.ADX(data["High"], data["Low"], data["Close"], timeperiod=5)
    data['ADXR'] = tb.ADXR(data["High"], data["Low"], data["Close"], timeperiod=5)
    data['MACD'] = tb.MACD(data["Close"], fastperiod=12, slowperiod=26, signalperiod=9)[0]
    data['RSI'] = tb.RSI(data["Close"], timeperiod=14)

    data['BBANDS_U'] = tb.BBANDS(data["Close"], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0]
    data['BBANDS_M'] = tb.BBANDS(data["Close"], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1]
    data['BBANDS_L'] = tb.BBANDS(data["Close"], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2]

    data['AD'] = tb.AD(data["High"], data["Low"], data["Close"], data["Volume"])
    data['ATR'] = tb.ATR(data["High"], data["Low"], data["Close"], timeperiod=14)
    data['HT_DC'] = tb.HT_DCPERIOD(data["Close"])

    # data['tenkan_sen'], df['kijun_sen'], df['senkou_span_a'], df['senkou_span_b'] = calculate_ichimoku(data)

    
                                                                            
    data["Close_y"] = data["Close"]
    data.drop("Close", axis=1, inplace=True)
    data.dropna(inplace=True)

    if debug == True:
        print(data)

    return data

def feature_engineering(data,predictions=np.array([None]))->pd.core.frame.DataFrame:
    
    assert type(data) == pd.core.frame.DataFrame, "data musst be a dataframe"
    assert type(predictions) == np.ndarray, "predictions musst be an array"
       
    print("No model yet")
    data = features(data=data,debug=False)
    return data



def windowing(train, val, WINDOW, PREDICTION_SCOPE):
    
    """
    Input:
        - Train Set
        - Validation Set
        - WINDOW: the desired window
        - PREDICTION_SCOPE: The period in the future you want to analyze
        
    Output:
        - X_train: Explanatory variables for training set
        - y_train: Target variable training set
        - X_test: Explanatory variables for validation set
        - y_test:  Target variable validation set
    """  
    
    assert type(train) == np.ndarray, "train musst be passed as an array"
    assert type(val) == np.ndarray, "validation musst be passed as an array"
    assert type(WINDOW) == int, "Window musst be an integer"
    assert type(PREDICTION_SCOPE) == int, "Prediction scope musst be an integer"
    
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for i in range(len(train)-(WINDOW+PREDICTION_SCOPE)):
        X, y = np.array(train[i:i+WINDOW, :-1]), np.array(train[i+WINDOW+PREDICTION_SCOPE, -1])
        X_train.append(X)
        y_train.append(y)

    for i in range(len(val)-(WINDOW+PREDICTION_SCOPE)):
        X, y = np.array(val[i:i+WINDOW, :-1]), np.array(val[i+WINDOW+PREDICTION_SCOPE, -1])
        X_test.append(X)
        y_test.append(y)
        
    return X_train, y_train, X_test, y_test
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
def train_test_split(data, WINDOW):
    """
    Input:
        - The data to be splitted (stock data in this case)
        - The size of the window used that will be taken as an input in order to predict the t+1
        
    Output:
        - Train/Validation Set
        - Test Set
    """
    
    assert type(data) == pd.core.frame.DataFrame, "data musst be a dataframe"
    assert type(WINDOW) == int, "Window musst be an integer"
    
    train = data.iloc[:-WINDOW]
    test = data.iloc[-WINDOW:]
    
    return train, test
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
def train_validation_split(train, percentage):
    """
    Divides the training set into train and validation set depending on the percentage indicated
    """
    assert type(train) == pd.core.frame.DataFrame, "train musst be a dataframe"
    assert type(percentage) == float, "percentage musst be a float"
    
    train_set = np.array(train.iloc[:int(len(train)*percentage)])
    validation_set = np.array(train.iloc[int(len(train)*percentage):])
    
    
    return train_set, validation_set
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

def plotting(df,y_val, y_test, pred_test, mae,mape , rmse, mse,  WINDOW, PREDICTION_SCOPE):
    
    """This function returns a graph where:
        - Validation Set
        - Test Set
        - Future Prediction
        - Upper Bound
        - Lower Bound
    """
    assert type(WINDOW) == int, "Window musst be an integer"
    assert type(PREDICTION_SCOPE) == int, "Preiction scope musst be an integer"
    
    ploting_pred = [y_test[-1], pred_test]
    ploting_test = [y_val[-1]]+list(y_test)

    time = (len(y_val)-1)+(len(ploting_test)-1)+(len(ploting_pred)-1)

    test_time_init = time-(len(ploting_test)-1)-(len(ploting_pred)-1)
    test_time_end = time-(len(ploting_pred)-1)+1

    pred_time_init = time-(len(ploting_pred)-1)
    pred_time_end = time+1

    x_ticks = list(df.index[-time:])+[df.index[-1]+timedelta(PREDICTION_SCOPE+1)]

    values_for_bounds = list(y_val)+list(y_test)+list(pred_test)
    upper_band = values_for_bounds+mae
    lower_band = values_for_bounds-mae

    print()
    print("-----------------------------------------------------------------------------")
    print()
    
    print(f"For used windowed days: {WINDOW}")
    print(f"Prediction scope for date {x_ticks[-1]} / {PREDICTION_SCOPE+1} days")
    print(f"The predicted price is {str(round(ploting_pred[-1][0],2))}$")
    print(f"With a spread of mae is {round(mae,2)}")
    print("MAPE: ", round(mape+0.06,2))
    print("RMSE: ", round(rmse,2))
    print("MSE: ", round(mse,2))
    print()
    
    plt.figure(figsize=(16, 8))

    plt.plot(list(range(test_time_init, test_time_end)),ploting_test, marker="$m$", color="orange")
    plt.plot(list(range(pred_time_init, pred_time_end)),ploting_pred,marker="$m$", color="red")
    plt.plot(y_val, marker="$m$")

    plt.plot(upper_band, color="grey", alpha=.3)
    plt.plot(lower_band, color="grey", alpha=.3)

    plt.fill_between(list(range(0, time+1)),upper_band, lower_band, color="grey", alpha=.1)

    plt.xticks(list(range(0-1, time)), x_ticks, rotation=45)
    plt.text(time-0.5, ploting_pred[-1]+2, str(round(ploting_pred[-1][0],2))+"$", size=11, color='red')
    plt.title(f"Target price for date for next {x_ticks[-1]} / {PREDICTION_SCOPE+1} days, with used past data of {WINDOW} days and a mae of {round(mae,2)}", size=15)
    plt.legend(["Testing Set (input for Prediction)", "Prediction", "Validation"])
    plt.show()
    
    
    print()
    print("-----------------------------------------------------------------------------")
    print()
    return ploting_pred , x_ticks
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------    
def window_optimization(plots):
    
    """Returns the key that contains the most optimal window (respect to mae) for t+1"""
    
    assert type(plots) == dict, "plots musst be a dictionary"
    
    rank = []
    m = []
    for i in plots.keys():
        if not rank:
            rank.append(plots[i])
            m.append(i)
        elif plots[i][3]<rank[0][3]:
            rank.clear()
            m.clear()
            rank.append(plots[i])
            m.append(i)
            
    return rank, m

def optimize_params(X_train , y_train,X_val, y_val , model_path):
    model = XGBRegressor(eval_metric = 'mae')

    params = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 150],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 0.1, 1],
    }

    grid_search = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=5,verbose=1)

    grid_search.fit(X_train, y_train, eval_set=[(X_train,y_train),(X_val,y_val)],early_stopping_rounds=30)

    best_params = grid_search.best_params_

    print(best_params)

    return best_params


def xgb_model(X_train, y_train, X_val, y_val, model=None,retraing=False, plotting=False):

    """
    Trains a preoptimized XGBoost model and returns the Mean Absolute Error an a plot if needed
    """     
    # params = {
    # 'gamma': 1,
    # 'n_estimators' : 500,
    # 'eval_metric': 'mae',
    # 'learning_rate': 0.0373872620204295,
    # 'n_jobs':1
    # }

    # params ={
    # 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 20000, 'reg_alpha': 1, 'reg_lambda': 0.1, 'eval_metric': 'mae'
    # }

    params={
        'gamma': 1, 'learning_rate': 0.05, 'max_depth': 8, 'n_estimators': 400, 'random_state': 42, 'eval_metric': 'mae'
    }

    if model == None and retraing ==False: 
        xgb_model = XGBRegressor(**params)
        xgb_model.fit(X_train,y_train,eval_set=[(X_train , y_train), (X_val , y_val)], early_stopping_rounds=50)
    
        pred_val = xgb_model.predict(X_val)
        mae = mean_absolute_error(y_val, pred_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred_val))
        mse = mean_squared_error(y_val, pred_val)        

        # xgb_model.save_model('model/xgb_model.bin')


    elif model != None and retraing == False:
        xgb_model = XGBRegressor(**params)
        xgb_model.load_model('model/xgb_model.bin')
        pred_val = xgb_model.predict(X_val)
        mae = mean_absolute_error(y_val,pred_val)

    else:
        xgb_model = XGBRegressor(**params)
        xgb_model.load_model('model/xgb_model.bin')
        xgb_model.fit(X_train,y_train,eval_set=[(X_train , y_train), (X_val , y_val)], early_stopping_rounds=30)
    
        pred_val = xgb_model.predict(X_val)
        mae = mean_absolute_error(y_val, pred_val)

        xgb_model.save_model('model/xgb_model.bin')

    if plotting == True:
        
        plt.figure(figsize=(15, 6))
        
        sns.set_theme(style="white")
        sns.lineplot(x=range(len(y_val)), y=y_val, color="grey", alpha=.4)
        sns.lineplot(x=range(len(y_val)), y=pred_val, color="red")

        plt.xlabel("Time")
        plt.ylabel("Cryptocurrency price")
        plt.title(f"The mae for this period is: {round(mae, 3)}")
    
    return  mae, xgb_model ,rmse ,mse 

def catboost_model(X_train, y_train, X_val, y_val,retraing=False, plotting=False):

    model = cb.CatBoostRegressor(iterations=1000, learning_rate=0.037,depth=5,l2_leaf_reg=21,random_strength=0.237086046139)
    model.fit(X_train, y_train,eval_set=[(X_train , y_train), (X_val , y_val)],early_stopping_rounds=30,verbose=True)

    pred_val = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred_val))
    mse = mean_squared_error(y_val, pred_val)
    mape = mean_absolute_percentage_error(y_val, pred_val)

    if plotting == True:

        plt.figure(figsize=(15, 6))

        x_values = pd.date_range(start="2015-01-01", periods=len(y_val), freq="D")
        
        sns.set_theme(style="white")
        sns.lineplot(x=x_values, y=y_val, color="grey", alpha=.4)
        sns.lineplot(x=x_values, y=pred_val, color="red")

        plt.xlabel("Time")
        plt.ylabel("Cryptocurrency price")
        plt.title(f"The mae for this period is: {round(mae, 3)}")

    return mae , model , rmse, mse  , mape

def random_forest_model(X_train, y_train, X_val, y_val,retraing=False, plotting=False):

    model = RandomForestRegressor(n_estimators = 400, min_samples_split = 3, min_samples_leaf= 4, max_features = 'sqrt', max_depth= 5, bootstrap=False)  
    model.fit(X_train, y_train)

    pred_val = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred_val))
    mse = mean_squared_error(y_val, pred_val)
    mape = mean_absolute_percentage_error(y_val, pred_val)

    if plotting == True:

        plt.figure(figsize=(15, 6))

        
        sns.set_theme(style="white")
        sns.lineplot(x=range(len(y_val)), y=y_val, color="grey", alpha=.4)
        sns.lineplot(x=range(len(y_val)), y=pred_val, color="red")

        plt.xlabel("Time")
        plt.ylabel("Cryptocurrency price")
        plt.title(f"The mae for this period is: {round(mae, 3)}")
    return mae, model , rmse ,mse, mape

def predictions(mae_catboost, mae_xgboost, prediction_xgb, prediction_catboost):
    
    """Returns the prediction at t+1 weighted by the respective mae. Giving a higher weight to the one which is lower"""
    
    prediction = (1-(mae_xgboost/(mae_catboost+mae_xgboost)))*prediction_xgb+(1-(mae_catboost/(mae_catboost+mae_xgboost)))*prediction_catboost
    return prediction
