import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import ccxt
import datetime as dt
import talib as tb
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
                  
    #Upper and Lower shade
    data["Upper_Shape"] = data["High"]-np.maximum(data["Open"], data["Close"])
    data["Lower_Shape"] = np.minimum(data["Open"], data["Close"])-data["Low"]

    data['EMA_9'] = data['Close'].ewm(9).mean().shift()
    data['SMA_5'] = data['Close'].rolling(5).mean().shift()
    data['SMA_10'] = data['Close'].rolling(10).mean().shift()
    data['SMA_15'] = data['Close'].rolling(15).mean().shift()
    data['SMA_30'] = data['Close'].rolling(30).mean().shift()
    data['MA5'] = tb.MA(data["Close"], timeperiod=5)
    data['MA10'] = tb.MA(data["Close"], timeperiod=10)
    data['MA20'] = tb.MA(data["Close"], timeperiod=20)
    data['MA60'] = tb.MA(data["Close"], timeperiod=60)
    data['MA120'] = tb.MA(data["Close"], timeperiod=120)
    data['MA5'] = tb.MA(data["Volume"], timeperiod=5)
    data['MA10'] = tb.MA(data["Volume"], timeperiod=10)
    data['MA20'] = tb.MA(data["Volume"], timeperiod=20)
    data['ADX'] = tb.ADX(data["High"], data["Low"], data["Close"], timeperiod=14)
    data['ADXR'] = tb.ADXR(data["High"], data["Low"], data["Close"], timeperiod=14)
    data['MACD'] = tb.MACD(data["Close"], fastperiod=12, slowperiod=26, signalperiod=9)[0]
    data['RSI'] = tb.RSI(data["Close"], timeperiod=14)
    data['BBANDS_U'] = tb.BBANDS(data["Close"], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0]
    data['BBANDS_M'] = tb.BBANDS(data["Close"], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1]
    data['BBANDS_L'] = tb.BBANDS(data["Close"], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2]
    data['AD'] = tb.AD(data["High"], data["Low"], data["Close"], data["Volume"])
    data['ATR'] = tb.ATR(data["High"], data["Low"], data["Close"], timeperiod=14)
    data['HT_DC'] = tb.HT_DCPERIOD(data["Close"])
    
                                                                            
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
    data = features(data=data)
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
def plotting(df,y_val, y_test, pred_test, mae, WINDOW, PREDICTION_SCOPE):
    
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
    
    print(f"For used windowed data: {WINDOW}")
    print(f"Prediction scope for date {x_ticks[-1]} / {PREDICTION_SCOPE+1} days")
    print(f"The predicted price is {str(round(ploting_pred[-1][0],2))}$")
    print(f"With a spread of mae is {round(mae,2)}")
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
    plt.title(f"Target price for date {x_ticks[-1]} / {PREDICTION_SCOPE+1} days, with used past data of {WINDOW} days and a mae of {round(mae,2)}", size=15)
    plt.legend(["Testing Set (input for Prediction)", "Prediction", "Validation"])
    plt.show()
    
    print()
    print("-----------------------------------------------------------------------------")
    print()
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

def xgb_model(X_train, y_train, X_val, y_val, plotting=False):

    """
    Trains a preoptimized XGBoost model and returns the Mean Absolute Error an a plot if needed
    """     
    params = {
    'gamma': 1,
    'n_estimators' : 200 
    }

    xgb_model = XGBRegressor(**params)
    xgb_model.fit(X_train,y_train)
    
    pred_val = xgb_model.predict(X_val)
    mae = mean_absolute_error(y_val, pred_val)

    if plotting == True:
        
        plt.figure(figsize=(15, 6))
        
        sns.set_theme(style="white")
        sns.lineplot(x=range(len(y_val)), y=y_val, color="grey", alpha=.4)
        sns.lineplot(x=range(len(y_val)), y=pred_val, color="red")

        plt.xlabel("Time")
        plt.ylabel("Cryptocurrency price")
        plt.title(f"The mae for this period is: {round(mae, 3)}")
    
    return  mae, xgb_model