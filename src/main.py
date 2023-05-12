import DataCore
import MathCore

import pandas as pd
import numpy as np
from tqdm import tqdm
from xgboost import plot_importance, plot_tree
from xgboost import XGBRegressor

# import customtkinter
import dearpygui.dearpygui as dpg

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

import matplotlib.pyplot as plt

df = DataCore.to_pands_df(DataCore.get_data_from_exchange())

PERCENTAGE = .995
WINDOW = 10
PREDICTION_SCOPE = 1

df = MathCore.feature_engineering(data=df)

train, test = MathCore.train_test_split(df, WINDOW)
train_set, validation_set = MathCore.train_validation_split(train, PERCENTAGE)

# print(f"train_set shape: {train_set.shape}")
# print(f"validation_set shape: {validation_set.shape}")
# print(f"test shape: {test.shape}")

X_train, y_train, X_val, y_val = MathCore.windowing(train_set, validation_set, WINDOW, PREDICTION_SCOPE)

#Convert the returned list into arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

# print(f"X_train shape: {X_train.shape}")
# print(f"y_train shape: {y_train.shape}")
# print(f"X_val shape: {X_val.shape}")
# print(f"y_val shape: {y_val.shape}")

#Reshaping the Data

X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)

# print(f"X_train shape: {X_train.shape}")
# print(f"X_val shape: {X_val.shape}")

mae, xgb_model = MathCore.xgb_model(X_train, y_train, X_val, y_val, plotting=True)

#================FEATURES==================#
# fig, ax = plt.subplots(1, 1, figsize=(15, 15))

# plot_importance(xgb_model,ax=ax,height=0.5, max_num_features=10)
# ax.set_title("Feature Importance", size=30)
# plt.xticks(size=30)
# plt.yticks(size=30)
# plt.ylabel("Feature", size=30)
# plt.xlabel("F-Score", size=30)
# plt.show()
#==========================================#

X_test = np.array(test.iloc[:, :-1])
y_test = np.array(test.iloc[:, -1])
X_test = X_test.reshape(1, -1)

# print(f"X_test shape: {X_test.shape}")

pred_test_xgb = xgb_model.predict(X_test)

MathCore.plotting(df,y_val, y_test, pred_test_xgb, mae, WINDOW, PREDICTION_SCOPE)

#================================================================================================#
# plots = {}

# for window in tqdm([1, 2, 3, 4, 5]):
    
#     for percentage in tqdm([.92, .95, .97, .98, .99, .995]):

#         WINDOW = window
#         pred_scope = 0
#         PREDICTION_SCOPE = pred_scope
#         PERCENTAGE = percentage

#         train = df.iloc[:int(len(df))-WINDOW]
#         test = df.iloc[-WINDOW:]
        
#         train_set, validation_set = MathCore.train_validation_split(train, PERCENTAGE)

#         X_train, y_train, X_val, y_val = MathCore.windowing(train_set, validation_set, WINDOW, PREDICTION_SCOPE)

#         X_train = np.array(X_train)
#         y_train = np.array(y_train)

#         X_val = np.array(X_val)
#         y_val = np.array(y_val)

#         X_test = np.array(test.iloc[:, :-1])
#         y_test = np.array(test.iloc[:, -1])

#         X_train = X_train.reshape(X_train.shape[0], -1)
#         try:
#             X_val = X_val.reshape(X_val.shape[0], -1)
#             X_test = X_test.reshape(1, -1)
#         except ValueError:
#             break

#         xgb_model = XGBRegressor(gamma=1)
#         xgb_model.fit(X_train, y_train)

#         pred_val = xgb_model.predict(X_val)

#         mae = mean_absolute_error(y_val, pred_val)

#         pred_test = xgb_model.predict(X_test)
#         plotii= [y_test[-1], pred_test]

#         plots[str(window)+str(pred_scope)] = [y_val, y_test, pred_test, mae, WINDOW, PREDICTION_SCOPE, PERCENTAGE]
  

# MathCore.window_optimization(plots)

# for key in list(plots.keys())[:9]:
#     MathCore.plotting(plots[key][0], plots[key][1], plots[key][2], plots[key][3], plots[key][4], plots[key][5])

