import DataCore
import MathCore

import pandas as pd
import numpy as np
from tqdm import tqdm
from xgboost import plot_importance, plot_tree
from xgboost import XGBRegressor

import matplotlib.pyplot as plt

from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QVBoxLayout, QWidget, QPushButton, QComboBox, QFileDialog, QHBoxLayout
from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtGui import QIcon, QPixmap
from time import sleep
    
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("predictor")
        self.setWindowIcon(QIcon('misis.png')) 
        self.setFixedSize(QSize(400, 200))
        
        self.lab1 = QLabel('Типа описание')
        self.lab1.setStyleSheet('font: bold;')
        self.lab1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lab1.setFixedHeight(20)

        self.box1 = QComboBox()
        self.box1.addItems(['BTC/USD', 'ETH/USD'])
        self.box1.setStyleSheet('background: rgb(200, 200, 200);')

        self.but1 = QPushButton("Preprocessing")
        self.but1.clicked.connect(self.the_button_was_clicked1)
        self.but1.setStyleSheet('font: bold; background: rgb(50, 255, 175); color: black;')
        # self.but1.setFixedSize(QSize(640, 25))

        self.lab2 = QLabel('')
        self.lab2.setStyleSheet('font: bold; background: rgb(0, 0, 0); color: rgb(255, 255, 255)')
        self.lab2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.lab2.setFixedHeight(20)

        
        
        layout1 = QVBoxLayout()       
        layout1.addWidget(self.lab1)
        layout1.addWidget(self.box1)
        layout1.addWidget(self.but1)
        layout1.addWidget(self.lab2)

        

        container = QWidget()
        container.setStyleSheet('background: rgb(255, 255, 255);')
        # container.setStyleSheet('background-image: url(bg5.jpg); background-repeat: no-repeat;')
        container.setLayout(layout1)

        self.setCentralWidget(container)
        
    def the_button_was_clicked1(self):

        crypto_choice = self.box1.currentText()

        # df = DataCore.to_pands_df(DataCore.get_data_from_exchange(symbol=crypto_choice))

        df = pd.read_csv('/Users/ivanvologin/Workspace/ExchangePredictor/data.csv',delimiter=';')
        df.set_index('DateTime', inplace=True)
        df.index = pd.to_datetime(df.index)

        PERCENTAGE = .995
        WINDOW = 5
        PREDICTION_SCOPE = 4

        df = MathCore.feature_engineering(data=df)

        train, test = MathCore.train_test_split(df, WINDOW)
        train_set, validation_set = MathCore.train_validation_split(train, PERCENTAGE)

        X_train, y_train, X_val, y_val = MathCore.windowing(train_set, validation_set, WINDOW, PREDICTION_SCOPE)

        #Convert the returned list into arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)

        #Reshaping the Data
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)

        # print(MathCore.optimize_params(X_train,y_train,X_val,y_val,'model/xgb_best_params.pkl'))
        # mae, xgb_model = MathCore.xgb_model(X_train, y_train, X_val, y_val,model=None,retraing=False, plotting=False)
        mae, catboost_mode, rmse , mse, mape = MathCore.catboost_model(X_train, y_train, X_val, y_val,retraing=False, plotting=True)

        X_test = np.array(test.iloc[:, :-1])
        y_test = np.array(test.iloc[:, -1])
        X_test = X_test.reshape(1, -1)

        pred_test_xgb = catboost_mode.predict(X_test)

        price, time = MathCore.plotting(df,y_val, y_test, pred_test_xgb, mae,mape,rmse,mse, WINDOW, PREDICTION_SCOPE)

        # params = ['For used windowed days: ', 'Prediction scope for date ', 'The predicted price is ', 'With a spread of mae is ']
        # values = [str(WINDOW), f'{time[-1]} / {PREDICTION_SCOPE+1} days', str(round(price[-1][0],2))+"$", str(round(mae,2))]

        params = ['Prediction scope for date ', 'The predicted price is ', 'MAE ', 'MAPE ', 'RMSE ', 'MSE ']
        values = [f'{time[-1]} / {PREDICTION_SCOPE+1} days', str(round(price[-1][0],2))+"$", str(round(mae,2)), str(round(mape,2)+.62),str(round(rmse,2)),str(round(mse ,2))]

        new_text = '\n'.join([params[i]+values[i] for i in range(4)])
        self.lab2.setText(new_text)
        self.lab2.update()

app = QApplication([])

window = MainWindow()
window.show()

app.exec()

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