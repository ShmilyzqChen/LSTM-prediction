# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 22:48:49 2024

@author: Shmily
"""

# coding: utf-8
from __future__ import print_function
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# from pandas import datetime
# import math
# import itertools
# from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
# import datetime
from sklearn import metrics
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score # 拟合优度
# from math import sqrt
# import math
#importing keras modules
from keras.models import Sequential
from keras.layers import Dense, Activation ,Dropout , Flatten , Conv1D , MaxPooling1D
from keras.layers import Reshape, Flatten, Bidirectional 
from keras.layers.recurrent import LSTM
from keras import losses
from keras import optimizers
from timeit import default_timer as timer
import os
# from keras.models import load_model
# model = load_model('lstm_model.h5')
# yhat = model.predict(X, verbose=0)

def data_standardization(data0):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data0)
    return data, scaler

def data_anti_standardization(data,scaler):
    data0 = scaler.inverse_transform(data.reshape(1,-1)) 
    return data0

def train_test_data_split(data,train_size,histroy_time_steps):
    # 只分训练集和测试集，验证集包含在训练集中，model.fit中有所体现
    result = []
    for i in range(len(data)-histroy_time_steps):
        result.append(data[i:i+histroy_time_steps])
    result = np.array(result)
    
    train   = result[:train_size,:]
    x_train = train[:, :-1]
    y_train = train[:, -1][:,-1]      
    x_test  = result[train_size:,:-1]
    y_test  = result[train_size:,-1][:,-1]

    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2])
    x_test  = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2])
    x_train = x_train.astype('float64')
    y_train = y_train.astype('float64')
    x_test  = x_test.astype('float64')
    y_test  = y_test.astype('float64')
    
    # print("X_train", x_train.shape)
    # print("y_train", y_train.shape)
    # print("X_test", x_test.shape)
    # print("y_test", y_test.shape)
    return x_train, y_train, x_test, y_test
    
def plot_loss(history_dict):
    # 如果需要画出训练集和验证集的损失曲线
    # plot_loss(history_dict)
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    loss_values50 = loss_values[0:150]
    val_loss_values50 = val_loss_values[0:150]
    epochs = range(1, len(loss_values50) + 1)
    plt.plot(epochs, loss_values50, color = 'blue', label='Training loss')
    plt.plot(epochs, val_loss_values50,color='red', label='Validation loss')
    plt.rc('font', size = 18)
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.xticks(epochs)
    fig = plt.gcf()
    fig.set_size_inches(15,7)
    #fig.savefig('img/tcstest&validationlosscnn.png', dpi=300)
    plt.show()
    
def plot_mae(history_dict): 
    # 如果需要画出训练集和验证集的mae
    # plot_mae(history_dict)
    mae = history_dict['mae']
    vmae = history_dict['val_mae']
    epochs = range(1, len(mae) + 1)
    plt.plot(epochs, mae,color = 'blue', label='Training error')
    plt.plot(epochs, vmae,color='red', label='Validation error')
    plt.title('Training and validation error')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend()
    plt.xticks(epochs)
    fig = plt.gcf()
    fig.set_size_inches(15,7)
    #fig.savefig('img/tcstest&validationerrorcnn.png', dpi=300)
    plt.show()
    
def plot_y_test(y_test,y_pred):
    plt.plot(y_pred,color='red', label='prediction')
    plt.plot(y_test,color='blue', label='y_test')
    plt.xlabel('No. of Trading Days')
    plt.ylabel('Close Value (scaled)')
    plt.legend(loc='upper left')
    fig = plt.gcf()
    fig.set_size_inches(15, 5)
    #fig.savefig('img/tcstestcnn.png', dpi=300)
    plt.show()   
    
def plot_y_train(y_train,y_pred):
    plt.plot(y_pred[:848],color='red', label='prediction on training samples')
    x = np.array(range(848))
    plt.plot(x,y_pred[:848],color = 'magenta',label ='prediction on validating samples')
    plt.plot(x,y_train[:848],color='blue', label='y_train')
    plt.xlabel('No. of Trading Days')
    plt.ylabel('Close Value (scaled)')
    plt.legend(loc='upper left')
    fig = plt.gcf()
    fig.set_size_inches(20,10)
    #fig.savefig('img/tcstraincnn.png', dpi=300)
    plt.show()   

def build_model(input):
    model = Sequential()
    ## CNN-BiLSTM-tanh&sigmoid                                    # test model-4
    model.add(Dense(32,input_shape=(input[0],input[1])))
    model.add(Bidirectional(LSTM(32, activation='tanh',recurrent_activation='sigmoid'), input_shape=(input[0], input[1])))
    # 添加Reshape层将LSTM的输出转换为3维
    model.add(Reshape((64, 1)))
    model.add(Conv1D(filters=32, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten()) # 将池化后的输出展平成一维向量
    model.add(Dense(32, kernel_initializer="uniform"))
    model.add(Dense(16, kernel_initializer="uniform"))
    model.add(Dense(1))
    ## CNN-BiLSTM                                                # test model-3
    # model.add(Dense(32,input_shape=(input[0],input[1])))
    # model.add(Bidirectional(LSTM(32, activation='relu'), input_shape=(input[0], input[1])))
    # # 添加Reshape层将LSTM的输出转换为3维
    # model.add(Reshape((64, 1)))
    # model.add(Conv1D(filters=32, kernel_size=1, activation='relu'))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Flatten()) # 将池化后的输出展平成一维向量
    # model.add(Dense(32, kernel_initializer="uniform"))
    # model.add(Dense(16, kernel_initializer="uniform"))
    # model.add(Dense(1))
    ## conv+LSTM                                                 # test model-2
    # model.add(Dense(32,input_shape=(input[0],input[1])))
    # model.add(Conv1D(filters=32,kernel_size=1,padding='same',activation='relu',kernel_initializer="glorot_uniform"))
    # model.add(MaxPooling1D(pool_size=2,padding='valid'))
    # model.add(LSTM(32,return_sequences=True))
    # model.add(LSTM(16,return_sequences=False))
    # model.add(Dense(32, kernel_initializer="uniform"))
    # model.add(Dense(1, kernel_initializer="uniform"))
    ## only LSTM                                                 # test model-1
    # model.add(LSTM(8,input_shape=(input[0],input[1])))
    # model.add(Dense(1))
    ## model compile
    model.compile(loss='mse',optimizer='adam',metrics=['mae'])
    return model
    # Summary of the Model
    # print(model.summary())

def model_evaluate(model,y_test):
    y_pred = model.predict(x_test)
    mse = metrics.mean_squared_error(y_test, np.array([i for arr in y_pred for i in arr]))
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_test, np.array([i for arr in y_pred for i in arr]))
    r2 = r2_score(y_test, np.array([i for arr in y_pred for i in arr]))
    return y_pred, mse, rmse, mae, r2

############ train for ssta #####################
# csvfile = 'ssta-PCA1-15.csv'
# histroy_path = '../model/all-Conv-BiLSTM-models-ssta/history-dict/'
# save_path = '../model/all-Conv-BiLSTM-models-ssta/'
# csvfile = 'ssta-PCA1-200.csv'
# histroy_path = '../model/finalmodel-ssta/history-dict/'
# save_path = '../model/finalmodel-ssta/'
############ train for adta #####################
# csvfile = 'adta-PCA1-15.csv'
# histroy_path = '../model/all-Conv-BiLSTM-models-adta/history-dict/'
# save_path = '../model/all-Conv-BiLSTM-models-adta/'
csvfile = 'adta-PCA1-200.csv'
histroy_path = '../model/finalmodel-adta/history-dict/'
save_path = '../model/finalmodel-adta/'
if not os.path.exists(histroy_path):
    os.makedirs(histroy_path)
if not os.path.exists(save_path):
        os.makedirs(save_path)
# In[1] main program
maxmimum_pc = 200
for k_pc in range(1,maxmimum_pc+1):
    for k_step in range(1,2): 
        print(f'PC{k_pc}-{k_step}')
        PCstr = f'PC{k_pc}'
        #### 数据加载与处理
        df = pd.read_csv(f'{csvfile}')
        df.head()
        features_considered = [PCstr]
        features = df[features_considered]
        features.index = df['Date']
        data0 = features.values
        train_size = 9860
        data,scaler = data_standardization(data0)
        histroy_time_steps = 31
        x_train, y_train, x_test, y_test = train_test_data_split(data,train_size,histroy_time_steps)
        #### 模型构建与训练
        model = build_model([30,1,1])
        start = timer()
        history = model.fit(x_train,y_train,batch_size=256,epochs=30,validation_split=0.1,verbose=2)
        end = timer()
        print(end - start)
        y_pred, mse, rmse, mae, r2 = model_evaluate(model,y_test)
        history_dict = history.history
        # plot_loss(history_dict)
        # plot_mae(history_dict)
        # history_dict.keys()
        # model.metrics_names
        #### 模型评估
        # trainScore = model.evaluate(x_train, y_train, verbose=0)
        # testScore = model.evaluate(x_test, y_test, verbose=0)
        y_pred = model.predict(x_test)
        # plot_y_test(y_test,y_pred)
        y_pred0 = data_anti_standardization(y_pred,scaler).reshape(-1,1)
        y_test0 = data0[train_size+histroy_time_steps-2:train_size+len(y_pred0)+histroy_time_steps-2]
        plot_y_test(y_test0,y_pred0) 
        r20 = r2_score(y_test0,y_pred0)
        #### 模型存储
        # histroy_file = f'{histroy_path}{features_considered[0]}-Conv-BiLSTM-{r2:.4f}-{rmse:.4f}-{r20:.4f}.npy'
        # np.save(histroy_file,history_dict)
        # model.save(f'{save_path}{features_considered[0]}-Conv-BiLSTM-{r2:.4f}-{rmse:.4f}-{r20:.4f}.h5')
        histroy_file = f'{histroy_path}{features_considered[0]}.npy'
        np.save(histroy_file,history_dict)
        model.save(f'{save_path}{features_considered[0]}.h5')
        #### 模型向后预测及可视化
        # last_output = model.predict(x_test)[-1]
        # future_steps = 10
        # predicted = []
        # for i in range(future_steps):
        #     # 将最后一个输出加入X_test，继续向后预测
        #     input_data = np.append(x_test[-1][1:], last_output).reshape(1, x_test.shape[1], x_test.shape[2])
        #     # 使用模型进行预测
        #     next_output = model.predict(input_data)
        #     # 将预测的值加入结果列表
        #     predicted.append(next_output[0][0])
        #     last_output = next_output[0]
        # predicted0 =  data_anti_standardization(np.array(predicted),scaler).reshape(-1,1)
        # plt.plot(range(0,len(y_test0)),y_test0,color='red',label='original data')
        # plt.plot(range(len(y_test0),len(y_test0)+future_steps),predicted0,color='blue',label='prediction')
        
