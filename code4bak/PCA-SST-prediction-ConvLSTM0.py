# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 23:33:02 2024

@author: Shmily
"""

import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from keras.layers import Dense, Activation ,Dropout , Flatten , Conv1D , MaxPooling1D
from keras.layers.recurrent import LSTM
from keras import losses
from keras import optimizers
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)

def create_time_steps(length):
    return list(range(-length, 0))
def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])  # 返回-20到-1的列表
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt

def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# zip_path = tf.keras.utils.get_file(
#     origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
#     fname='jena_climate_2009_2016.csv.zip',
#     extract=True)
csv_path = './PCA1-15.csv'
df = pd.read_csv(csv_path)

uni_data = df['PC1']
uni_data.index = df['Date']
uni_data = uni_data.values

TRAIN_SPLIT = 9860  
# uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
# uni_train_std = uni_data[:TRAIN_SPLIT].std()
# uni_data = (uni_data-uni_train_mean)/uni_train_std
scaler = MinMaxScaler(feature_range=(0, 1))
uni_data= scaler.fit_transform(uni_data.reshape(-1,1))

univariate_past_history = 30
univariate_future_target = 1
STEP = 1
x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)
BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

# simple_lstm_model = tf.keras.models.Sequential([
#     tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
#     tf.keras.layers.Dense(1)
# ])
simple_lstm_model = tf.keras.models.Sequential()
simple_lstm_model.add(Dense(32,input_shape=(univariate_past_history,1)))
simple_lstm_model.add(Conv1D(filters=32,kernel_size=1,padding='same',activation='relu',kernel_initializer="glorot_uniform"))
simple_lstm_model.add(MaxPooling1D(pool_size=2,padding='valid'))
simple_lstm_model.add(LSTM(32,return_sequences=True))
simple_lstm_model.add(LSTM(16,return_sequences=False))
simple_lstm_model.add(Dense(32, kernel_initializer="uniform"))
simple_lstm_model.add(Dense(1, kernel_initializer="uniform"))
simple_lstm_model.compile(loss='mse',optimizer='adam',metrics=['mae'])

# simple_lstm_model.compile(optimizer='adam', loss='mae')

EVALUATION_INTERVAL = 200
EPOCHS = 25

simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50,verbose=1)

for x, y in val_univariate.take(1):
    plot = show_plot([x[0].numpy(), y[0].numpy(),
                    simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
    plot.show()
 
c = simple_lstm_model.predict(x_val_uni)
# c = simple_lstm_model.predict(x_val_uni[0].reshape(1,40,1))
# pc = simple_lstm_model.predict(x)[0] * uni_train_std + uni_train_mean

plt.figure(figsize=(12, 6))
plt.plot(range(0,len(y_val_uni)),y_val_uni, label='Truth', color='blue', linestyle='-')
plt.plot(range(0,len(y_val_uni)),c, label='Truth', color='red', linestyle='-')
    