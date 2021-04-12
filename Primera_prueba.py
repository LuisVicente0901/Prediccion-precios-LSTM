# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 22:19:42 2021

@author: Luis Vicente
"""

#Tesis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

dataframe = pd.read_csv('C:/Users/Luis Vicente/Documents/Tesis/Proyecto Tesis/Prediccion-ML/datos_IPC.csv')

dataframe.info()

dataframe.head()

plt.figure(figsize = (15, 5))
plt.subplot(1, 1, 1)
plt.plot(dataframe['Close'], label = 'close')
plt.title('Stock Prices')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()


dataframe_close = dataframe['Close']
dataframe_close.head()


# creating 80% of training data and 20% of testing data
training_data_length = round((80/100 * len(dataframe_close)))

training_dataframe = dataframe_close[:training_data_length]
training_dataframe

testing_dataframe = dataframe_close[training_data_length:]
testing_dataframe

print(training_dataframe.shape)
training_dataframe = training_dataframe.values.reshape(-1, 1)
print(training_dataframe.shape)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_training_dataset = scaler.fit_transform(training_dataframe)
scaled_training_dataset

#scaled_training_dataset[0:1, 0]

# creating a moving window
X_train, y_train = [], []
moving_window_size = 60 # standard size for all time series analysis

for i in range(moving_window_size, len(scaled_training_dataset)):
    X_train.append(scaled_training_dataset[i-moving_window_size:i, 0])
    y_train.append(scaled_training_dataset[i, 0])

# we need to convert list into array to fed it into recurrent neural networks
X_train = np.array(X_train)
y_train = np.array(y_train)

print('X_train shape {0}'.format(X_train.shape))
print('y_train shape {0}'.format(y_train.shape))


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
y_train = y_train.reshape(y_train.shape[0], 1)
print('X_train shape {0}'.format(X_train.shape))
print('y_train shape {0}'.format(y_train.shape))


# RNN
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# initializing the neural network
rnn = Sequential()
rnn.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
rnn.add(Dropout(rate = 0.2))
rnn.add(LSTM(units = 50, return_sequences = True))
rnn.add(Dropout(rate = 0.2))
rnn.add(LSTM(units = 50, return_sequences = False))
rnn.add(Dropout(rate = 0.2))
rnn.add(Dense(units = 1))

# compiling the neural network
rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')

# fitting the model
rnn.fit(X_train, y_train, epochs = 10, batch_size = 32)

rnn.summary()


rnn.history.history.keys()
plt.plot(rnn.history.history['loss'], 'o-')
plt.xlabel('epochs')
plt.ylabel('loss')

# we are testing the model
print(testing_dataframe.shape)
testing_dataframe = testing_dataframe.values.reshape(-1, 1)
print(testing_dataframe.shape)

# scaling the testing dataframe
scaled_testing_dataframe = scaler.fit_transform(testing_dataframe)
#scaled_testing_dataframe[0]

# preparing the dataframe for testing
X_test , y_test = [], []
for i in range(moving_window_size, len(testing_dataframe)):
    X_test.append(scaled_testing_dataframe[i-moving_window_size : i, 0])
    y_test.append(scaled_testing_dataframe[i])


X_test = np.array(X_test)
X_test.shape
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
X_test.shape


predictions = rnn.predict(X_test)


plt.plot(y_test)
plt.plot(predictions)
xlabels = np.arange(len(y_test))
plt.plot(xlabels, y_test, label= 'Actual')
plt.plot(xlabels, predictions, label = 'Pred')
plt.legend()

#Promedio de error
((y_test-predictions).sum())/len(y_test)



