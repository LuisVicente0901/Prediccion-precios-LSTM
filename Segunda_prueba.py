# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 21:02:32 2021

@author: Luis Vicente
"""

#importing the libraries
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import random as rd
plt.style.use('fivethirtyeight')

data = pd.read_csv('C:/Users/Luis Vicente/Documents/Tesis/Proyecto Tesis/Prediccion-ML/datos_IPC.csv')

data

data.isnull().sum()

#data = data.loc[(data['symbol'] == 'AAPL')]
#data = data.drop(columns=['symbol'])
#data = data[['date','open','close','low','volume','high']]
#data

#We shall find out the number of rows and column in the dataset now.
data.shape

#Let us graphically represent the closing prices of the stock.
plt.figure(figsize=(16,8))
plt.title('Closing Price of the Stock Historically')
plt.plot(data['Close'])
plt.xlabel('Year', fontsize=20)
plt.ylabel('Closing Price Historically ($)', fontsize=20)
plt.show()


#LSTM Algorithm
#We need to create a seperate dataframe with the "close" column

data = data.filter(['Close'])
dataset = data.values

#Find out the number of rows that are present in this dataset in order to train our model.
training_data_len = math.ceil(len(dataset)* .8)
training_data_len

#Now, we need to scale the data

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data

#Create the training data for the model
train_data = scaled_data[0:training_data_len , :]
x_train = []
y_train = []

for j in range(60, len(train_data)):
    x_train.append(train_data[j-60:j,0])
    y_train.append(train_data[j,0])
    if j<=60:
        print(x_train)
        print(y_train)
        print()


x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape


#Building the LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

#Training the Model
model.fit(x_train, y_train, batch_size=10, epochs=40)


test_data = scaled_data[training_data_len - 60: , :]
x_test = []
y_test = dataset[training_data_len:, :]

for j in range(60, len(test_data)):
    x_test.append(test_data[j-60:j, 0])

x_test = np.array(x_test)    
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Finding out the Root Mean Squared Error
rmse = np.sqrt( np.mean( predictions - y_test)**2)
rmse


#Plot the graph
train = data[:training_data_len]
val = data[training_data_len:]
val['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('LSTM Model Data')
plt.xlabel('Date', fontsize=16)
plt.ylabel('Close Price', fontsize=16)
plt.plot(train['Close'])
plt.plot(val[['Close', 'Predictions']])
plt.legend(['Trained Dataset', 'Actual Value', 'Predictions'])
plt.show()

val

#Podemos gráficar solo los precios actuales y las predicciones sin el train

plt.figure(figsize=(16,8))
plt.title('LSTM Model Data')
plt.xlabel('Date', fontsize=16)
plt.ylabel('Close Price', fontsize=16)
plt.plot(val[['Close', 'Predictions']])
plt.legend(['Actual Value', 'Predictions'])
plt.show()


new_data = pd.read_csv('C:/Users/Luis Vicente/Documents/Tesis/Proyecto Tesis/Prediccion-ML/datos_IPC.csv')
new_data = data.filter(['Close'])
last_60_days = new_data[-60:].values
last_60_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_price = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price)
print('The predicted price of the final value of the dataset', predicted_price)

new_data.tail(1)
#EL precio predecido es de 45,833.83 MXN mientras que el valor actual observado es 45,909.308594 