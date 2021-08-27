# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 22:19:33 2021

@author: Luis Vicente
"""

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense
from keras.layers.normalization import BatchNormalization
plt.style.use('Solarize_Light2')

#Semillas para replicar los resultados: 9,2,4,5,7
np.random.seed(9)
tf.random.set_seed(9)

ipc = pd.read_csv('C:/Users/Luis Vicente/Documents/Tesis/Proyecto Tesis/Prediccion-ML/datos_IPC.csv')
ipc['Date']=pd.to_datetime(ipc['Date'])
ipc=ipc.drop(['Adj Close','Volume','High','Open','Low'],axis=1)
ipc['Date'] = pd.to_datetime(ipc['Date'])

df_final= ipc.set_index('Date')


dataset = df_final.values
print(dataset[:10])
print(dataset.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_dataset = scaler.fit_transform(df_final)
scaled_dataset

def split_sequence(sequence, n_steps):
    """Función que dividie el dataset en datos de entrada y datos que
    funcionan como etiquetas."""
    X, Y = [], []
    for i in range(sequence.shape[0]):
      if (i + n_steps) >= sequence.shape[0]:
        break
      # Divide los datos entre datos de entrada (input) y etiquetas (output)
      seq_X, seq_Y = sequence[i: i + n_steps], sequence[i + n_steps, 0]
      X.append(seq_X)
      Y.append(seq_Y)
    return np.array(X), np.array(Y)

dataset_size = scaled_dataset.shape[0]
index_train=ipc.index[ipc['Date']=='2014-12-31'].tolist()
index_validation=ipc.index[ipc['Date']=='2015-12-31'].tolist()

#Entrenamiento y validación
x_train, y_train = split_sequence(scaled_dataset[0: index_train[0]+1], 30)
x_val, y_val = split_sequence(scaled_dataset[index_train[0]+1:index_validation[0]+1], 30)


print("dataset.shape: {}".format(dataset.shape))
print("df.shape: {}".format(ipc.shape))
print("x_train.shape: {}".format(x_train.shape))
print("y_train.shape: {}".format(y_train.shape))
print("x_val.shape: {}".format(x_val.shape))
print("y_val.shape: {}".format(y_val.shape))
print('=======================')


batch_size = 100
buffer_size = x_train.shape[0]
# Crea un conjunto de datos que incrementa de acuerdo a lo que se necesite
train_iterator = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size).batch(batch_size).repeat()
# Crea un conjunto de datos que incrementa de acuerdo a lo que se necesite
val_iterator = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).repeat()


n_steps = x_train.shape[-2]
n_features = x_train.shape[-1]

#Se define el modelo
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='tanh',return_sequences=False,input_shape=(n_steps, n_features))))
model.add(BatchNormalization(center=False))
model.add(Dense(1))

# Compilación del modelo 
model.compile(optimizer='adam', loss='mse')


epochs = 60
steps_per_epoch = 95
validation_steps = 45
# Entrenar y validar el modelo
history = model.fit(train_iterator, epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=val_iterator,
                    validation_steps=validation_steps)


x_test,y_test=split_sequence(scaled_dataset[index_validation[0]+1:], 30)
print(x_test.shape)
print(y_test.shape)


predictions = model.predict(x_test)

plt.plot(y_test)
plt.plot(predictions)
xlabels = np.arange(len(y_test))
plt.plot(xlabels, y_test, label= 'Actual')
plt.plot(xlabels, predictions, label = 'Pred')
plt.legend()




#MSE

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test,predictions)


scaler.inverse_transform(predictions)
#Predecir el siguiente día 

a=[]
a.append(scaled_dataset[1718:1748])
#scaler.inverse_transform(a)
a=np.array(a)
a.shape
prediction_1=model.predict(a)
scaler.inverse_transform(prediction_1)



b=np.empty(30)
for i in range(30):
    if(i<29):
        b[i]=scaled_dataset[1719+i]
    else:
        b[i]=prediction_1
        
b=np.array(b)
b=b.reshape(1,b.shape[0],1)
b.shape
prediction_2=model.predict(b)
scaler.inverse_transform(prediction_2)

c=np.empty(30)
for i in range(30):
    if(i<28):
        c[i]=scaled_dataset[1720+i]
    elif(i==28):
        c[i]=prediction_1
    else:
        c[i]=prediction_2
        
c=np.array(c)
c=c.reshape(1,c.shape[0],1)
c.shape
prediction_3=model.predict(c)
scaler.inverse_transform(prediction_3)



#Hacerlo de manera automatizada las predicciones
repeticiones=4
arreglo=np.empty(30+repeticiones)
for i in range(30):
    arreglo[i]=scaled_dataset[1718+i]

for j in range(repeticiones):
    datos=np.empty(30)
    for i in range(30):
        datos[i]=arreglo[i+j]
    datos=np.array(datos)
    datos=datos.reshape(1,datos.shape[0],1)
    predi=model.predict(datos)
    arreglo[30+j]=predi
    print(scaler.inverse_transform(np.array(arreglo[30+j]).reshape(1,1)))
    
    
#scaler.inverse_transform(np.array(.83).reshape(1,1))
      
        
    