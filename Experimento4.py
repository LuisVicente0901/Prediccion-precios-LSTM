# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 23:14:08 2021

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

a_movil = pd.read_csv('C:/Users/Luis Vicente/Documents/Tesis/Proyecto Tesis/Prediccion-ML/datos_A_Movil.csv')
a_movil['Date']=pd.to_datetime(a_movil['Date'])
a_movil=a_movil.drop(['Adj Close','Volume','High','Open','Low'],axis=1)


from functools import reduce
dfs = [ipc,a_movil]
df = reduce(lambda left,right: pd.merge(left,right,on='Date'), dfs)
df.columns=['Date','IPC','America Movil']


df_final= df.set_index('Date')


dataset = df_final.values
print(dataset[:10])
print(dataset.shape)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_dataset = scaler.fit_transform(df_final)
scaled_dataset

#La función MinMaxScaler hace esto
#min(df_final.iloc[:,0])
#max(df_final.iloc[:,0])
#(df_final.iloc[0,0]-min(df_final.iloc[:,0]))/(max(df_final.iloc[:,0])-min(df_final.iloc[:,0]))




def split_sequence(sequence, n_steps):
    """Función que dividie el dataset en datos de entrada y datos que
    funcionan como etiquetas."""
    X, Y = [], []
    for i in range(sequence.shape[0]):
      if (i + n_steps) >= sequence.shape[0]:
        break
      # Divide los datos entre datos de entrada (input) y etiquetas (output)
      seq_X, seq_Y = sequence[i: i + n_steps], sequence[i + n_steps, 1]
      X.append(seq_X)
      Y.append(seq_Y)
    return np.array(X), np.array(Y)

dataset_size = scaled_dataset.shape[0]
index_train=df.index[df['Date']=='2014-12-31'].tolist()
index_validation=df.index[df['Date']=='2015-12-31'].tolist()

#Entrenamiento y validación
x_train, y_train = split_sequence(scaled_dataset[0: index_train[0]+1], 30)
x_val, y_val = split_sequence(scaled_dataset[index_train[0]+1:index_validation[0]+1], 30)


print("dataset.shape: {}".format(dataset.shape))
print("df.shape: {}".format(df.shape))
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



def plot_1(history, title):
  """función que grafica los errores obtenidos del modelo"""
  plt.figure(figsize=(8,6))
  plt.plot(history.history['loss'], 'o-', mfc='none', markersize=10, 
  label='Train',color='deepskyblue')
  plt.plot(history.history['val_loss'], 'o-', mfc='none', 
  markersize=10, label='Validation',color='green')
  plt.title('Curva de aprendizaje')
  plt.xlabel('Epocas')
  plt.ylabel('Error cuadrático medio')
  plt.legend()
  plt.show()
  
# Gráfica de la curva de aprendizaje del modelo en los conjuntos de 
#entrenamiento y validación
plot_1(history, 'Training / Validation Losses from History')


x_test,y_test=split_sequence(scaled_dataset[index_validation[0]+1:], 30)
print(x_test.shape)
print(y_test.shape)


predictions = model.predict(x_test)



plt.plot(y_test)
plt.plot(predictions)
xlabels = np.arange(len(y_test))
plt.plot(xlabels, y_test, label= 'Actual',color='deepskyblue')
plt.plot(xlabels, predictions, label = 'Predicción',color='green')
plt.legend()


#MSE

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test,predictions)


#Predecir el siguiente día 

a=[]
a.append(scaled_dataset[1718:1748])
#scaler.inverse_transform(a)
a=np.array(a)
a.shape
prediction_1=model.predict(a)

(prediction_1*(max(df_final.iloc[:,1])-min(df_final.iloc[:,1])))+min(df_final.iloc[:,1])


#Grafica de barras
configuraciones = ['LSTM Bidireccional \ncon BN', 'LSTM\n Bidireccional', 'LSTM con BN', 'LSTM']
predicciones = [13.05,13.14,12.54,13.06]
plt.bar(configuraciones,predicciones,color=['red','blue','orange','brown'])
plt.xticks( configuraciones,  ha='center', rotation=0, fontsize=9.5, fontname='monospace')
plt.axhline(y=13.03, color='black', linestyle='-')
plt.ylim(12.3,13.3)
plt.show()


#Grafica de puntos
plt.plot(13.03,marker="o",label='Real',color='red')
plt.plot(13.053839,marker="x",label='LSTM Bidireccional con BN',color='royalblue',markersize=8.5)
plt.plot(13.144364,marker="x",label='LSTM Bidireccional',color='black',markersize=8.5)
plt.plot(12.54138,marker="x",label='LSTM con BN',color='orange',markersize=8.5)
plt.plot(13.066681,marker="x",label='LSTM',color='green',markersize=8.5)
plt.legend(fontsize=8)

#seed(9)
#Bidirectional y BatchNormalization
#0.0007445248210100825
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.053839

#Bidirectional 
#0.0009481819618347559
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.144364

#Unidireccional y BatchNormalization
#0.003563969546647305
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 12.54138

#Unidireccional 
#0.0007858096070223015
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.066681

#seed(2)
#Bidirectional y BatchNormalization
#0.000829304766925709
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.0581

#Bidirectional 
#0.0007526004042428698
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.009533

#Unidireccional y BatchNormalization
#0.0008176681355401593
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.044346

#Unidireccional 
#0.0007943241409002743
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.057686

#seed(4)
#Bidirectional y BatchNormalization
#0.0008037687170623582
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 12.96292

#Bidirectional 
#0.0009385170492834545
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.104019

#Unidireccional y BatchNormalization
#0.0017478837533404538
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.270288

#Unidireccional 
#0.0008094745281384949
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.061566

#seed(5)
#Bidirectional y BatchNormalization
#0.003366079380869385
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.432211

#Bidirectional 
#0.0008208913132246707
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.076336

#Unidireccional y BatchNormalization
#0.0022102112323744397
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.223236

#Unidireccional 
#0.0009094302506841915
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.104133

#seed(7)
#Bidirectional y BatchNormalization
#0.01166928884459115
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 12.308899

#Bidirectional 
#0.000873895257094298
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.074442

#Unidireccional y BatchNormalization
#0.003038200003602742
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.38566 

#Unidireccional 
#0.00077557212151561
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 12.995448

#Media
np.mean((0.00074,0.00082,0.0008,0.00336,0.01166))
np.mean((0.00094,0.00075,0.00093,0.00082,0.00087))
np.mean((0.00356,0.00081,0.00174,0.00221,0.00303))
np.mean((0.00078,0.00079,0.0008,0.0009,0.00077))


#Desviación estándar
np.std((0.00074,0.00082,0.0008,0.00336,0.01166))
np.std((0.00094,0.00075,0.00093,0.00082,0.00087))
np.std((0.00356,0.00081,0.00174,0.00221,0.00303))
np.std((0.00078,0.00079,0.0008,0.0009,0.00077))