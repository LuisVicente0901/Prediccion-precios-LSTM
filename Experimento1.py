# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 21:31:49 2021

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



#Plot de la comparación entre lo predicho y los valores actuales no escalados
#y_test=y_test.reshape(y_test.shape[0],1)
# plt.plot(scaler.inverse_transform(y_test))
# plt.plot(scaler.inverse_transform(predictions))
# xlabels = np.arange(len(y_test))
# plt.plot(xlabels, scaler.inverse_transform(y_test), label= 'Actual')
# plt.plot(xlabels, scaler.inverse_transform(predictions), label = 'Pred')
# plt.legend()


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
scaler.inverse_transform(prediction_1)

scaler.inverse_transform(predictions)


#Graficas de barras
configuraciones = ['LSTM Bidireccional \ncon BN', 'LSTM\n Bidireccional', 'LSTM con BN', 'LSTM']
predicciones = [45897.05,45671.406,46458.83,45698.582]
plt.bar(configuraciones,predicciones,color=['red','blue','orange','brown'])
plt.xticks( configuraciones,  ha='center', rotation=0, fontsize=9.5, fontname='monospace')
plt.axhline(y=45695.10, color='black', linestyle='-')
plt.ylim(45000,46600)
plt.show()


#Grafica de puntos 
plt.plot('2 de enero\n del 2017',45695.10,marker="o",label='Real',color='red',markersize=7)
plt.plot(45897.05,marker="x",label='LSTM Bidireccional con BN',color='royalblue',markersize=8.5)
plt.plot(45671.406,marker="x",label='LSTM Bidireccional',color='black',markersize=8.5)
plt.plot(46458.83,marker="x",label='LSTM con BN',color='orange',markersize=8.5)
plt.plot(45698.582,marker="x",label='LSTM',color='green',markeredgewidth=2,markersize=8.5)
plt.legend(fontsize=8)
#Calcular el MSE agregando el valor del día 2 de enero del 2017 para poder
#escalar los datos y obtener el MSE escalado
# arreglo=np.empty(len(dataset)+1)
# for i in range(len(dataset)+1):
#     if(i<len(dataset)):
#         arreglo[i]=dataset[i]
#     else:
#         arreglo[i]=45695.10

# mean = np.mean(arreglo, axis=0)
# stddev = np.std(arreglo, axis=0)
# scaled = (45695.10 - mean) / stddev
# print(scaled)


# mean_squared_error([1.177541438186244], [0.8417416])

# plt.plot(0,1.177541,marker="o")
# plt.plot(0,0.84174,marker="o")


#seed(9)
#Bidirectional y BatchNormalization
#0.0006737754349927639
#El valor del 2 de enero del 2017 es 45695.10 y el predicho es 45897.05

#Bidirectional 
#0.00043528094162891397
#El valor del 2 de enero del 2017 es 45695.10 y el predicho es 45671.406

#Unidireccional y BatchNormalization
#0.002755068110756451
#El valor del 2 de enero del 2017 es 45695.10 y el predicho es 46458.83

#Unidireccional 
#0.00043989147267038176
#El valor del 2 de enero del 2017 es 45695.10 y el predicho es 45698.582

#seed(2)
#Bidirectional y BatchNormalization
#0.0005014814286786162
#El valor del 2 de enero del 2017 es 45695.10 y el predicho es 45498.992

#Bidirectional 
#0.00043923833615363173
#El valor del 2 de enero del 2017 es 45695.10 y el predicho es 45590.027

#Unidireccional y BatchNormalization
#0.0005172228903057433
#El valor del 2 de enero del 2017 es 45695.10 y el predicho es 45764.223

#Unidireccional 
#0.0004454334891662235
#El valor del 2 de enero del 2017 es 45695.10 y el predicho es 45526.484

#seed(4)
#Bidirectional y BatchNormalization
#0.0017666400192423509
#El valor del 2 de enero del 2017 es 45695.10 y el predicho es 44971.17

#Bidirectional 
#0.0004505057394983748
#El valor del 2 de enero del 2017 es 45695.10 y el predicho es 45732.83

#Unidireccional y BatchNormalization
#0.0011706685727812963
#El valor del 2 de enero del 2017 es 45695.10 y el predicho es 45132.242

#Unidireccional 
#0.00043285889887170823
#El valor del 2 de enero del 2017 es 45695.10 y el predicho es 45572.133

#seed(5)
#Bidirectional y BatchNormalization
#0.0013884488190202282
#El valor del 2 de enero del 2017 es 45695.10 y el predicho es 46162.98

#Bidirectional 
#0.0004542824577311829
#El valor del 2 de enero del 2017 es 45695.10 y el predicho es 45552.992

#Unidireccional y BatchNormalization
#0.00133331682526621
#El valor del 2 de enero del 2017 es 45695.10 y el predicho es 45104.992

#Unidireccional 
#0.0004442473112291213
#El valor del 2 de enero del 2017 es 45695.10 y el predicho es 45571.082

#seed(7)
#Bidirectional y BatchNormalization
#0.005655586767932898
#El valor del 2 de enero del 2017 es 45695.10 y el predicho es 44363.152

#Bidirectional 
#0.0005111434378563864
#El valor del 2 de enero del 2017 es 45695.10 y el predicho es 45463.867

#Unidireccional y BatchNormalization
#0.004103502604736718
#El valor del 2 de enero del 2017 es 45695.10 y el predicho es 44604.656

#Unidireccional 
#0.00048794720841843173
#El valor del 2 de enero del 2017 es 45695.10 y el predicho es 45480.582 


#Media
np.mean((0.00067,0.0005,0.00176,0.00138,0.00565))
np.mean((0.00043,0.00043,0.00045,0.00045,0.00051))
np.mean((0.00275,0.00051,0.00117,0.00133,0.0041))
np.mean((0.00043,0.00044,0.00043,0.00044,0.00048))


#Desviación estándar
np.std((0.00067,0.0005,0.00176,0.00138,0.00565))
np.std((0.00043,0.00043,0.00045,0.00045,0.00051))
np.std((0.00275,0.00051,0.00117,0.00133,0.0041))
np.std((0.00043,0.00044,0.00043,0.00044,0.00048))

