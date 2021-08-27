# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 23:06:48 2021

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

a_movil = pd.read_csv('C:/Users/Luis Vicente/Documents/Tesis/Proyecto Tesis/Prediccion-ML/datos_A_Movil.csv')
a_movil['Date']=pd.to_datetime(a_movil['Date'])
a_movil=a_movil.drop(['Adj Close','Volume','High','Open','Low'],axis=1)
a_movil['Date'] = pd.to_datetime(a_movil['Date'])

df_final= a_movil.set_index('Date')


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
index_train=a_movil.index[a_movil['Date']=='2014-12-31'].tolist()
index_validation=a_movil.index[a_movil['Date']=='2015-12-31'].tolist()

#Entrenamiento y validación
x_train, y_train = split_sequence(scaled_dataset[0: index_train[0]+1], 30)
x_val, y_val = split_sequence(scaled_dataset[index_train[0]+1:index_validation[0]+1], 30)


print("dataset.shape: {}".format(dataset.shape))
print("df.shape: {}".format(a_movil.shape))
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
a.append(scaled_dataset[1721:1751])
#scaler.inverse_transform(a)
a=np.array(a)
a.shape
prediction_1=model.predict(a)
scaler.inverse_transform(prediction_1)

#scaler.inverse_transform(predictions)


#Grafica de barras
configuraciones = ['LSTM Bidireccional \ncon BN', 'LSTM\n Bidireccional', 'LSTM con BN', 'LSTM']
predicciones = [13.19,13.07,12.978207,13.101506]
plt.bar(configuraciones,predicciones,color=['red','blue','orange','brown'])
plt.xticks( configuraciones,  ha='center', rotation=0, fontsize=9.5, fontname='monospace')
plt.axhline(y=13.03, color='black', linestyle='-')
plt.ylim(12.5,13.5)
plt.show()


#Grafica de puntos 
plt.plot(13.03,marker="o",label='Real',color='red',markersize=7)
plt.plot(13.190034,marker="x",label='LSTM Bidireccional con BN',color='royalblue',markersize=8.5)
plt.plot(13.078709,marker="x",label='LSTM Bidireccional',color='black',markersize=8.5)
plt.plot(12.978207,marker="x",label='LSTM con BN',color='orange',markersize=8.5)
plt.plot(13.101506,marker="x",label='LSTM',color='green',markeredgewidth=2,markersize=8.5)
plt.legend(fontsize=8)

#seed(9)
#Bidirectional y BatchNormalization
#0.0010863048992674233
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.190034

#Bidirectional 
#0.0008398531489357989
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.078709

#Unidireccional y BatchNormalization
#0.0012079798514902421
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 12.978207

#Unidireccional 
#0.001038070724156784
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.101506

#seed(2)
#Bidirectional y BatchNormalization
#0.0009793315972119057
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.189381

#Bidirectional 
#0.000771297461095305
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.056782

#Unidireccional y BatchNormalization
#0.0009281003792834392
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.14103

#Unidireccional 
#0.0008302280052797988
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.084866

#seed(4)
#Bidirectional y BatchNormalization
#0.0007611898226357818
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.047703

#Bidirectional 
#0.0009192615760150387
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.054613

#Unidireccional y BatchNormalization
#0.0008258038668196
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 12.948607

#Unidireccional 
#0.000778925661109494
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.066179

#seed(5)
#Bidirectional y BatchNormalization
#0.0008145186064425515
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.086187

#Bidirectional 
#0.0008293386830097381
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.028512

#Unidireccional y BatchNormalization
#0.0009075414551025184
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.156241

#Unidireccional 
#0.0008004594053161302
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.055098

#seed(7)
#Bidirectional y BatchNormalization
#0.000941814240991467
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.149855

#Bidirectional 
#0.0008822328365876882
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.085762

#Unidireccional y BatchNormalization
#0.0009189446739004243
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.156306 

#Unidireccional 
#0.0008452554666710617
#El valor del 2 de enero del 2017 es 13.03 y el predicho es 13.080153


#Media
np.mean((0.00108,0.00097,0.00076,0.00081,0.00094))
np.mean((0.00083,0.00077,0.00091,0.00082,0.00088))
np.mean((0.0012,0.00092,0.00082,0.0009,0.00091))
np.mean((0.00103,0.00083,0.00077,0.0008,0.00084))

#Desviación estándar
np.std((0.00108,0.00097,0.00076,0.00081,0.00094))
np.std((0.00083,0.00077,0.00091,0.00082,0.00088))
np.std((0.0012,0.00092,0.00082,0.0009,0.00091))
np.std((0.00103,0.00083,0.00077,0.0008,0.00084))
