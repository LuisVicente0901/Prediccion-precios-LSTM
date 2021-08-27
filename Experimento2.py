# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 19:34:37 2021

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

walmex = pd.read_csv('C:/Users/Luis Vicente/Documents/Tesis/Proyecto Tesis/Prediccion-ML/datos_WALMEX.csv')
walmex['Date']=pd.to_datetime(walmex['Date'])
walmex=walmex.drop(['Adj Close','Volume','High','Open','Low'],axis=1)

femsa = pd.read_csv('C:/Users/Luis Vicente/Documents/Tesis/Proyecto Tesis/Prediccion-ML/datos_Femsa.csv')
femsa['Date']=pd.to_datetime(femsa['Date'])
femsa=femsa.drop(['Adj Close','Volume','High','Open','Low'],axis=1)

### EL FEMSA TIENE UN VALOR NAN, SUSTITUIRLO CON EL PROMEDIO DE LOS PRECIOS ANTERIORES Y POSTERIORES
#null_columns=femsa.columns[femsa.isnull().any()]
#femsa[null_columns].isnull().sum()
#print(femsa[femsa.isnull().any(axis=1)][null_columns].head())
femsa.iloc[49,1]=((femsa.iloc[47:49,1].sum()+femsa.iloc[50:52,1].sum())/4)

televisa = pd.read_csv('C:/Users/Luis Vicente/Documents/Tesis/Proyecto Tesis/Prediccion-ML/datos_Televisa.csv')
televisa['Date']=pd.to_datetime(televisa['Date'])
televisa=televisa.drop(['Adj Close','Volume','High','Open','Low'],axis=1)

gfnorte = pd.read_csv('C:/Users/Luis Vicente/Documents/Tesis/Proyecto Tesis/Prediccion-ML/datos_GFNorte.csv')
gfnorte['Date']=pd.to_datetime(gfnorte['Date'])
gfnorte=gfnorte.drop(['Adj Close','Volume','High','Open','Low'],axis=1)


from functools import reduce
dfs = [ipc,a_movil,walmex,femsa,televisa,gfnorte]
df = reduce(lambda left,right: pd.merge(left,right,on='Date'), dfs)
df.columns=['Date','IPC','America Movil','Walmex','Femsa','Televisa','GFNorte']


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
      seq_X, seq_Y = sequence[i: i + n_steps], sequence[i + n_steps, 0]
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
a.append(scaled_dataset[1575:1605])
#scaler.inverse_transform(a)
a=np.array(a)
a.shape
prediction_1=model.predict(a)

(prediction_1*(max(df_final.iloc[:,0])-min(df_final.iloc[:,0])))+min(df_final.iloc[:,0])


#Graficas de barras
configuraciones = ['LSTM Bidireccional \ncon BN', 'LSTM\n Bidireccional', 'LSTM con BN', 'LSTM']
predicciones = [45699.902,45751.055,44931.34,45619.64]
plt.bar(configuraciones,predicciones,color=['red','blue','orange','brown'])
plt.xticks( configuraciones,  ha='center', rotation=0, fontsize=9.5, fontname='monospace')
plt.axhline(y=45695.10, color='black', linestyle='-')
plt.ylim(44000,46000)
plt.show()


#Graficas de puntos 
plt.plot(45695.10,marker="o",label='Real',color='red')
plt.plot(45699.902,marker="x",label='LSTM Bidireccional con BN',color='royalblue',markeredgewidth=2,markersize=8.5)
plt.plot(45751.055,marker="x",label='LSTM Bidireccional',color='black',markersize=8.5)
plt.plot(44931.34,marker="x",label='LSTM con BN',color='orange',markersize=8.5)
plt.plot(45619.64,marker="x",label='LSTM',color='green',markersize=8.5)
plt.legend(fontsize=8)

#seed(9)
#Bidireccional y BachNormalization
# 0.003761850351631536
#El valor predicho del 2 de enero del 2017 es 45699.902

#Bidireccioanl
#0.0048734056208300834
#El valor predicho del 2 de enero del 2017 es 45751.055

#LSTM y BachNormalization
#0.001817826130900373
#El valor predicho del 2 de enero del 2017 es 44931.34

#LSTM
#0.000984006427999835
#El valor predicho del 2 de enero del 2017 es 45619.64

#seed(2)
#Bidireccional y BachNormalization
# 0.001006917153648329
#El valor predicho del 2 de enero del 2017 es 44840.586

#Bidireccioanl
#0.0009056350824921962
#El valor predicho del 2 de enero del 2017 es 44865.73

#LSTM y BachNormalization
#0.0005480880156096802
#El valor predicho del 2 de enero del 2017 es 45722.684

#LSTM
#0.0007451930841086339
#El valor predicho del 2 de enero del 2017 es 45214.93

#seed(4)
#Bidireccional y BachNormalization
# 0.0032394235262259824
#El valor predicho del 2 de enero del 2017 es 44045.703

#Bidireccioanl
#0.0009872125602345498
#El valor predicho del 2 de enero del 2017 es 44852.133

#LSTM y BachNormalization
#0.002499917108755281
#El valor predicho del 2 de enero del 2017 es 44706.957

#LSTM
#0.0009599544670236326
#El valor predicho del 2 de enero del 2017 es 45184.59

#seed(5)
#Bidireccional y BachNormalization
# 0.0011215041277672798
#El valor predicho del 2 de enero del 2017 es 45394.83

#Bidireccioanl
#0.0027383477903983517
#El valor predicho del 2 de enero del 2017 es 44848.89

#LSTM y BachNormalization
#0.003883271260674369
#El valor predicho del 2 de enero del 2017 es 44763.71

#LSTM
#0.0007262330475428146
#El valor predicho del 2 de enero del 2017 es 45132.188

#seed(7)
#Bidireccional y BachNormalization
# 0.005417665946540367
#El valor predicho del 2 de enero del 2017 es 43853.812

#Bidireccioanl
#0.0013219006827103795
#El valor predicho del 2 de enero del 2017 es 44951.445

#LSTM y BachNormalization
#0.0028611877451148785
#El valor predicho del 2 de enero del 2017 es 44790.953

#LSTM
#0.0005966969866985619
#El valor predicho del 2 de enero del 2017 es 45424.934


#Media
np.mean((0.00376,0.001,0.00323,0.00112,0.00541))
np.mean((0.00487,0.0009,0.00098,0.00273,0.00132))
np.mean((0.00181,0.00054,0.00249,0.00388,0.00286))
np.mean((0.00098,0.00074,0.00095,0.00072,0.00059))


#Desviación estándar
np.std((0.00376,0.001,0.00323,0.00112,0.00541))
np.std((0.00487,0.0009,0.00098,0.00273,0.00132))
np.std((0.00181,0.00054,0.00249,0.00388,0.00286))
np.std((0.00098,0.00074,0.00095,0.00072,0.00059))
