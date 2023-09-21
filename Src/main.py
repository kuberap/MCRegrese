import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, losses, models


def get_model(units=[32,32,32], dropouts=[0.1,0.1,0.1], activations=['relu', 'relu', 'relu']):
  model = models.Sequential()
  model.add(layers.InputLayer(input_shape = (IN_DIM,)))
  for u,d,a in zip(units, dropouts, activations)
  model.add(layers.Dense(units = 32, activation = ))
  model.add(layers.Dense(units = 32, activation='relu'))
  model.add(layers.Dense(units=32, activation='tanh'))
  model.add(layers.Dense(OUT_DIM, activation='tanh'))
  model.compile(loss='mse', optimizer='adam', metrics=['mse',"mae"])
  return model


EPOCHS = 5
IN_DIM = 21
OUT_DIM = 4

if __name__=="__main__":

   df = pd.read_csv("../Data/data_for_NN_test.txt")

   X = df[[f"layer{i}" for i in range(1,IN_DIM+1)]]
   y = df[["A1", "A2","T1","T2"]]
   print(y.describe())

   X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

   scaler_x = MinMaxScaler()
   scaler_y = MinMaxScaler(feature_range=(0, 1))
   X_train_sc = scaler_x.fit_transform(X_train)
   X_test_sc = scaler_x.transform(X_test)

   y_train_sc = scaler_y.fit_transform(y_train)
   y_test_sc = scaler_y.transform(y_test)

   model = get_model()
   print(model.summary())
   history = model.fit(X_train_sc, y_train_sc, epochs=EPOCHS, validation_split=0.1)



   plt.plot(history.history['loss'], label='loss- training data')
   plt.plot(history.history['val_loss'], label='loss - validating data')
   plt.grid()
   plt.legend()
   plt.show()



