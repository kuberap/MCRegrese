import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import tensorflow as tf

from Src.models import get_model

# HACK hazi to warningy pro distribuovane pcitani, tak je potlacuji. Zdroj moznych problemu.
# =========================================================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# =========================================================================================


EPOCHS = 20
EPOCHS_PATIENCE_ES = EPOCHS//5 # kdyz se to po danem poctu epoch nezlepsi, tak to stopni
EPOCHS_PATIENCE_RP = EPOCHS//10 # kdyz se to na validacnich datech nezlepsuje po danem poctu epoch, tak zmenis learning rate
BATCH_SIZE =  128 # bude prepsano pri optimalizaci hyperparametru
SPLITS = 5
IN_DIM = 21
OUT_DIM = 4

assert  EPOCHS_PATIENCE_RP < EPOCHS_PATIENCE_ES, f" Patience for Early stops {EPOCHS_PATIENCE_ES} must be less than paience for ReduceonPlato {EPOCHS_PATIENCE_RP}"


def run_k_fold(model, scaler_x,scaler_y, X,y, batch_size, val_ratio = 0.1):
   """
   Spusti vicenasobnou validaci.
   Predpoklada se vstup vytvoreneho modelu a scaleru
   :param model:
   :param scaler_x:
   :param sclaer_y:
   :param X:
   :param y:
   :return:
   """
   kf = KFold(n_splits=SPLITS, shuffle=True)
   mse = []
   for fold_index, (train_index, test_index) in enumerate(kf.split(X, y)):
      print(f'FOLD: {fold_index}')
      # Vybere data pro jednotlive foldy a naskaluje je
      X_train, X_test, y_train, y_test = X[train_index,:], X[test_index,:], y[train_index], y[test_index]
      X_train_sc = scaler_x.fit_transform(X_train)
      X_test_sc = scaler_x.transform(X_test)
      y_train_sc = scaler_y.fit_transform(y_train)
      y_test_sc = scaler_y.transform(y_test)
      # pro paralelizaci na vice GPU
      tr_samples = int(X_train_sc.shape[0]*(1-val_ratio))
      train_dataset = tf.data.Dataset.from_tensor_slices((X_train_sc[:tr_samples], y_train_sc[:tr_samples])).cache().batch(batch_size=batch_size)
      val_dataset = tf.data.Dataset.from_tensor_slices((X_train_sc[tr_samples:], y_train_sc[tr_samples:])).cache().batch(batch_size=batch_size)
      test_dataset = tf.data.Dataset.from_tensor_slices((X_test_sc, y_test_sc)).batch(batch_size=batch_size)
      # nauc model
      early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=EPOCHS_PATIENCE_ES) #kdyz se to dany pocet epoch nezlepsi stopni
      reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=EPOCHS_PATIENCE_RP, min_lr=0.001)
      model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, verbose=0, callbacks=[early_stop_cb, reduce_lr_cb])
      # udelej predikci na testovacich datech
      # data jsou preskalovana na puvodni rozmer, abych odstranil vliv intervalu skalovani
      y_hat_sc = model.predict(test_dataset, verbose=False)
      y_hat = scaler_y.inverse_transform(y_hat_sc)
      mse.append(mean_squared_error(y_test, y_hat))
   print(mse)
   return np.array(mse)


def train_evalate_model(config, X, y):
   scaler_x = MinMaxScaler()
   scaler_y = MinMaxScaler(feature_range=(-1, 1))  # TODO range parametrizovat
   # TODO paraleliyace na groot ?? test
   # strategy = tf.distribute.MirroredStrategy()  # vem si mirror paralelni strategii
   # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
   # if strategy.num_replicas_in_sync > 1:
   #    with strategy.scope():
   #       model = get_model(in_dim=IN_DIM, out_dim=OUT_DIM)
   #       model.compile(loss='mse', optimizer='adam', metrics=['mse', "mae"])  # TODO optimizer parametrizovat
   # else:
   model = get_model(in_dim=IN_DIM, out_dim=OUT_DIM)
   model.compile(loss='mse', optimizer='adam', metrics=['mse', "mae"])  # TODO optimizer parametrizovat

   mse = run_k_fold(model, scaler_x, scaler_y, X, y, batch_size=BATCH_SIZE)  # TODO batch size parametrizovat
   return np.mean(mse) # vrat prumer za jednotlive foldy

if __name__=="__main__":

   df = pd.read_csv("../Data/data_for_NN_test.txt")

   X = df[[f"layer{i}" for i in range(1,IN_DIM+1)]]
   y = df[["A1", "A2","T1","T2"]]
   #print(y.describe())
   train_evalate_model(None, X.values, y.values)
   exit(1)



   #
   #
   #
   # plt.plot(history.history['loss'], label='loss- training data')
   # plt.plot(history.history['val_loss'], label='loss - validating data')
   # plt.grid()
   # plt.legend()
   # plt.show()
   #


