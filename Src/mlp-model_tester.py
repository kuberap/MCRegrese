import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import mean_squared_error
import tensorflow as tf
import tensorflow.python.keras.backend as K
from models import get_mlp_model_with_dropout_1
from drawing_results import *

EPOCHS = 1000  # 00  # pocet epoch pro uceni

IN_DIM = 21
OUT_DIM = 4
LR = 0.001  # 0.0001
BATCH = 4096


# parametry modelu
DEEP = 2
UNITS = 2048
DROPOUT = 0.1

NAMES = ["A1", "A2", "T1", "T2"]
X_LOG_TRANSFORM = False
Y_LOG_TRANSFORM = True

def weighted_mean_absolute_error(weights):
    def loss(y_true, y_pred):

        return tf.math.reduce_mean(
            tf.math.multiply(tf.math.abs(tf.cast(y_true, tf.float32) - y_pred), weights)
        )
    return loss



if __name__ == "__main__":
    df = pd.read_csv(f"../Data/data_for_NN_test2.txt")
    X = df[[f"layer{i}" for i in range(1, IN_DIM + 1)]]
    y = df[NAMES]

    if X_LOG_TRANSFORM:  # zda logaritmicky skalovat vstupni data
        X = np.log(X)


    print(X.describe())
    print(y.describe())

    # X.describe().to_csv("X-summary.csv")
    # y.describe().to_csv("y-summary.csv")

    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.1, random_state=42)

    # zapamatuj si puvodni data
    y_train_original = y_train.copy()
    y_test_original = y_test.copy()
    if Y_LOG_TRANSFORM:  # zda log. skalovat vystupni data

        y_train = np.log(y_train)
        y_test = np.log(y_test)

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X_train_sc = scaler_x.fit_transform(X_train)
    X_test_sc = scaler_x.transform(X_test)

    y_train_sc = scaler_y.fit_transform(y_train)
    y_test_sc = scaler_y.transform(y_test)

    units = [UNITS for i in range(DEEP)]
    activations = ["relu" for i in range(DEEP)]
    dropouts = [DROPOUT for i in range(DEEP)]

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LR,
        decay_steps=1000,
        decay_rate=0.9)

    model = get_mlp_model_with_dropout_1(units=units, activations=activations, dropouts=dropouts, in_dim=IN_DIM,
                                         out_dim=OUT_DIM)

    weighted_mae=weighted_mean_absolute_error(np.array([1.0,5.0,1.0,1.0]))
    model.compile(loss=weighted_mae, optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), metrics=['mse', "mae"])
    history = model.fit(X_train_sc, y_train_sc, epochs=EPOCHS, validation_split=0.1, verbose=2,
                        batch_size=BATCH, shuffle=True)

    loss = history.history["loss"]
    # vzpocet odhadu smernice loss krivky z ucicich dat - zda ma smysl dal pokracovat
    last_n=int(len(loss)*0.1)
    p=np.polyfit([i for i in range(last_n)], loss[-last_n:], deg=1) # proloz koncove body primkou a vem linearni koeficient

    draw_history(history)

    # udelej predikce
    y_hat_train_sc = model.predict(X_train_sc)
    y_hat_test_sc = model.predict(X_test_sc)

    # nejprve to preskaluj zpet na puvodni meritko
    y_hat_train_unsc = scaler_y.inverse_transform(y_hat_train_sc)
    y_hat_test_unsc = scaler_y.inverse_transform(y_hat_test_sc)


    if Y_LOG_TRANSFORM: # kdyz je logaritmicka transformace, tak pred vypoctem korelace vratim zpet
        y_hat_train_unsc = np.exp(y_hat_train_unsc)
        y_hat_test_unsc = np.exp(y_hat_test_unsc)

    print(f"LOSS SLOPE:{p[0]}")
    print(f"TRAIN MSE SCALED:{mean_squared_error(y_hat_train_sc, y_train_sc)}")
    for i in range(4):
        print(f"CORR:{np.corrcoef(y_train_original[:, i], y_hat_train_unsc[:, i])[0, 1]}")

    print(f"TEST MSE SCALED:{mean_squared_error(y_hat_test_sc, y_test_sc)}")
    for i in range(4):
        print(f"CORR:{np.corrcoef(y_test_original[:, i], y_hat_test_unsc[:, i])[0, 1]}")

    draw_distribution(y_train_original, y_hat_train_unsc)
    draw_corr(y_train_original, y_hat_train_unsc)
