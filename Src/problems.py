import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import mean_squared_error
import tensorflow as tf
import os
# import seaborn as sns
from models import get_model

EPOCHS = 1000  # pocet epoch pro uceni

IN_DIM = 21
OUT_DIM = 4
LR = 0.001 # 0.0001
BATCH = 4096
DEEP = 4

NAMES = ["A1", "A2", "T1", "T2"]


def draw_history(history):
    plt.rcParams['figure.figsize'] = [8, 4]
    figure, ax = plt.subplots(1, 2)
    ax[0].plot(history.history["loss"], label=f"Loss trainig data ")
    ax[0].plot(history.history["val_loss"], label=f"Loss validation data ")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(history.history["mse"], label=f"MSE trainig data ")
    ax[1].plot(history.history["val_mse"], label=f"MSE validation data ")
    ax[1].grid()
    ax[1].legend()
    plt.show()


def draw_distribution(y, y_hat, label="Train"):
    plt.rcParams['figure.figsize'] = [8, 8]
    figure, ax = plt.subplots(2, 2)
    name = NAMES[0]
    ax[0, 0].hist(y_hat[:, 0], bins=100, label=f"Computed distribution of {name}")
    ax[0, 0].hist(y[:, 0], bins=100, label=f"Real distribution of {name}", alpha=0.5, color="red")
    ax[0, 0].legend()
    ax[0, 0].grid()

    name = NAMES[1]
    ax[0, 1].hist(y_hat[:, 1], bins=100, label=f"Computed distribution of {name}")
    ax[0, 1].hist(y[:, 1], bins=100, label=f"Real distribution of {name}", alpha=0.5, color="red")
    ax[0, 1].legend()
    ax[0, 1].grid()

    name = NAMES[2]
    ax[1, 0].hist(y_hat[:, 2], bins=100, label=f"Computed distribution of {name}")
    ax[1, 0].hist(y[:, 2], bins=100, label=f"Real distribution of {name}", alpha=0.5, color="red")
    ax[1, 0].legend()
    ax[1, 0].grid()

    name = NAMES[3]
    ax[1, 1].hist(y_hat[:, 3], bins=100, label=f"Computed distribution of {name}")
    ax[1, 1].hist(y[:, 3], bins=100, label=f"Real distribution of {name}", alpha=0.5, color="red")
    ax[1, 1].legend()
    ax[1, 1].grid()

    figure.suptitle(f"Distribution {label}:{NAMES}")
    plt.show()


def draw_corr(y, y_hat, label="Train"):
    import mpl_scatter_density
    plt.rcParams['figure.figsize'] = [8, 8]
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], N=256)
    fig = plt.figure()
    for i in range(OUT_DIM):
        ax = fig.add_subplot(2, 2, i+1, projection='scatter_density')
        name = NAMES[i]
        corr = np.corrcoef(y[:, i], y_hat[:, i])[0, 1]
        density = ax.scatter_density(y[:, i], y_hat[:, i], cmap=white_viridis, dpi=None)
        ax.set_title(name+f" corr: {corr}")
        fig.colorbar(density, label='Density')

    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(f"../Data/data_for_NN_test.txt")
    X = df[[f"layer{i}" for i in range(1, IN_DIM + 1)]]
    y = df[NAMES]

    print(X.describe())
    print(y.describe())

    # X.describe().to_csv("X-summary.csv")
    # y.describe().to_csv("y-summary.csv")

    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.1, random_state=42)

    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    X_train_sc = scaler_x.fit_transform(X_train)
    X_test_sc = scaler_x.transform(X_test)

    # mmin = X_train.min()
    # mmax = X_train.max()
    # X_train_sc = 2 * (X_train - mmin) / (mmax - mmin) - 1
    # X_test_sc = 2 * (X_test - mmin) / (mmax - mmin) - 1

    y_train_sc = scaler_y.fit_transform(y_train)
    y_test_sc = scaler_y.transform(y_test)
    units = [512 for i in range(DEEP)]
    activations = ["relu" for i in range(DEEP)]
    dropouts = [0.1 for i in range(DEEP)]

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LR,
        decay_steps=1000,
        decay_rate=0.9)

    model = get_model(units=units, activations=activations, dropouts=dropouts, in_dim=IN_DIM, out_dim=OUT_DIM)
    model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), metrics=['mse', "mae"])
    history = model.fit(X_train_sc, y_train_sc, epochs=EPOCHS, validation_split=0.1, verbose=2,
                        batch_size=BATCH, shuffle=True)


    y_hat_train_sc = model.predict(X_train_sc)
    y_hat_test_sc = model.predict(X_test_sc)



    print(f"TRAIN>>>{mean_squared_error(y_hat_train_sc, y_train_sc)}")
    for i in range(4):
        print(f"CORR:{np.corrcoef(y_train_sc[:, i],y_hat_train_sc[:, i])[0,1]}")

    print(f"TEST>>>{mean_squared_error(y_hat_test_sc, y_test_sc)}")
    for i in range(4):
        print(f"CORR:{np.corrcoef(y_test_sc[:, i], y_hat_test_sc[:, i])[0, 1]}")

    to_predict =np.array([41.79857899,47.80665522,44.95921279,43.75694528,40.0470282,35.33130118,1260.492509,594.2514374,441.3921548,418.7659755,402.7798095,358.1815222,336.9068776,321.1603479,287.4529931,266.1460777,240.1359104,222.0845567,197.2556906,156.0158626,141.8509294])
    mmin = to_predict.min()
    mmax = to_predict.max()
    to_predict_sc = 2*(to_predict-mmin)/(mmax-mmin)-1
    print(to_predict_sc)
    to_predict_sc = to_predict_sc.reshape(1,21)

    #to_predict_sc = scaler_x.transform(to_predict)
    y_hat_signal_sc = model.predict(to_predict_sc)
    y_hat_signal = scaler_y.inverse_transform(y_hat_signal_sc)
    print(y_hat_signal)

    draw_history(history)
    draw_distribution(y_train_sc, y_hat_train_sc)
    draw_corr(y_train_sc, y_hat_train_sc)
