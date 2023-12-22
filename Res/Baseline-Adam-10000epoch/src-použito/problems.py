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

EPOCHS = 10000  # pocet epoch pro uceni

IN_DIM = 21
OUT_DIM = 4
LR = 0.0001
BATCH = 4096
DEEP = 6

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
        density = ax.scatter_density(y[:, i], y_hat[:, i], cmap=white_viridis)
        ax.set_title(name+f" corr: {corr}")
        fig.colorbar(density, label='Density')

    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(f"../Data/data_for_NN_test.txt")
    X = df[[f"layer{i}" for i in range(1, IN_DIM + 1)]]
    y = df[NAMES]

    print(X.describe())
    print(y.describe())

    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.1, random_state=42)

    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    X_train_sc = scaler_x.fit_transform(X_train)
    X_test_sc = scaler_x.transform(X_test)
    y_train_sc = scaler_y.fit_transform(y_train)
    units = [512 for i in range(DEEP)]
    activations = ["relu" for i in range(DEEP)]
    dropouts = [0.1 for i in range(DEEP)]

    model = get_model(units=units, activations=activations, dropouts=dropouts, in_dim=IN_DIM, out_dim=OUT_DIM)
    model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(learning_rate=LR), metrics=['mse', "mae"])
    history = model.fit(X_train_sc, y_train_sc, epochs=EPOCHS, validation_split=0.1, verbose=2,
                        batch_size=BATCH, shuffle=True)


    y_hat_train_sc = model.predict(X_train_sc)
    print(f"TRAIN>>>{mean_squared_error(y_hat_train_sc, y_train_sc)}")
    for i in range(4):
        print(f"CORR:{np.corrcoef(y_train_sc[:, i],y_hat_train_sc[:, i])[0,1]}")

    draw_history(history)
    draw_distribution(y_train_sc, y_hat_train_sc)
    draw_corr(y_train_sc, y_hat_train_sc)
