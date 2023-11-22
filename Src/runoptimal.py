import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import os
from models import get_model

EPOCHS = 20  # pocet epoch pro uceni
SPLITS = 5
IN_DIM = 21
OUT_DIM = 1  # 4


def train_evaluate_model(config, data_dir=None):
    # print(f"::::::::::::::::::{data_dir}")
    df = pd.read_csv(f"{data_dir}/Data/data_for_NN_test.txt")
    X = df[[f"layer{i}" for i in range(1, IN_DIM + 1)]]
    y = df[["A1"]]  # df[["A1", "A2", "T1", "T2"]]
    print(X.describe())
    print(y.describe())
    X = X.values[:10000]
    y = y.values[:10000]
    kf = KFold(n_splits=SPLITS, shuffle=True, random_state=42)
    mse = []
    for fold_index, (train_index, test_index) in enumerate(kf.split(X, y)):
        model = get_model(in_dim=IN_DIM, out_dim=OUT_DIM)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
                      metrics=['mse', "mae"])
        print(model.summary())
        X_train, X_test, y_train, y_test = X[train_index, :].copy(), X[test_index, :].copy(), y[train_index].copy(), y[
            test_index].copy()
        scaler_x = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        X_train_sc = scaler_x.fit_transform(X_train)
        X_test_sc = scaler_x.transform(X_test)
        y_train_sc = scaler_y.fit_transform(y_train)
        # y_test_sc = scaler_y.transform(y_test)


        history = model.fit(X_train_sc, y_train_sc, epochs=EPOCHS, validation_split=0.1, verbose=2,
                             batch_size=config["batch"], shuffle=True)

        # udelej predikci na testovacich datech
        # data jsou preskalovana na puvodni rozmer, abych odstranil vliv intervalu skalovani

        y_hat_train_sc = model.predict(X_train_sc)
        print(f"TRAIN>>>{mean_squared_error(y_hat_train_sc, y_train_sc)}")
        plt.scatter(x=y_train_sc[:, 0], y=y_hat_train_sc[:,0])
        plt.show()

        y_hat_test_sc = model.predict(X_test_sc)
        y_hat_test = scaler_y.inverse_transform(y_hat_test_sc)
        mse.append(mean_squared_error(y_test, y_hat_test))
        plt.plot(history.history["loss"], label=f"Trainig data fold {fold_index}")
        plt.plot(history.history["val_loss"], label=f"Testing data fold {fold_index}")
        plt.grid()
        plt.legend()
        plt.show()

        print(mse)
        exit(1)

    print(np.array(mse).mean())


if __name__ == "__main__":
    data_dir = os.path.dirname(os.getcwd())
    config = {
        "lr": 0.001,
        "batch": 2048,
    }

    train_evaluate_model(config, data_dir)
