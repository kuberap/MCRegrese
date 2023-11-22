import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from functools import partial
import ray
from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.air.integrations.keras import ReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from ray.air import Checkpoint, session
import json

from models import get_model

# =========================================================================================
import os

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# =========================================================================================


EPOCHS = 20
EPOCHS_PATIENCE_ES = EPOCHS // 5  # kdyz se to po danem poctu epoch nezlepsi, tak to stopni
EPOCHS_PATIENCE_RP = EPOCHS // 10  # kdyz se to na validacnich datech nezlepsuje po danem poctu epoch, tak zmenis learning rate
BATCH_SIZE = 1024  # bude prepsano pri optimalizaci hyperparametru
SPLITS = 5
IN_DIM = 21
OUT_DIM = 4
OPT_MAX_SAMPLES = 200

assert EPOCHS_PATIENCE_RP < EPOCHS_PATIENCE_ES, f" Patience for Early stops {EPOCHS_PATIENCE_ES} must be less than paience for ReduceonPlato {EPOCHS_PATIENCE_RP}"


def run_k_fold(model, scaler_x, scaler_y, X, y, batch_size, val_ratio=0.1):
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
    kf = KFold(n_splits=SPLITS, shuffle=True, random_state=42)
    mse = []
    for fold_index, (train_index, test_index) in enumerate(kf.split(X, y)):
        # print(f'FOLD: {fold_index}')
        # Vybere data pro jednotlive foldy a naskaluje je
        X_train, X_test, y_train, y_test = X[train_index, :], X[test_index, :], y[train_index], y[test_index]
        X_train_sc = scaler_x.fit_transform(X_train)
        X_test_sc = scaler_x.transform(X_test)
        y_train_sc = scaler_y.fit_transform(y_train)
        # y_test_sc = scaler_y.transform(y_test)

        early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                         patience=EPOCHS_PATIENCE_ES)  # kdyz se to dany pocet epoch nezlepsi stopni
        reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=EPOCHS_PATIENCE_RP,
                                                            min_lr=0.001)
        model.fit(X_train_sc, y_train_sc, epochs=EPOCHS, validation_split=val_ratio, verbose=0,
                  callbacks=[early_stop_cb, reduce_lr_cb], batch_size=batch_size)  # validation_data=val_dataset,
        # udelej predikci na testovacich datech
        # data jsou preskalovana na puvodni rozmer, abych odstranil vliv intervalu skalovani
        y_hat_sc = model.predict(X_test_sc, verbose=False)
        y_hat = scaler_y.inverse_transform(y_hat_sc)
        mse.append(mean_squared_error(y_test, y_hat))
        train.report({"loss": np.mean(np.array(mse))})
    # print(mse)
    return np.array(mse)


def train_evaluate_model(config, data_dir=None):
    # print(f"::::::::::::::::::{data_dir}")
    df = pd.read_csv(f"{data_dir}/Data/data_for_NN_test.txt")
    X = df[[f"layer{i}" for i in range(1, IN_DIM + 1)]]
    y = df[["A1", "A2", "T1", "T2"]]
    X = X.values
    y = y.values

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    units = [config["hidden1"], config["hidden2"], config["hidden3"]]
    dropouts = [config["drop1"], config["drop2"], config["drop3"]]
    activations = ["relu", "relu", "relu"]  # [config["act1"],config["act2"],config["act3"]]
    layers = config["layers"]
    kf = KFold(n_splits=SPLITS, shuffle=True, random_state=42)
    mse = []
    for fold_index, (train_index, test_index) in enumerate(kf.split(X, y)):
        model = get_model(units=units[0:layers], dropouts=dropouts[0:layers], activations=activations[0:layers],
                      in_dim=IN_DIM, out_dim=OUT_DIM)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]), metrics=['mse', "mae"])
        X_train, X_test, y_train, y_test = X[train_index, :], X[test_index, :], y[train_index], y[test_index]
        X_train_sc = scaler_x.fit_transform(X_train)
        X_test_sc = scaler_x.transform(X_test)
        y_train_sc = scaler_y.fit_transform(y_train)
        # y_test_sc = scaler_y.transform(y_test)

        early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                         patience=EPOCHS_PATIENCE_ES)  # kdyz se to dany pocet epoch nezlepsi stopni
        reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=EPOCHS_PATIENCE_RP,
                                                            min_lr=0.001)
        model.fit(X_train_sc, y_train_sc, epochs=EPOCHS, validation_split=0.1, verbose=0,
                  callbacks=[early_stop_cb, reduce_lr_cb], batch_size=config["batch"])  # validation_data=val_dataset,
        # udelej predikci na testovacich datech
        # data jsou preskalovana na puvodni rozmer, abych odstranil vliv intervalu skalovani
        y_hat_sc = model.predict(X_test_sc, verbose=False)
        y_hat = scaler_y.inverse_transform(y_hat_sc)
        mse.append(mean_squared_error(y_test, y_hat))
        train.report({"loss": np.mean(np.array(mse))})

        train.report({"loss": np.mean(mse)})

    # print(f">>>>{np.mean(mse)}")  # vrat prumer za jednotlive foldy


def tune_mcregression():
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=100,
        grace_period=1,
        reduction_factor=2,
    )
    config = {
        "threads": 2,
        "lr": tune.loguniform(1e-6, 1e-2),
        "hidden1": tune.choice([2 ** i for i in range(4, 11)]),
        "hidden2": tune.choice([2 ** i for i in range(4, 11)]),
        "hidden3": tune.choice([2 ** i for i in range(4, 11)]),
        "drop1": tune.uniform(0.0, 0.5),
        "drop2": tune.uniform(0.0, 0.5),
        "drop3": tune.uniform(0.0, 0.5),
        "layers": tune.choice([1, 2, 3]),
        # "act1": tune.choice(["relu", "elu", "tanh"]),
        # "act2": tune.choice(["relu", "elu", "tanh"]),
        # "act3": tune.choice(["relu", "elu", "tanh"]),
        "batch": tune.choice([32, 64, 128, 256, 512, 1024]),
    }

    result = tune.run(
        partial(train_evaluate_model, data_dir=os.path.dirname(os.getcwd())),
        resources_per_trial={"cpu": 8, "gpu": 0},
        config=config,
        num_samples=OPT_MAX_SAMPLES,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    with open('bestmodel.txt', 'w') as convert_file:
        convert_file.write(json.dumps(best_trial.config))


if __name__ == "__main__":
    print(tf.__version__)

    tune_mcregression()

    # print(y.describe())
    # train_evaluate_model(None, X.values, y.values)
    # exit(1)

    #
    #
    #
    # plt.plot(history.history['loss'], label='loss- training data')
    # plt.plot(history.history['val_loss'], label='loss - validating data')
    # plt.grid()
    # plt.legend()
    # plt.show()
    #
