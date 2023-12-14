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
LR = 0.001  # 0.001
BATCH = 4096

# parametry modelu
DEEP = 2
UNITS = 512
DROPOUT = 0.1

NAMES = ["A1", "A2", "T1", "T2"]
X_LOG_TRANSFORM = False
Y_LOG_TRANSFORM = True

# zda budu trenovat, nebo jen pouzivat naucenou sit
TRAIN=False
MODEL_NAME=f"DEEP-{DEEP}-UNITS-{UNITS}-BATCH-{BATCH}-EPOCHS-{EPOCHS}.keras"
TRAINING_RESULTS_NAME=f"DEEP-TRAIN-{DEEP}-UNITS-{UNITS}-BATCH-{BATCH}-EPOCHS-{EPOCHS}.csv"
TESTING_RESULTS_NAME=f"DEEP-TEST-{DEEP}-UNITS-{UNITS}-BATCH-{BATCH}-EPOCHS-{EPOCHS}.csv"

@tf.keras.saving.register_keras_serializable()
def weighted_mean_absolute_error(weights):
    def loss(y_true, y_pred):
        return tf.math.reduce_mean(
            tf.math.multiply(tf.math.abs(tf.cast(y_true, tf.float32) - y_pred), weights)
        )
    return loss
@tf.keras.saving.register_keras_serializable()
def weighted_custom_loss(weights):
    def loss(y_true, y_pred):
        y_t = tf.cast(y_true, tf.float32)
        return tf.math.reduce_mean(
            tf.math.multiply(tf.math.abs(y_t - y_pred), weights)
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

    X.describe().to_csv("X-summary.csv")
    y.describe().to_csv("y-summary.csv")

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
        decay_steps=100000,
        decay_rate=0.9) # spolu s ADAM
    #lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay([500,800], [LR, 0.5*LR,0.1*LR])
    model = get_mlp_model_with_dropout_1(units=units, activations=activations, dropouts=dropouts, in_dim=IN_DIM,
                                         out_dim=OUT_DIM)

    #weighted_mae = weighted_mean_absolute_error(np.array([1.0, 5.0, 1.0, 1.0]))
    weighted_loss = weighted_custom_loss(np.array([1.0, 10.0, 1.0, 2.0]))
    model.compile(loss=weighted_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  metrics=['mse', "mae"])
    if TRAIN:
        history = model.fit(X_train_sc, y_train_sc, epochs=EPOCHS, validation_split=0.1, verbose=2,
                        batch_size=BATCH, shuffle=True)
        model.save(f"../Models/{MODEL_NAME}")
        loss = history.history["loss"]
        # vypocet odhadu smernice loss krivky z ucicich dat - zda ma smysl dal pokracovat
        loss_slope = None
        if EPOCHS > 10:
            last_n = int(len(loss) * 0.1)
            p = np.polyfit([i for i in range(last_n)], loss[-last_n:],
                       deg=1)  # proloz koncove body primkou a vem linearni koeficient
            loss_slope = p[0]
        draw_history(history)
    else:
        model = tf.keras.models.load_model(f"../Models/{MODEL_NAME}")

    # udelej predikce
    y_hat_train_sc = model.predict(X_train_sc)
    y_hat_test_sc = model.predict(X_test_sc)

    # nejprve to preskaluj zpet na puvodni meritko
    y_hat_train_unsc = scaler_y.inverse_transform(y_hat_train_sc)
    y_hat_test_unsc = scaler_y.inverse_transform(y_hat_test_sc)

    to_predict = np.array(
        [41.79857899, 47.80665522, 44.95921279, 43.75694528, 40.0470282, 35.33130118, 1260.492509, 594.2514374,
         441.3921548, 418.7659755, 402.7798095, 358.1815222, 336.9068776, 321.1603479, 287.4529931, 266.1460777,
         240.1359104, 222.0845567, 197.2556906, 156.0158626, 141.8509294]) # posledni hodnota vyjmuta 141.8509294


    to_predict = to_predict.reshape(1, 21)

    to_predict_sc = scaler_x.transform(to_predict)
    print(to_predict_sc)
    y_hat_signal_sc = model.predict(to_predict_sc)
    y_hat_signal = scaler_y.inverse_transform(y_hat_signal_sc)


    if Y_LOG_TRANSFORM:  # kdyz je logaritmicka transformace, tak pred vypoctem korelace vratim zpet
        y_hat_train_unsc = np.exp(y_hat_train_unsc)
        y_hat_test_unsc = np.exp(y_hat_test_unsc)
        y_hat_signal_unsc= np.exp(y_hat_signal)

    if TRAIN:
        print(f"LOSS SLOPE:{loss_slope}")

    print(f"signal:{y_hat_signal_unsc}")
    exit(-1)


    #export vysledku training
    result_training={}
    titles=[f"layer{i}" for i in range(1, IN_DIM + 1)]
    titles.extend(NAMES)
    output_names_sc=[f"{name}_sc" for name in NAMES]
    titles.extend(output_names_sc)
    output_names_hat=[f"{name}_hat" for name in NAMES]
    titles.extend(output_names_hat)
    output_names_hat_sc=[f"{name}_hat_sc" for name in NAMES]
    titles.extend(output_names_hat_sc)

    for t in titles:
        result_training[t]=[]
    for layer in range(1, IN_DIM + 1): # pro kazdy sloupec vstupu X
        column_name=f"layer{layer}"
        for row in X_train:
            result_training[column_name].append(row[layer-1])
    for col, name in enumerate(NAMES): # pro kazdy sloupec vystupu y_original
        for row in y_train_original:
            result_training[name].append(row[col])
    for col, name in enumerate(output_names_sc): # pro kazdy sloupec vystupu y_sc
        for row in y_train_sc:
            result_training[name].append(row[col])
    for col, name in enumerate(output_names_hat): # pro kazdy sloupec vystupu y_hat
        for row in y_hat_train_unsc:
            result_training[name].append(row[col])
    for col, name in enumerate(output_names_hat_sc): # pro kazdy sloupec vystupu y_hat
        for row in y_hat_train_sc:
            result_training[name].append(row[col])
    df_training_results = pd.DataFrame.from_dict(data=result_training)
    df_training_results.to_csv(TRAINING_RESULTS_NAME)
    # export vysledku testing
    result_testing = {}
    titles = [f"layer{i}" for i in range(1, IN_DIM + 1)]
    titles.extend(NAMES)
    output_names_sc = [f"{name}_sc" for name in NAMES]
    titles.extend(output_names_sc)
    output_names_hat = [f"{name}_hat" for name in NAMES]
    titles.extend(output_names_hat)
    output_names_hat_sc = [f"{name}_hat_sc" for name in NAMES]
    titles.extend(output_names_hat_sc)

    for t in titles:
        result_testing[t] = []
    for layer in range(1, IN_DIM + 1):  # pro kazdy sloupec vstupu X
        column_name = f"layer{layer}"
        for row in X_test:
            result_testing[column_name].append(row[layer - 1])
    for col, name in enumerate(NAMES):  # pro kazdy sloupec vystupu y_original
        for row in y_test_original:
            result_testing[name].append(row[col])
    for col, name in enumerate(output_names_sc):  # pro kazdy sloupec vystupu y_sc
        for row in y_test_sc:
            result_testing[name].append(row[col])
    for col, name in enumerate(output_names_hat):  # pro kazdy sloupec vystupu y_hat
        for row in y_hat_test_unsc:
            result_testing[name].append(row[col])
    for col, name in enumerate(output_names_hat_sc):  # pro kazdy sloupec vystupu y_hat
        for row in y_hat_test_sc:
            result_testing[name].append(row[col])
    df_training_results = pd.DataFrame.from_dict(data=result_testing)
    df_training_results.to_csv(TESTING_RESULTS_NAME)

    #------------------------
    print(f"TRAIN MSE SCALED:{mean_squared_error(y_hat_train_sc, y_train_sc)}")
    for i in range(4):
        print(f"CORR:{np.corrcoef(y_train_original[:, i], y_hat_train_unsc[:, i])[0, 1]}")

    print(f"TEST MSE SCALED:{mean_squared_error(y_hat_test_sc, y_test_sc)}")
    for i in range(4):
        print(f"CORR:{np.corrcoef(y_test_original[:, i], y_hat_test_unsc[:, i])[0, 1]}")


    draw_distribution(y_train_original, y_hat_train_unsc)
    draw_corr(y_train_original, y_hat_train_unsc)

    draw_distribution(y_test_original, y_hat_test_unsc, label="TEST")
    draw_corr(y_test_original, y_hat_test_unsc, label="TEST")
