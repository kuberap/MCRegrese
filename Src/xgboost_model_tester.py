import xgboost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from drawing_results import *
from sklearn.metrics import mean_squared_error

IN_DIM = 21
OUT_DIM = 4
LOAD_MODEL = False #True
SAVE_MODEL = False
NAMES = ["A1", "A2", "T1", "T2"]

TRAINING_RESULTS_NAME=f"XGBOOST-TRAIN.csv"
TESTING_RESULTS_NAME=f"XGBOOST-TEST.csv"



if __name__=="__main__":
    df = pd.read_csv(f"../Data/data_for_NN_test2.txt")
    X = df[[f"layer{i}" for i in range(1, IN_DIM + 1)]]
    y = df[NAMES]
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.1, random_state=42)

    # zapamatuj si puvodni data
    y_train_original = y_train.copy()
    y_test_original = y_test.copy()
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X_train_sc = scaler_x.fit_transform(X_train)
    X_test_sc = scaler_x.transform(X_test)

    y_train_sc = scaler_y.fit_transform(y_train)
    y_test_sc = scaler_y.transform(y_test)
    if LOAD_MODEL:
        model = xgboost.XGBRegressor()
        model.load_model("../Models/xgboost_optimal.model")
    else:
        # parametry:{'colsample_bytree': 0.9, 'eta': 0.05, 'max_depth': 9, 'n_estimators': 900, 'random_state': 42, 'subsample': 0.7}
        model = XGBRegressor(colsample_bytree= 0.9, eta=0.05, max_depth= 9, n_estimators=900, random_state=42, subsample=0.7)

    model.fit(X_train_sc, y_train_sc)
    if SAVE_MODEL:
        model.save_model("../Models/xgboost_optimal.model")
    # udelej predikce
    y_hat_train_sc = model.predict(X_train_sc)
    y_hat_test_sc = model.predict(X_test_sc)

    # nejprve to preskaluj zpet na puvodni meritko
    y_hat_train_unsc = scaler_y.inverse_transform(y_hat_train_sc)
    y_hat_test_unsc = scaler_y.inverse_transform(y_hat_test_sc)

    to_predict = np.array(
        [41.79857899, 47.80665522, 44.95921279, 43.75694528, 40.0470282, 35.33130118, 1260.492509, 594.2514374,
         441.3921548, 418.7659755, 402.7798095, 358.1815222, 336.9068776, 321.1603479, 287.4529931, 266.1460777,
         240.1359104, 222.0845567, 197.2556906, 156.0158626, 141.8509294])  # posledni hodnota vyjmuta 141.8509294

    to_predict = to_predict.reshape(1, 21)

    to_predict_sc = scaler_x.transform(to_predict)
    print(to_predict_sc)
    y_hat_signal_sc = model.predict(to_predict_sc)
    y_hat_signal = scaler_y.inverse_transform(y_hat_signal_sc)

    print(f"signal:{y_hat_signal}")

    # export vysledku training
    result_training = {}
    titles = [f"layer{i}" for i in range(1, IN_DIM + 1)]
    titles.extend(NAMES)
    output_names_sc = [f"{name}_sc" for name in NAMES]
    titles.extend(output_names_sc)
    output_names_hat = [f"{name}_hat" for name in NAMES]
    titles.extend(output_names_hat)
    output_names_hat_sc = [f"{name}_hat_sc" for name in NAMES]
    titles.extend(output_names_hat_sc)

    for t in titles:
        result_training[t] = []
    for layer in range(1, IN_DIM + 1):  # pro kazdy sloupec vstupu X
        column_name = f"layer{layer}"
        for row in X_train:
            result_training[column_name].append(row[layer - 1])
    for col, name in enumerate(NAMES):  # pro kazdy sloupec vystupu y_original
        for row in y_train_original:
            result_training[name].append(row[col])
    for col, name in enumerate(output_names_sc):  # pro kazdy sloupec vystupu y_sc
        for row in y_train_sc:
            result_training[name].append(row[col])
    for col, name in enumerate(output_names_hat):  # pro kazdy sloupec vystupu y_hat
        for row in y_hat_train_unsc:
            result_training[name].append(row[col])
    for col, name in enumerate(output_names_hat_sc):  # pro kazdy sloupec vystupu y_hat
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

    print(f"TRAIN MSE SCALED:{mean_squared_error(y_hat_train_sc, y_train_sc)}")
    for i in range(4):
        print(f"CORR:{np.corrcoef(y_train_original[:, i], y_hat_train_unsc[:, i])[0, 1]}")

    print(f"TEST MSE SCALED:{mean_squared_error(y_hat_test_sc, y_test_sc)}")
    for i in range(4):
        print(f"CORR:{np.corrcoef(y_test_original[:, i], y_hat_test_unsc[:, i])[0, 1]}")

    draw_distribution(y_train_original, y_hat_train_unsc)
    draw_corr(y_train_original, y_hat_train_unsc)
