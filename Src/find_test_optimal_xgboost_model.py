import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from drawing_results import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
IN_DIM = 21
OUT_DIM = 4
DEVICE="gpu"

NAMES = ["A1", "A2", "T1", "T2"]



if __name__=="__main__":
    df = pd.read_csv(f"../Data/data_for_NN_test2.txt")
    X = df[[f"layer{i}" for i in range(1, IN_DIM + 1)]]
    y = df[NAMES]
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.1, random_state=42)
    parameters = {
        "max_depth": [i for i in range(5, 15)],
        "n_estimators": [i for i in range(100, 1000, 100)],
        "eta" : [0.005, 0.01,0.05,0.1,0.2,0.3,0.4],
        "subsample" : [0.3,0.4,0.5,0.6,0.7,0.8,0.9],
        "colsample_bytree":[0.6,0.7, 0.8,0.9],
        "random_state": [42]

    }  # nastav si grid parametrů

    model = XGBRegressor(device=DEVICE)
    gs = GridSearchCV(model, parameters, verbose=2, scoring="neg_root_mean_squared_error")  # vloz model do hledace
    gs.fit(X_train, y_train)  # spust hledani
    print(f"Nejlepší parametry:{gs.best_params_}")  # vytiskni nejlepsi parametry
    optimal_model=gs.best_estimator_  # vyber nejlepsi model
    optimal_model.save_model("../Models/xgboost_optimal_nejsem jistjist_zda_je nejlepsi.model")
    # zapamatuj si puvodni data
    y_train_original = y_train.copy()
    y_test_original = y_test.copy()
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X_train_sc = scaler_x.fit_transform(X_train)
    X_test_sc = scaler_x.transform(X_test)

    y_train_sc = scaler_y.fit_transform(y_train)
    y_test_sc = scaler_y.transform(y_test)

    # udelej predikce
    optimal_model.fit(X_train_sc, y_train_sc)
    y_hat_train_sc = optimal_model.predict(X_train_sc)
    y_hat_test_sc = optimal_model.predict(X_test_sc)

    # nejprve to preskaluj zpet na puvodni meritko
    y_hat_train_unsc = scaler_y.inverse_transform(y_hat_train_sc)
    y_hat_test_unsc = scaler_y.inverse_transform(y_hat_test_sc)

    print(f"TRAIN MSE SCALED:{mean_squared_error(y_hat_train_sc, y_train_sc)}")
    for i in range(4):
        print(f"CORR:{np.corrcoef(y_train_original[:, i], y_hat_train_unsc[:, i])[0, 1]}")

    print(f"TEST MSE SCALED:{mean_squared_error(y_hat_test_sc, y_test_sc)}")
    for i in range(4):
        print(f"CORR:{np.corrcoef(y_test_original[:, i], y_hat_test_unsc[:, i])[0, 1]}")

    draw_distribution(y_test_original, y_hat_test_unsc)
    draw_corr(y_test_original, y_hat_test_unsc)


