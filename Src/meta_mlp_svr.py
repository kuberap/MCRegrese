"""
Co se stane, když použijeme MLP a SVR a pak uděláme regresi na jejich výsledky?
"""

import pandas as pd
import numpy as np  # pro práci s maticemi
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score as r2


if __name__ == "__main__":
    mlp_data = pd.read_csv("../Res/analyza-neuspechu/DEEP-TRAIN-2-UNITS-512-BATCH-4096-EPOCHS-1000.csv")
    svr_data = pd.read_csv("../Res/analyza-neuspechu/SVR-TRAIN.csv")

    y = mlp_data[['A1_sc', 'A2_sc', 'T1_sc', 'T2_sc']].values  ## toto chci urcit

    X_svr = svr_data[['A1_hat_sc', 'A2_hat_sc', 'T1_hat_sc', 'T2_hat_sc']].values  ## toto mam
    X_mlp = mlp_data[['A1_hat_sc', 'A2_hat_sc', 'T1_hat_sc', 'T2_hat_sc']].values  ## toto mam
    X = np.hstack((X_mlp, X_svr))  ## matice systému
    c = 0.00
    A = X.T @ X + c * np.eye(X.shape[1])
    print(A.shape)
    y_hat = np.zeros(y.shape)
    alphas = []
    for i in range(4):
        alpha = np.linalg.solve(A, X.T @ y[:, i])
        y_hat[:, i] = X @ alpha  # predikce
        alphas.append(alpha)
    print(f"TRAIN MSE SCALED:{mean_squared_error(y_hat, y)}")
    for i in range(4):
        print(f"CORR:{np.corrcoef(y[:, i], y_hat[:, i])[0, 1]}")

    mlp_data_test = pd.read_csv("../Res/analyza-neuspechu/DEEP-TEST-2-UNITS-512-BATCH-4096-EPOCHS-1000.csv")
    svr_data_test = pd.read_csv("../Res/analyza-neuspechu/SVR-TEST.csv")
    y_test = mlp_data_test[['A1_sc', 'A2_sc', 'T1_sc', 'T2_sc']].values  ## toto chci urcit
    X_svr_test = svr_data_test[['A1_hat_sc', 'A2_hat_sc', 'T1_hat_sc', 'T2_hat_sc']].values  ## toto mam
    X_mlp_test = mlp_data_test[['A1_hat_sc', 'A2_hat_sc', 'T1_hat_sc', 'T2_hat_sc']].values  ## toto mam
    X_test = np.hstack((X_mlp_test, X_svr_test))  ## matice systému
    y_hat_test = np.zeros(y_test.shape)
    for i in range(4):
        y_hat_test[:, i] = X_test @ alphas[i]  # predikce
    print(f"TEST MSE SCALED:{mean_squared_error(y_hat_test, y_test)}")
    for i in range(4):
        print(f"CORR:{np.corrcoef(y_test[:, i], y_hat_test[:, i])[0, 1]}")
    print(f"TRAIN R2")
    for i in range(4):
        print(f"component r2:{r2(y[:, i], y_hat[:, i])}")

    print(f"TEST R2")
    for i in range(4):
        print(f"component r2:{r2(y[:, i], y_hat[:, i])}")