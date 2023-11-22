import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


IN_DIM = 21
OUT_DIM = 4
LR = 0.0001 # 0.0001
BATCH = 4096
DEEP = 4

NAMES = ["A1", "A2", "T1", "T2"]

if __name__ == "__main__":
    df = pd.read_csv(f"../Data/data_for_NN_test2.txt")
    X = df[[f"layer{i}" for i in range(1, IN_DIM + 1)]]


    y = df[NAMES]

    print(X.describe())
    print(y.describe())


    for l in [10,10,1,0.1,0.01,0.001]:
        X_tr = (X ** l - 1) / l
        plt.hist(X_tr.iloc[:, 0], bins=100, label=f"Real distribution layer {l}", alpha=0.5, color="red")
        plt.legend()
        plt.show()
