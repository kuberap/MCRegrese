import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
IMAGE_PATH = "../Res/analyza-neuspechu/images"


if __name__=="__main__":
    df = pd.read_csv("../Res/analyza-neuspechu/ERRORS-DEEP-TRAIN-2-UNITS-512-BATCH-4096-EPOCHS-1000-ML.csv")
    inputs = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5',
       'layer6', 'layer7', 'layer8', 'layer9', 'layer10', 'layer11', 'layer12',
       'layer13', 'layer14', 'layer15', 'layer16', 'layer17', 'layer18',
       'layer19', 'layer20', 'layer21']
    X = df[inputs].values
    for e in ['err_A1_sc', 'err_A2_sc', 'err_T1_sc', 'err_T2_sc']:
       print(f"---------------{e}--------------------")
       y = df[e].values
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
       model = RandomForestClassifier(class_weight="balanced_subsample", max_features=10) #use random forest
       model.fit(X_train, y_train)
       y_hat_test = model.predict(X_test)
       print(model.score(X_train, y_train))
       print(model.score(X_test, y_test))
       print(classification_report(y_test, y_hat_test, target_names=["OK", "BAD"]))
       feature_importances = model.feature_importances_
       plt.barh(inputs, feature_importances)
       plt.title(f"Feature importances in {e}")
       plt.grid()
       plt.savefig(f"{IMAGE_PATH}/feature_importance-{e}.png")
       plt.clf()
