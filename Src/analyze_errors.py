import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score as r2


IMAGE_PATH = "../Res/analyza-neuspechu/images"
def draw_mse_distribution(df_err_sc):
    mse = np.sum(df_err_sc.values ** 2, axis=0)
    print(f"MSE distribution 'A1', 'A2', 'T1', 'T2':{mse}")
    plt.pie(mse, labels=['A1', 'A2', 'T1', 'T2'], autopct='%1.1f%%')
    plt.title("MSE-distribution")
    # plt.savefig(f"{IMAGE_PATH}/mse_distribution.png")
    # plt.clf()
    plt.show()


def draw_component_error_distribution(df_err_sc):
    for c in df_err_sc.columns:
        df_err_sc[c].hist(bins=100)
        plt.title(c)
        # plt.savefig(f"{IMAGE_PATH}/{c}-err-distribution.png")
        # plt.clf()
        plt.show()


def draw_rmse_distribution(df_err_sc):
    rmse = np.sqrt(np.sum(df_err_sc.values ** 2, axis=1) / 4)
    rmse_percentil = np.percentile(rmse, 90)
    print(f"RMSE percentil: {rmse_percentil}")
    plt.hist(rmse, bins=100, range=(0, 0.2))
    plt.title("RMSE-distribution")
    plt.grid()
    # plt.savefig(f"{IMAGE_PATH}/RMSE-distribution.png")
    # plt.clf()
    plt.show()
    good = rmse < rmse_percentil
    bad = rmse >= rmse_percentil
    for x_k, y_k, e in zip(['A1', 'A2', 'T1', 'T2'], ['A1_hat', 'A2_hat', 'T1_hat', 'T2_hat'],
                           ['err_A1_sc', 'err_A2_sc', 'err_T1_sc', 'err_T2_sc']):
        rmse_component = np.abs(df_err_sc[e].values)  # spoctu chybu pro kazdou komponentu
        component_percentil = np.percentile(rmse_component, 90)
        component_bad = rmse_component >= component_percentil
        component_good = rmse_component < component_percentil

        # plt.scatter(data[x_k], data[y_k], s=0.1, label="OK-90", color="green")
        # plt.scatter(data[x_k][bad], data[y_k][bad], s=0.1, label="BAD-90", color="red")
        plt.scatter(data[x_k][component_good], data[y_k][component_good], s=0.1, label="OK-90-component", color="green")
        plt.scatter(data[x_k][component_bad], data[y_k][component_bad], s=0.1, label="BAD-90-component", color="red")

        plt.title(f"{x_k} vs {y_k}")
        plt.grid()
        plt.legend()
        # plt.savefig(f"{IMAGE_PATH}/{x_k}-vs-{y_k}-corr.png")
        # plt.clf()
        plt.show()


if __name__ == "__main__":
    """Index(['Unnamed: 0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5',
       'layer6', 'layer7', 'layer8', 'layer9', 'layer10', 'layer11', 'layer12',
       'layer13', 'layer14', 'layer15', 'layer16', 'layer17', 'layer18',
       'layer19', 'layer20', 'layer21', 'A1', 'A2', 'T1', 'T2', 'A1_sc',
       'A2_sc', 'T1_sc', 'T2_sc', 'A1_hat', 'A2_hat', 'T1_hat', 'T2_hat',
       'A1_hat_sc', 'A2_hat_sc', 'T1_hat_sc', 'T2_hat_sc'],
      dtype='object')"""
    #data = pd.read_csv("../Res/analyza-neuspechu/DEEP-TEST-2-UNITS-512-BATCH-4096-EPOCHS-1000.csv")
    data = pd.read_csv("../Res/analyza-neuspechu/SVR-TEST.csv")
    #data = pd.read_csv("../Res/analyza-neuspechu/XGBOOST-TEST.csv")

    err_sc = data[['A1_sc', 'A2_sc', 'T1_sc', 'T2_sc']].values - data[
        ['A1_hat_sc', 'A2_hat_sc', 'T1_hat_sc', 'T2_hat_sc']].values

    df_err_sc = pd.DataFrame(data=err_sc, columns=['err_A1_sc', 'err_A2_sc', 'err_T1_sc', 'err_T2_sc'])
    print(df_err_sc.describe())
    #draw_component_error_distribution(df_err_sc)
    draw_mse_distribution(df_err_sc)
    #draw_rmse_distribution(df_err_sc)

    ml_df = data[['layer1', 'layer2', 'layer3', 'layer4', 'layer5',
       'layer6', 'layer7', 'layer8', 'layer9', 'layer10', 'layer11', 'layer12',
       'layer13', 'layer14', 'layer15', 'layer16', 'layer17', 'layer18',
       'layer19', 'layer20', 'layer21']].copy()

    rmse = np.sqrt(np.sum(df_err_sc.values ** 2, axis=1) / 4)
    rmse_percentil = np.percentile(rmse, 90)
    for e in ['err_A1_sc', 'err_A2_sc', 'err_T1_sc', 'err_T2_sc']:
        rmse_component = np.abs(df_err_sc[e].values)  # spoctu chybu pro kazdou komponentu
        component_percentil = np.percentile(rmse_component, 90)
        component_bad = rmse_component >= component_percentil
        ml_df[e] = component_bad

    for v,hat in zip(['A1_sc', 'A2_sc', 'T1_sc', 'T2_sc'], ['A1_hat_sc', 'A2_hat_sc', 'T1_hat_sc', 'T2_hat_sc']):
        print(f"R2 {v} vs {hat}: {r2(data[v], data[hat])}")


    #ml_df.to_csv("../Res/analyza-neuspechu/ERRORS-DEEP-TRAIN-2-UNITS-512-BATCH-4096-EPOCHS-1000-ML.csv")
    print(ml_df[['err_A1_sc', 'err_A2_sc', 'err_T1_sc', 'err_T2_sc']].corr())





