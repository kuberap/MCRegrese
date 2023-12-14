import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




if __name__=="__main__":
    mlp_data = pd.read_csv("../Res/analyza-neuspechu/DEEP-TEST-2-UNITS-512-BATCH-4096-EPOCHS-1000.csv")
    svr_data = pd.read_csv("../Res/analyza-neuspechu/SVR-TEST.csv")

    mlp_err_sc = mlp_data[['A1_sc', 'A2_sc', 'T1_sc', 'T2_sc']].values - mlp_data[
        ['A1_hat_sc', 'A2_hat_sc', 'T1_hat_sc', 'T2_hat_sc']].values

    mlp_df_err_sc = pd.DataFrame(data=mlp_err_sc, columns=['err_A1_sc', 'err_A2_sc', 'err_T1_sc', 'err_T2_sc'])
    svr_err_sc = svr_data[['A1_sc', 'A2_sc', 'T1_sc', 'T2_sc']].values - svr_data[
        ['A1_hat_sc', 'A2_hat_sc', 'T1_hat_sc', 'T2_hat_sc']].values

    svr_df_err_sc = pd.DataFrame(data=svr_err_sc, columns=['err_A1_sc', 'err_A2_sc', 'err_T1_sc', 'err_T2_sc'])

    for e in ['err_A1_sc', 'err_A2_sc', 'err_T1_sc', 'err_T2_sc']:
        rmse_component = np.abs(mlp_df_err_sc[e].values)  # spoctu chybu pro kazdou komponentu
        component_percentil = np.percentile(rmse_component, 90)
        component_bad = rmse_component >= component_percentil
        mlp_df_err_sc[e+"_OK"] = component_bad
    print(len(mlp_err_sc))
    for e in ['err_A1_sc', 'err_A2_sc', 'err_T1_sc', 'err_T2_sc']:
        rmse_component = np.abs(svr_df_err_sc[e].values)  # spoctu chybu pro kazdou komponentu
        component_percentil = np.percentile(rmse_component, 90)
        component_bad = rmse_component >= component_percentil
        svr_df_err_sc[e+"_OK"] = component_bad
    for e in ['err_A1_sc', 'err_A2_sc', 'err_T1_sc', 'err_T2_sc']:
        bad_mlp_component=set(mlp_df_err_sc.index[mlp_df_err_sc[e+"_OK"]==True])
        bad_svr_component=set(svr_df_err_sc.index[svr_df_err_sc[e+"_OK"]==True])
        jindex=len(bad_mlp_component.intersection(bad_svr_component))/len(bad_mlp_component.union(bad_svr_component))
        print(f"Jaccard index: {jindex}")