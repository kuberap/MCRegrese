import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __name__ == "__main__":
    """Index(['Unnamed: 0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5',
       'layer6', 'layer7', 'layer8', 'layer9', 'layer10', 'layer11', 'layer12',
       'layer13', 'layer14', 'layer15', 'layer16', 'layer17', 'layer18',
       'layer19', 'layer20', 'layer21', 'A1', 'A2', 'T1', 'T2', 'A1_sc',
       'A2_sc', 'T1_sc', 'T2_sc', 'A1_hat', 'A2_hat', 'T1_hat', 'T2_hat',
       'A1_hat_sc', 'A2_hat_sc', 'T1_hat_sc', 'T2_hat_sc'],
      dtype='object')"""
    data = pd.read_csv("../Res/analyza-neuspechu/DEEP-TEST-2-UNITS-512-BATCH-4096-EPOCHS-1000.csv")
    to_predict = np.array(
        [41.79857899, 47.80665522, 44.95921279, 43.75694528, 40.0470282, 35.33130118, 1260.492509, 594.2514374,
         441.3921548, 418.7659755, 402.7798095, 358.1815222, 336.9068776, 321.1603479, 287.4529931, 266.1460777,
         240.1359104, 222.0845567, 197.2556906, 156.0158626, 141.8509294]) # posledni hodnota vyjmuta 141.8509294

    for i,l in enumerate(['layer1', 'layer2', 'layer3', 'layer4', 'layer5',
       'layer6', 'layer7', 'layer8', 'layer9', 'layer10', 'layer11', 'layer12',
       'layer13', 'layer14', 'layer15', 'layer16', 'layer17', 'layer18',
       'layer19', 'layer20', 'layer21']):

        plt.hist(data[l], bins=100)
        plt.vlines(to_predict[i], 0, 100, label=f"signal={to_predict[i]}", color="red")
        plt.legend()
        plt.title(l)
        plt.grid()
        #plt.show()
        plt.savefig(f"../Res/analyza-neuspechu/images/distribuce_input/{l}-distribution.png")
        plt.clf()
