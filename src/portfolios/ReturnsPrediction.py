import sys,os
import torch
import numpy as np
import pandas as pd
from data.custom_dataset import TestDataset
from data.data_preprocessing import sep_target_idx, split_data_test
import models.neural_net.metric as metric
from torch.utils.data import DataLoader
import config

class ReturnsPrediction():
    def __init__(self, test_loader, model=None):
        self.__pct = 0.1
        
        if model is None:
            print('Model has not been passed, trying to load from previously saved models')
            if os.path.exists(config.paths['modelsPath'] + config.SavedNetFileName) == False:
                print('Trained model does not exist, exiting..')
                sys.exit()
            else:
                model = torch.load(config.paths['modelsPath'] + config.SavedNetFileName)

        self.__model = model
        self.__model.eval()

        self.__test_loader = test_loader

        accuracies, pred_df = self.prediction_loop(self.__pct)

        # print('Saving df with predicted returns...')
        # pred_df.to_csv(config.paths['ProcessedDataPath']+'/predicted_ret.csv')
        
        avg_accurancy = np.mean(accuracies) 
        print(f'Avg accuracy over the test set at {self.__pct*100}% is: {round(avg_accurancy*100)}%')

        return pred_df

    def prediction_loop(self):
        self.__model.eval()

        Correct = 0
        Wrong = 0
        accuracies = []
        permno = []
        yyyymm = []
        ret = []
        predicted_ret = []

        for batch_index, data in enumerate(self.__test_loader):
            accuracy, stats, prediction = metric.calc_accuracy_and_predict(self.__model, data, self.__pct)
            accuracies.append(accuracy)
            # Stats and Correct, wrong can be deleted as redundant. Same numbers can be calculated with accuracies
            Correct = Correct + stats['Correct']
            Wrong = Wrong + stats['Wrong']
            permno.append(prediction['permno'].item())
            yyyymm.append(prediction['yyyymm'].item())
            ret.append(prediction['ret'].item())
            predicted_ret.append(prediction['predicted_ret'].item())
        
        pred_df = pd.DataFrame(list(zip(permno, yyyymm, ret, predicted_ret)), columns=['permno','yyyymm','ret','predicted_ret'])
        print(f'N. correct: {Correct} | N. wrong: {Wrong}')
    
        return accuracies, pred_df