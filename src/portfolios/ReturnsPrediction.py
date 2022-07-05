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
        self.pred_df = None        

        # The following if could be deleted because I don't want to load from file at the moment, what is the benefit?
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

        accuracies, self.pred_df = self.prediction_loop()
        # print('Saving df with predicted returns...')
        # pred_df.to_csv(config.paths['ProcessedDataPath']+'/predicted_ret.csv')
        
        avg_accurancy = np.mean(accuracies) 
        print(f'Avg accuracy over the test set at {self.__pct*100}% is: {round(avg_accurancy*100)}%')

        
    def prediction_loop(self):
        self.__model.eval()

        Correct = 0
        Wrong = 0
        accuracies = []
        print('In test_loader loop')
        for index, data in enumerate(self.__test_loader):
            print(f'loop n. {index+1}')
            accuracy, prediction = metric.calc_accuracy_and_predict(self.__model, data, self.__pct)
            accuracies.append(accuracy)
            prediction['predicted_ret'] = prediction['predicted_ret'].reshape(-1)
               
        pred_df = pd.DataFrame.from_dict(prediction) 

        return accuracies, pred_df