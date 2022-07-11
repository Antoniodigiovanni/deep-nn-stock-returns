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
    def __init__(self, test_loader, model):
        self.__pct = 0.1
        self.pred_df = None        


        self.__model = model
        self.__model.eval()

        self.__test_loader = test_loader

        accuracies, self.pred_df = self.prediction_loop()
        
        avg_accurancy = np.mean(accuracies) 
        print(f'Avg accuracy over the test set at {self.__pct*100}% is: {round(avg_accurancy*100)}%')

        
    def prediction_loop(self):
        self.__model.eval()
        prediction = {}
        accuracies = []
        print('In test_loader loop')
        for index, data in enumerate(self.__test_loader):
            print(f'loop n. {index+1}')
            print(f'Test loader batch shape:')
            print(data['X'].shape)
            accuracy, batch_prediction = metric.calc_accuracy_and_predict(self.__model, data, self.__pct)
            print(batch_prediction)
            batch_prediction['predicted_ret'] = batch_prediction['predicted_ret'].reshape(-1)
            accuracies.append(accuracy)
            for k in batch_prediction:
                if index == 0:
                    prediction[k] = []
                prediction[k].extend(batch_prediction[k])
               
        pred_df = pd.DataFrame.from_dict(prediction) 
        print(pred_df.shape)
        return accuracies, pred_df