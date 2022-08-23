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

        acc, self.pred_df = self.prediction_loop()
        
        print(f'Avg accuracy over the test set at {self.__pct*100}% is: {round(acc)}%')

        
    def prediction_loop(self):
        self.__model.eval()
        prediction = {}
        accuracies = []
        total_correct = 0
        total = 0
        
        for index, data in enumerate(self.__test_loader):
            #print(f'loop n. {index+1}')
            #print(f'Test loader batch shape:')
            #print(data['X'].shape)
            correct, batch_prediction = metric.calc_accuracy_and_predict(self.__model, data, self.__pct)
            total += data['X'].size(0)
            total_correct += correct
            #accuracies.append(accuracy)
            
            for k in batch_prediction:
                if index == 0:
                    prediction[k] = []
                prediction[k].extend(batch_prediction[k])

        accuracy = 100.*total_correct/total   
        pred_df = pd.DataFrame.from_dict(prediction) 
        
        return accuracy, pred_df