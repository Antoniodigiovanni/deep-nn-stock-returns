import sys,os
import torch
import numpy as np
import pandas as pd
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

        self.pred_df = self.predict()
        
        # print(f'Avg accuracy over the test set at {self.__pct*100}% is: {round(acc)}%')

        
    def predict(self):
        self.__model.eval()
        # prediction = {}
        # total_correct = 0
        # total = 0
        
        dataiter = iter(self.__test_loader)
        inputs, target, labels = dataiter.next()

        inputs = inputs.to(config.device)
        target = target.to(config.device)
        labels = labels.to(config.device)
            
        with torch.no_grad():
            outputs = self.__model(inputs.float())
            
            outputs = outputs.to('cpu')
            labels = labels.to('cpu')
            target = target.to('cpu')
            inputs = inputs.to('cpu')
            pred_df = pd.DataFrame(
                {'permno': labels[:,1].numpy(),
                'yyyymm': labels[:,0].numpy(),
                'ret': target.squeeze().numpy(),
                'predicted_ret': outputs.squeeze().numpy()
                }
            )
    
        
        return pred_df