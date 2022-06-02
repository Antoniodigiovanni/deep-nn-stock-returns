import sys,os
import torch
import numpy as np
import pandas as pd
from data.custom_dataset import TestDataset
from data.data_preprocessing import sep_target_idx, split_data
import models.nn.metric as metric
from torch.utils.data import DataLoader
import config


def prediction_loop(model, data_loader, pct):
    model.eval()

    Correct = 0
    Wrong = 0
    accuracies = []
    permno = []
    yyyymm = []
    ret = []
    predicted_ret = []

    for batch_index, data in enumerate(data_loader):
        accuracy, stats, prediction = metric.calc_accuracy(model, data, pct)
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


currentPath = os.getcwd()#os.path.dirname(sys.argv[0])
modelsPath = config.paths['modelsPath'] #currentPath + '/../saved/models'
dataPath = config.dataPath
ProcessedDataPath = config.paths['ProcessedDataPath']
if os.path.exists(ProcessedDataPath + '/dataset.csv'):
    crsp = pd.read_csv(ProcessedDataPath+'/dataset.csv', index_col=0)
    print(crsp.shape)
else:
    print('Processed dataset not present, exiting...')
    sys.exit()


if os.path.exists(modelsPath + '/RegressionNet_model.pt') == False:
    print(modelsPath+'/RegressionNet_model.pt')
    print('Model not trained, exiting')
    sys.exit()

model = torch.load(modelsPath + '/RegressionNet_model.pt')
model.eval()

_,_,test = split_data(crsp)
X_test, Y_test, index = sep_target_idx(test)
print(X_test.shape)

test = TestDataset(X_test, Y_test, index)
del[X_test, Y_test, index]

test_loader = DataLoader(test, num_workers=2)

pct = 0.1

accuracies, pred_df = prediction_loop(model, test_loader, pct)
print('Saving df with predicted returns...')
pred_df.to_csv(ProcessedDataPath+'/predicted_ret.csv')
avg_accurancy = np.mean(accuracies) 
print(f'Avg accuracy over the test set at {pct*100}% is: {round(avg_accurancy*100)}%')
