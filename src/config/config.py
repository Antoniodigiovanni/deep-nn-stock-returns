import os
import torch
import torch.nn as nn
from datetime import datetime as dt



# Select training device - GPU or CPU based on availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')


currentPath = os.path.dirname(os.path.abspath(__file__))
dataPath = (currentPath + '/../../data')

paths = dict (
    CRSPretPath = (dataPath + '/external/crspmret.csv'),
    CRSPinfoPath=(dataPath+'/external/crspminfo.csv'),
    FFPath = (dataPath + '/external/F-F_Research_Data_5_Factors_2x3.csv'),
    FFMomPath = (dataPath + '/external/F-F_Momentum_Factor.CSV'), 
    SignalsPath = (dataPath + '/external/signed_predictors_all_wide.csv'), 
    ProcessedDataPath = (dataPath + '/processed'),
    PredictedRetPath = (dataPath + '/processed/predicted_ret.csv'),
    modelsPath = (currentPath + '/../../saved/models'),
    resultsPath = (currentPath + '/../../saved/results'),
    logsPath = (currentPath + '/../../saved/logs')
    )

logFileName = dt.now().strftime('/TrainRun-%Y_%m_%d-%H_%M.log')
bestParamsFileName = dt.now().strftime('/BestNeuralNetworkParameters-%Y_%m_%d-%H_%M.json') 


# Variables
ForcePreProcessing = False # Used to force the data pre-processing even if the processed dataset already exists
ForceTraining = True #Used to force the training of the model even when a saved model already exists

batch_size_validation = 128
ep_log_interval = 2
