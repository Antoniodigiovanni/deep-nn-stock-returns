import os
import torch
import torch.nn as nn
# Select training device - GPU or CPU based on availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')


currentPath = os.path.dirname(os.path.abspath(__file__))
dataPath = (currentPath + '/../../data')

paths = dict (
    CRSPretPath = (dataPath + '/external/crspmret.csv'),
    CRSPinfoPath=(dataPath+'/external/crspminfo.csv'),
    FFPath = (dataPath + '/external/FamaFrenchData.csv'),
    SignalsPath = (dataPath + '/external/signed_predictors_all_wide.csv'),
    ProcessedDataPath = (dataPath + '/processed'),
    modelsPath = (currentPath + '/../../saved/models'),
    resultsPath = (currentPath + '/../../saved/results'),
    logsPath = (currentPath + '/../../saved/logs')
    )

ForcePreProcessing = False
ForceTraining = True

# Hyperparameters space
batch_size_train = 64
batch_size_validation = 128
n_epochs = 200
ep_log_interval = 15
learning_rate = 0.001
loss_fn = nn.MSELoss()
