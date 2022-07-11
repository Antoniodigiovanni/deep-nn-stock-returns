import os
import torch
import torch.nn as nn
from datetime import datetime as dt
from argparse import ArgumentParser
import argparse

parser = ArgumentParser()

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
    SignalsPath = (dataPath + '/external/signed_predictors_dl_wide.csv'), 
    ProcessedDataPath = (dataPath + '/processed'),
    PredictedRetPath = (dataPath + '/processed/predicted_ret.csv'),
    modelsPath = (currentPath + '/../../saved/models'),
    resultsPath = (currentPath + '/../../saved/results'),
    logsPath = (currentPath + '/../../saved/logs')
    )

logFileName = dt.now().strftime('/TrainRun-%Y_%m_%d-%H_%M.log')
bestParamsFileName = dt.now().strftime('/BestNeuralNetworkParameters-%Y_%m_%d-%H_%M.json') 
SavedNetFileName = dt.now().strftime('/NeuralNetwork-%Y_%m_%d-%H_%M.pt')

# Variables
ForcePreProcessing = False # Used to force the data pre-processing even if the processed dataset already exists
ForceTraining = True #Used to force the training of the model even when a saved model already exists

end_train = '198512' 
end_val = '199512'

parser.add_argument("--end_train", default=end_train, type=str)
parser.add_argument("--end_val", default=end_val, type=str)

batch_size_validation = 128
ep_log_interval = 20
epochs = 2

parser.add_argument('--epochs', default=epochs, type=int)
parser.add_argument('--batch_size_validation', default=batch_size_validation, type=int)
parser.add_argument('--ep_log_interval', default=ep_log_interval, type=int)



""" Portfolio creation configs"""
n_cuts = 10 # Number of quantiles in which returns are divided to construct portfolios
rebalancing_frequency = 'yearly' # choose between yearly, monthly, and quarterly
weighting = 'VW' # choose between Value Weighting and Equal Weighting

parser.add_argument('--n_cuts_portfolio', default=n_cuts, type=int)
parser.add_argument('--rebalancing_frequency', default=rebalancing_frequency, type=str)
parser.add_argument('--weighting', default=weighting, type=str)

###################################
# Additional important parameters #
###################################
parser.add_argument('--tuningExperiment', action=argparse.BooleanOptionalAction)
parser.add_argument('--predict', action=argparse.BooleanOptionalAction)
parser.add_argument('--guNetworkTuning', action=argparse.BooleanOptionalAction)

#args = parser.parse_args()
args, unknown = parser.parse_known_args() # Using this to avoid error with notebooks


n_cuts = args.n_cuts_portfolio
print(f'No. of epochs is: {args.epochs}')