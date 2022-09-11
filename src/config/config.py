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
guResultsTimestamp = dt.now().strftime('/GuTuningResults-%m_%d-%H:%M:%S:%f')

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
    logsPath = (currentPath + '/../../saved/logs'),
    guTuningResultsPath = (currentPath + '/../../saved/results' + '/GuTuningResults'),
    hpoResultsPath = (currentPath + '/../../saved/results' + '/tuningResults')
    )

logFileName = dt.now().strftime('/TrainRun-%Y_%m_%d-%H_%M.log')
bestParamsFileName = dt.now().strftime('/BestNeuralNetworkParameters-%Y_%m_%d-%H_%M.json') 
SavedNetFileName = dt.now().strftime('/NeuralNetwork-%Y_%m_%d-%H_%M.pt')


# end_train = '198512' 
# end_val = '199512'

# parser.add_argument("--end_train", default=end_train, type=str)
# parser.add_argument("--end_val", default=end_val, type=str)

# batch_size_validation = 128
# ep_log_interval = 5
# epochs = 100

# parser.add_argument('-e', '--epochs', default=epochs, type=int, help='Set the maximum number of epochs')
# parser.add_argument('--batch_size_validation', default=batch_size_validation, type=int)
# parser.add_argument('--ep_log_interval', default=ep_log_interval, type=int)



""" Portfolio creation configs"""
# n_cuts = 10 # Number of quantiles in which returns are divided to construct portfolios
# rebalancing_frequency = 'yearly' # choose between yearly, monthly, and quarterly
# weighting = 'VW' # choose between Value Weighting and Equal Weighting

# parser.add_argument('--n_cuts_portfolio', default=n_cuts, type=int)
# parser.add_argument('--rebalancing_frequency', default=rebalancing_frequency, type=str)
# parser.add_argument('--weighting', default=weighting, type=str)

###################################
# Additional important parameters #
###################################

parser.add_argument('--expandingTuning', action='store_true', help="Expanding Window training. Architecture and hyperparams from scratch")
parser.add_argument('--normalTuning', action='store_true', help="Normal one-shot training. Architecture and hyperparams from scratch")
# parser.add_argument('--predict', action='store_true')
parser.add_argument('--guNetworkTuning', action='store_true', help="Expanding Window training. Gu et al.'s NN4")
# parser.add_argument('--resumeTuning', action='store_true')
parser.add_argument('--guSimpleTuning', action='store_true', help="Normal one-shot training. Gu et al.'s NN4")
parser.add_argument('--batchExperiment', action='store_true')
parser.add_argument('--saveDirName', default='analysisResults', help='Specifies the path in which experiment results are saved' )



#args = parser.parse_args()
args, unknown = parser.parse_known_args() # Using this to avoid error with notebooks

saveDir = paths['resultsPath'] + '/' + args.saveDirName
print(f'Save Dir is: {saveDir}\nRemember to write the snippet to create the directory if that does not exist')


import configparser

configuration = configparser.ConfigParser()
configuration.read(currentPath + '/config.ini')

# Variables
ForcePreProcessing = configuration.get('Data', 'ForcePreProcessing') 
ForceTraining = configuration.get('Data', 'ForceTraining')

if ForcePreProcessing == 'False':
    ForcePreProcessing = False
elif ForcePreProcessing == 'True':
    ForcePreProcessing = True

if ForceTraining == 'False':
    ForceTraining = False
elif ForceTraining == 'True':
    ForceTraining = True


ep_log_interval = int(configuration.get('Training', 'ep_log_interval'))
epochs = int(configuration.get('Training', 'epochs'))


n_cuts = int(configuration.get('Portfolios', 'n_cuts'))
rebalancing_frequency = configuration.get('Portfolios', 'rebalancing_frequency')
weighting = configuration.get('Portfolios', 'weighting')

n_train_years = int(configuration.get('Training', 'number_initial_training_years'))
n_val_years = int(configuration.get('Training', 'number_validation_years'))

print(f'Epochs: {epochs}')
