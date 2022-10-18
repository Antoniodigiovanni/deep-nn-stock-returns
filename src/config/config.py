import os
import torch
import torch.nn as nn
from datetime import datetime as dt
from argparse import ArgumentParser
import argparse

parser = ArgumentParser()


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    hpoResultsPath = (currentPath + '/../../saved/results' + '/tuningResults'),
    finalDatasetPath=(dataPath + '/processed/dataset.csv')
    )

logFileName = dt.now().strftime('/TrainRun-%Y_%m_%d-%H_%M.log')
bestParamsFileName = dt.now().strftime('/BestNeuralNetworkParameters-%Y_%m_%d-%H_%M.json') 
SavedNetFileName = dt.now().strftime('/NeuralNetwork-%Y_%m_%d-%H_%M.pt')


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
parser.add_argument('--ensemblePrediction', action='store_true', help='Use the final network to predict portfolio returns using an ensemble method')
parser.add_argument('--guEnsemblePrediction', action='store_true', help='')
parser.add_argument('--ensemblePrediction', action='store_true', help='')
parser.add_argument('--finalTuning', action='store_true', help='Used for performing the final operations when an optimal architecture is found (Grid search + Ensemble to create portfolios)')
parser.add_argument('--expandingLearningRateTuning', action='store_true', help='')




#args = parser.parse_args()
args, unknown = parser.parse_known_args() # Using this to avoid error with notebooks

saveDir = paths['resultsPath'] + '/' + args.saveDirName


import configparser

configuration = configparser.ConfigParser()
configuration.read(currentPath + '/config.ini')

# Variables
ForcePreProcessing = configuration.get('Data', 'ForcePreProcessing') 
ForceCrspDownload = configuration.get('Data', 'ForceCrspDownload')

if ForcePreProcessing == 'False':
    ForcePreProcessing = False
elif ForcePreProcessing == 'True':
    ForcePreProcessing = True

if ForceCrspDownload == 'False':
    ForceCrspDownload = False
elif ForceCrspDownload == 'True':
    ForceCrspDownload = True



ep_log_interval = int(configuration.get('Training', 'ep_log_interval'))
epochs = int(configuration.get('Training', 'epochs'))


n_cuts = int(configuration.get('Portfolios', 'n_cuts'))
rebalancing_frequency = configuration.get('Portfolios', 'rebalancing_frequency')
weighting = configuration.get('Portfolios', 'weighting')

n_train_years = int(configuration.get('Training', 'number_initial_training_years'))
n_val_years = int(configuration.get('Training', 'number_validation_years'))

print(f'Epochs: {epochs}')
