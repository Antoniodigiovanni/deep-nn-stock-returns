import sys,os

# To import config from top_level folder
currentPath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentPath+'/../')
print(currentPath)
from models.neural_net.gu_et_al_NN4 import GuNN4
from trainer.expanding_window_trainer import ExpandingWindowTraining
import nni
from torch import nn
import torch
import logging
import torch.optim as optim
import config
import pandas as pd
from data.base_dataset import BaseDataset
from data.custom_dataset import CustomDataset
from data.data_preprocessing import *
from torch.utils.data import DataLoader
from trainer.trainer import NeuralNetTrainer



logger = logging.getLogger('Grid search experiment')


# These are the hyperparameters that will be tuned.
params = {
    'learning_rate': 0.0005,
    'l1_lambda1': 5e-6,
    'patience': 2,
    'adam_beta_1': 0.9,
    'adam_beta_2': 0.999
}

# Get optimized hyperparameters
# -----------------------------
# If run directly, :func:`nni.get_next_parameter` is a no-op and returns an empty dict.
# But with an NNI *experiment*, it will receive optimized hyperparameters from tuning algorithm.
optimized_params = nni.get_next_parameter()
params.update(optimized_params)

print(params)

# Load data
if config.ForcePreProcessing == False and os.path.exists(config.paths['ProcessedDataPath']+'/dataset.csv'):
    print('Trying to load data')
    crsp = pd.read_csv(config.paths['ProcessedDataPath']+'/dataset.csv', index_col=0)
    print('Data Loaded')
else:
    print('Data Pre-processing will start soon')
    data = BaseDataset().load_dataset_in_memory()
    crsp = pd.read_csv(config.paths['ProcessedDataPath']+'/dataset.csv', index_col=0)    
    del data

# Model should be included here but I don't have n_inputs at this point, how to correct?

# n_inputs = train.data.shape[1]
#model = GuNN4(n_inputs, params).to(config.device)
#optimizer = optim.Adam(model.parameters(),params['learning_rate'],betas=[params['adam_beta_1'], params['adam_beta_2']])
#loss_fn = None



print('Starting Training process')
trainer = ExpandingWindowTraining(crsp, params)
trainer.fit()