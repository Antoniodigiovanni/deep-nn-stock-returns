import sys,os


# To import config from top_level folder
currentPath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentPath+'/../')

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
# from trainer.trainer import NeuralNetTrainer
from tuning_utils import *
from trainer.trainer import GeneralizedTrainer
import argparse
from argparse import ArgumentParser


logger = logging.getLogger('Tuning experiment')

# These are the hyperparameters that will be tuned.
params = {
    'hidden_size_1': 4,
    'hidden_size_2': 8,
    'hidden_size_3': 0,
    'hidden_size_4': 0,
    'hidden_size_5': 0,
    'act_func': 'ReLu',
    'optimizer': 'Adam',
    'loss': 'MSELoss',
    'learning_rate': 0.001,
    'momentum': 0,
    'l1_lambda1': 1,
    "patience": 10
}

# Get optimized hyperparameters
# -----------------------------
# If run directly, :func:`nni.get_next_parameter` is a no-op and returns an empty dict.
# But with an NNI *experiment*, it will receive optimized hyperparameters from tuning algorithm.
optimized_params = nni.get_next_parameter()
params.update(optimized_params)

# params['epochs'] = int(params['epochs'])
# params['batch_size'] = int(params['batch_size'])

print(params)

def validate_params(params):
    if params['hidden_size_2'] == 0 and (params['hidden_size_3'] != 0 or params['hidden_size_4'] != 0 or params['hidden_size_5'] != 0):
        return False
    if params['hidden_size_3'] == 0 and (params['hidden_size_4'] != 0 or params['hidden_size_5'] != 0):
        return False
    if params['hidden_size_4'] == 0 and params['hidden_size_5'] != 0:
        return False

    return True

    
class OptimizeNet(nn.Module):
    def __init__(self, n_inputs, params):
        super(OptimizeNet, self).__init__()
        self.hidden_size_1 = params['hidden_size_1']
        self.hidden_size_2 = params['hidden_size_2']
        self.hidden_size_3 = params['hidden_size_3']
        self.hidden_size_4 = params['hidden_size_4']
        self.hidden_size_5 = params['hidden_size_5']

        act_func = map_act_func(params['act_func'])

        self.fc1 = self._fc_block(n_inputs, self.hidden_size_1, act_func)
        if self.hidden_size_2 > 0:
            self.fc2 = self._fc_block(self.hidden_size_1, self.hidden_size_2, act_func)
            last_layer_size = self.hidden_size_2
        if self.hidden_size_3 > 0:
            self.fc3 = self._fc_block(self.hidden_size_2, self.hidden_size_3, act_func)
            last_layer_size = self.hidden_size_3
        if self.hidden_size_4 > 0:
            self.fc4 = self._fc_block(self.hidden_size_3, self.hidden_size_4, act_func)
            last_layer_size = self.hidden_size_4
        if self.hidden_size_5 > 0:
            self.fc5 = self._fc_block(self.hidden_size_4, self.hidden_size_5, act_func)
            last_layer_size = self.hidden_size_5
        

        self.out = self._fc_block(last_layer_size, 1, act_func)

    def forward(self, x):
        x = self.fc1(x)
        if self.hidden_size_2:
            x = self.fc2(x)
        if self.hidden_size_3:
            x = self.fc3(x)
        if self.hidden_size_4:
            x = self.fc4(x)
        if self.hidden_size_5:
            x = self.fc5(x)
        x = self.out(x)
        return x
    
    def _fc_block(self, in_c, out_c, act_func):
        block = nn.Sequential(
            nn.Linear(in_c, out_c),
            act_func
        )
        return block

if not validate_params(params): 
# for invalid param combinations, report the worst possible result
        print('Invalid Parameters set')
        nni.report_final_result(np.inf)

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

parser = ArgumentParser()
parser.add_argument('--expandingTuning', action=argparse.BooleanOptionalAction)
parser.add_argument('--normalTuning', action=argparse.BooleanOptionalAction)


args, unknown = parser.parse_known_args() # Using this to avoid error with notebooks



loss_fn = map_loss_func(params['loss'])
if args.expandingTuning:
    method = 'expanding'
elif args.normalTuning:
    method = 'normal'

if 'l1_lambda' in params:
    l1_reg = True
else:
    l1_reg = False
    
trainer = GeneralizedTrainer(crsp, params, loss_fn, methodology=method, l1_reg=l1_reg)
n_inputs = trainer.n_inputs

model = OptimizeNet(n_inputs, params).to(config.device)
print(f'Device from config: {config.device}')
print(f'N. of epochs set at {config.epochs}')
optimizer = map_optimizer(params['optimizer'], model.parameters(), params['learning_rate'], params['momentum'])

print('Starting Training process')
trainer.fit(model, optimizer)
