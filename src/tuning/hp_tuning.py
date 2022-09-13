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
from data.dataset import BaseDataset
from data.data_preprocessing import *
from torch.utils.data import DataLoader
from tuning_utils import *
from trainer.trainer import GeneralizedTrainer
import argparse
from argparse import ArgumentParser
from models.neural_net.Optimize_Net import OptimizeNet
#import base.set_random_seed

torch.manual_seed(21)
logger = logging.getLogger('Tuning experiment')


""" Nested PARAMETERS"""
params = {
    
        # 'hidden_layer1':    {"_name": "linear", "size":2048},
        # 'hidden_layer2':    {"_name": "linear", "size":512},
        # 'hidden_layer3':    {"_name": "linear", "size":128},
        # 'hidden_layer4':    {"_name": "linear", "size":64},
        # 'hidden_layer5':    {"_name": "linear", "size":32},
        # 'use_dropout':      {'_name': 1, 'rate': 0.5},
        'hidden_layer1':    1024,
        'hidden_layer2':    512,
        'hidden_layer3':    128,
        'hidden_layer4':    0,
        'hidden_layer5':    0,
        'act_func':         "ReLU",
        'learning_rate':    0.001,
        'optimizer':        "Adam"
        # 'optimizer':        {"_name": "RMSprop", "momentum": 0.5},
        # 'use_l1_reg':       {"_name": "True", "lambda": 1e-5}
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
    if params['hidden_layer2'] == 0 and (params['hidden_layer3'] != 0 or params['hidden_layer4'] != 0 or params['hidden_layer5'] != 0):
        return False
    if params['hidden_layer3'] == 0 and (params['hidden_layer4'] != 0 or params['hidden_layer5'] != 0):
        return False
    if params['hidden_layer4'] == 0 and params['hidden_layer5'] != 0:
        return False

    return True

if not validate_params(params): 
    # for invalid param combinations, report the worst possible result
    print('Invalid Parameters set')
    nni.report_final_result(np.inf)
        

# Load data
dataset = BaseDataset()
df = dataset.df
    
df.drop(['melag', 'prc', 'me','me_nyse20'], axis=1, inplace=True, errors='ignore')

parser = ArgumentParser()
parser.add_argument('--expandingTuning', action='store_true')
parser.add_argument('--normalTuning', action='store_true')


args, unknown = parser.parse_known_args() # Using this to avoid error with notebooks



# loss_fn = map_loss_func(params['loss'])
loss_fn = nn.MSELoss()
if args.expandingTuning:
    method = 'expanding'
elif args.normalTuning:
    method = 'normal'

# if params['use_l1_reg']['_name'] == 'True':
#     l1_reg = True
# else:
#     l1_reg = False
# print(f'l1_reg is of type: {type(l1_reg)} and is: {l1_reg}')    
l1_reg= False

trainer = GeneralizedTrainer(df, params, loss_fn, methodology=method, l1_reg=l1_reg)
n_inputs = trainer.n_inputs

model = OptimizeNet(n_inputs, params).to(config.device)
print(f'Device from config: {config.device}')
print(f'N. of epochs set at {config.epochs}')

print('Initializing weights')
def initialize_weights(m):
    # print(m)
    if isinstance(m, nn.Linear):
        if params['act_func'] == 'LeakyReLU':
            # print('Activation Function is LeakyReLU')
            nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        elif params['act_func'] == 'ReLU':
            # print('Activation Function is ReLU')
            nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        else:
            # print('Xavier Uniform for other activation functions')
            nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(initialize_weights)

optimizer = map_optimizer(params, model.parameters())

print('Starting Training process')
trainer.fit(model, optimizer)
