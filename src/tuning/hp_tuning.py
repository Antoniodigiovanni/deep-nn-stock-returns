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

torch.manual_seed(21)
logger = logging.getLogger('Tuning experiment')


""" Nested PARAMETERS"""
params = {
    
        # 'hidden_layer1':    {"_name": "linear", "size":2048},
    
        'hidden_layer1':    1024,
        'hidden_layer2':    512,
        'hidden_layer3':    128,
        'hidden_layer4':    0,
        'hidden_layer5':    0,
        'hidden_layer6':    0,
        'hidden_layer7':    0,
        'hidden_layer8':    0,
        'hidden_layer9':    0,
        'hidden_layer10':    0,
        'act_func':         "ReLU",
        'learning_rate':    0.001,
        'optimizer':        "Adam",
        'l1_lambda1':       0,
        'l2_lambda':        0,
        'dropout_prob':     0.1,
        "batch_norm":       0
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
        hidden_layers = [None]*10

        for param in params.keys():
            result = ''.join([i for i in param if not i.isdigit()])
            n_layer = ''.join([i for i in param if i.isdigit()])
            if result == 'hidden_layer':
                hidden_layers[int(n_layer)-1] = params[param]

        non_zero_layers = [i for i, e in enumerate(hidden_layers) if e != 0]
        invalid_parameters = False
        for i,_ in enumerate(non_zero_layers[:-1]):
            if (non_zero_layers[i+1] != non_zero_layers[i]+1):
                invalid_parameters = True

        return invalid_parameters

def initialize_weights(m):
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
    
invalid_params = validate_params(params)


if invalid_params: 
    # for invalid param combinations, report the worst possible result
    print('Invalid Parameters set')
    nni.report_final_result(np.inf)

else:     

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
    # loss_fn = nn.L1Loss()
    
    if args.expandingTuning:
        method = 'expanding'
    elif args.normalTuning:
        method = 'normal'

    if 'l1_lambda1' in params:
        l1_reg= True
    else:
        l1_reg = False
        
    if 'l2_lambda' in params:
        l2_reg = True
    else:
        l2_reg = False
    trainer = GeneralizedTrainer(df, params, loss_fn, methodology=method, l1_reg=l1_reg, l2_reg=l2_reg)
    n_inputs = trainer.n_inputs

    model = OptimizeNet(n_inputs, params)#.to(config.device)
    model= nn.DataParallel(model)
    model.to(config.device)
    
    print(f'Device from config: {config.device}')
    print(f'N. of epochs set at {config.epochs}')

    print('Initializing weights')
    model.apply(initialize_weights)

    optimizer = map_optimizer(params, model.parameters())

    print('Starting Training process')
    trainer.fit(model, optimizer)
