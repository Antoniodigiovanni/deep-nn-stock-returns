from argparse import ArgumentParser
import sys,os


# To import config from top_level folder
currentPath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentPath+'/../')
print(currentPath)
from models.neural_net.gu_et_al_NN4 import GuNN4
# from trainer.expanding_window_trainer import ExpandingWindowTraining
import nni
from torch import nn
import torch
import logging
import torch.optim as optim
import pandas as pd
from data.dataset import BaseDataset
from data.data_preprocessing import *
from torch.utils.data import DataLoader
from trainer.trainer import GeneralizedTrainer
import argparse

logger = logging.getLogger('Grid search experiment')


# These are the hyperparameters that will be tuned.
params = {
    'learning_rate': 0.0005,
    'l1_lambda1': 5e-6,
    # 'patience': 2,
    # 'adam_beta_1': 0.9,
    # 'adam_beta_2': 0.999
}

# Get optimized hyperparameters
# -----------------------------
# If run directly, :func:`nni.get_next_parameter` is a no-op and returns an empty dict.
# But with an NNI *experiment*, it will receive optimized hyperparameters from tuning algorithm.
optimized_params = nni.get_next_parameter()
params.update(optimized_params)

print(params)

# Load data
# Load data
dataset = BaseDataset()
df = dataset.df
    
df.drop(['melag', 'prc', 'me','me_nyse20'], axis=1, inplace=True, errors='ignore')


torch.manual_seed(2022)
# torch.use_deterministic_algorithms(True)


# n_inputs = train.data.shape[1]
#model = GuNN4(n_inputs, params).to(config.device)
#optimizer = optim.Adam(model.parameters(),params['learning_rate'],betas=[params['adam_beta_1'], params['adam_beta_2']])
#loss_fn = None

parser = ArgumentParser()
parser.add_argument('--ExpandingBatchTest', action='store_true')
parser.add_argument('--expandingTraining', action='store_true')
parser.add_argument('--normalTraining', action='store_true')



args, unknown = parser.parse_known_args() # Using this to avoid error with notebooks


print('Starting Training process')
# trainer = ExpandingWindowTraining(crsp, params)
# trainer.fit()
loss_fn = nn.L1Loss()
if args.ExpandingBatchTest:
    print('Expanding window - batches fixed in order to do the correlation test')
    trainer = GeneralizedTrainer(df, params, loss_fn, methodology='expanding', l1_reg=True)
elif args.normalTraining:
    trainer = GeneralizedTrainer(df, params, loss_fn, methodology='normal', l1_reg=True)
elif args.expandingTraining:
    trainer = GeneralizedTrainer(df, params, loss_fn, methodology='expanding', l1_reg=True)


n_inputs = trainer.n_inputs

model = GuNN4(n_inputs).to(config.device)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        print('Activation Function is ReLU')
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        m.bias.data.fill_(0.01)

model.apply(initialize_weights)

optimizer = optim.Adam(model.parameters(),
            params['learning_rate'],
            # Uncomment when will be using them as parameters
            # betas=(
            #     params['adam_beta_1'], 
            #     params['adam_beta_2'])
                )
print('Starting Training process')
trainer.fit(model, optimizer)