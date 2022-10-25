from sqlalchemy import false
import config
import pandas as pd
# from trainer.overfit_test_trainer import GeneralizedTrainer
from trainer.trainer import GeneralizedTrainer
import torch
import torch.nn as nn
from tuning.tuning_utils import *
from models.neural_net.Optimize_Net import OptimizeNet
from data.dataset import BaseDataset
from models.neural_net.gu_et_al_NN4 import GuNN4
import time
import data.data_preprocessing as dp


start_time = time.time()
torch.manual_seed(21)
#torch.use_deterministic_algorithms(True)
params = {
        'hidden_layer1':    64,
        'hidden_layer2':    32,
        'hidden_layer3':    16,
        'hidden_layer4':    0,
        'hidden_layer5':    0,
        'hidden_layer6':    0,
        'hidden_layer7':    0,
        'hidden_layer8':    0,
        'hidden_layer9':    0,
        'hidden_layer10':   0,
        'act_func':         "LeakyReLU",
        'learning_rate':    0.0001,
        'optimizer':        "Adam", #, "momentum": 0},
        'batch_norm':       0,
        'dropout_prob':     0.7,
        'l1_lambda1':       0.5,
        'l2_lambda':        0.4
    }

loss_fn = nn.MSELoss()
dataset = BaseDataset()
crsp = dataset.df

crsp.drop(['melag','prc','me'], axis=1, inplace=True, errors='ignore')
print(crsp.head())


print('Time until the dataset is loaded')
print("--- %s seconds ---" % (time.time() - start_time))

if 'l1_lambda1' in params:
    l1_reg=True
else:
    l1_reg = False

if 'l2_lambda' in params:
    l2_reg=True
else:
    l2_reg = False

trainer = GeneralizedTrainer(crsp, params, loss_fn, methodology='expanding', l1_reg=l1_reg, l2_reg=l2_reg, nni_experiment=False, train_window_years=config.n_train_years, val_window_years=config.n_val_years)
n_inputs = trainer.n_inputs

model = OptimizeNet(n_inputs, params).to(config.device)

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
print(f'Device from config: {config.device}')
print(f'N. of epochs set at {config.epochs}')

optimizer = map_optimizer(params, model.parameters(), )

print('Starting Training process of the first network')
trainer.fit(model, optimizer)

print('Time to run the entire training')
print("--- %s seconds ---" % (time.time() - start_time))
