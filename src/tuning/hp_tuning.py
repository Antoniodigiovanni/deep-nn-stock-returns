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
from trainer.trainer import NeuralNetTrainer



logger = logging.getLogger('Tuning experiment')

def map_act_func(af_name):
    if af_name == "ReLU":
        act_func = torch.nn.ReLU()
    elif af_name == "LeakyReLU":
        act_func = torch.nn.LeakyReLU()
    elif af_name == "Sigmoid":
        act_func = torch.nn.Sigmoid()
    elif af_name == "Tanh":
        act_func = torch.nn.Tanh()
    elif af_name == "Softplus":
        act_func = torch.nn.Softplus()
    else:
        sys.exit("Invalid activation function")
    return act_func

def map_optimizer(opt_name, net_params, lr):
    if opt_name == "SGD":
        opt = optim.SGD(net_params, lr=lr)
    elif opt_name == "Adam":
        opt = optim.Adam(net_params, lr=lr)
    elif opt_name == "RMSprop":
        opt = optim.RMSprop(net_params, lr=lr)
    else:
        sys.exit("Invalid optimizer")
    return opt

def map_loss_func(loss_name):
    if loss_name == "MSELoss":
        loss_func = torch.nn.MSELoss()
    elif loss_name == "SmoothL1Loss":
        loss_func = torch.nn.SmoothL1Loss()
    else:
        sys.exit("Invalid loss function")
    return loss_func

# These are the hyperparameters that will be tuned.
params = {
    'hidden_size_1': 4,
    'hidden_size_2': 8,
    'hidden_size_3': 0,
    'hidden_size_4': 0,
    'act_func': 'ReLu',
    'optimizer': 'Adam',
    'loss': 'MSELoss',
    'learning_rate': 0.001,
    'momentum': 0,
    'epochs': 20,
    'batch_size': 64
}

# Get optimized hyperparameters
# -----------------------------
# If run directly, :func:`nni.get_next_parameter` is a no-op and returns an empty dict.
# But with an NNI *experiment*, it will receive optimized hyperparameters from tuning algorithm.
optimized_params = nni.get_next_parameter()
params.update(optimized_params)

params['epochs'] = int(params['epochs'])
params['batch_size'] = int(params['batch_size'])

print(params)

def validate_params(params):
    if params['hidden_size_2'] == 0 and (params['hidden_size_3'] != 0 or params['hidden_size_4'] != 0):
        return False
    if params['hidden_size_3'] == 0 and params['hidden_size_4'] != 0:
        return False
    return True

    
class OptimizeNet(nn.Module):
    def __init__(self, n_inputs, params):
        super(OptimizeNet, self).__init__()
        self.hidden_size_1 = params['hidden_size_1']
        self.hidden_size_2 = params['hidden_size_2']
        self.hidden_size_3 = params['hidden_size_3']
        self.hidden_size_4 = params['hidden_size_4']
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

        self.out = self._fc_block(last_layer_size, 1, act_func)

    def forward(self, x):
        x = self.fc1(x)
        if self.hidden_size_2:
            x = self.fc2(x)
        if self.hidden_size_3:
            x = self.fc3(x)
        if self.hidden_size_4:
            x = self.fc4(x)
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
        nni.report_final_result(0.0)

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


train, val = split_data_train_val(crsp)
X_train, Y_train = sep_target(train)
X_val, Y_val = sep_target(val)

train = CustomDataset(X_train, Y_train)
del [X_train, Y_train]

val = CustomDataset(X_val, Y_val)
del[X_val, Y_val]

train_loader = DataLoader(train, batch_size=params['batch_size'], num_workers=2)
val_loader = DataLoader(val, batch_size=config.batch_size_validation, num_workers=2)


n_inputs = train.data.shape[1]
model = OptimizeNet(n_inputs, params).to(config.device)
optimizer = map_optimizer(params['optimizer'], model.parameters(), params['learning_rate'])
loss_fn = map_loss_func(params['loss'])



print('Starting Training process')
trainer = NeuralNetTrainer(model, train_loader, val_loader, optimizer, loss_fn, params, nni_experiment=True).train()
