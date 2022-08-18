import config
import pandas as pd
from trainer.trainer import GeneralizedTrainer
import torch
import torch.nn as nn
from tuning.tuning_utils import *

torch.manual_seed(2022)
torch.use_deterministic_algorithms(True)

# 20220806-19_12:05:303739 - trial_full.json
params = {
    'hidden_size_1': 64,
    'hidden_size_2': 8,
    'hidden_size_3': 8,
    'hidden_size_4': 16,
    'hidden_size_5': 0,
    'act_func': 'LeakyReLU',
    'optimizer': 'Adam',
    'loss': 'MSELoss',
    'learning_rate': 0.0007228993329829353,
    'momentum': 0.6450508869222246,
    'l1_lambda1': 0.001,
    "patience": 10
}

class Best_Net(nn.Module):
    def __init__(self, n_inputs):
        super(Best_Net, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
            nn.LeakyReLU()
        )
    # Is this forward method necessary?
    def forward(self, x):
        x = self.fc(x)
        return x


loss_fn = map_loss_func(params['loss'])
crsp = pd.read_csv(config.paths['ProcessedDataPath']+'/dataset.csv', index_col=0)

trainer = GeneralizedTrainer(crsp, params, loss_fn, methodology='expanding', l1_reg=True, nni_experiment=False)
n_inputs = trainer.n_inputs

model = Best_Net(n_inputs).to(config.device)
print(f'Device from config: {config.device}')
print(f'N. of epochs set at {config.epochs}')

optimizer = map_optimizer(params['optimizer'], model.parameters(), params['learning_rate'], params['momentum'])

print('Starting Training process of the first network')
trainer.fit(model, optimizer)

print('Training of the first model completed, now running the second one:')
# Second model
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

torch.manual_seed(2022)

model = OptimizeNet(n_inputs, params).to(config.device)
print(f'Printing the model:')
print(model)
print(f'Device from config: {config.device}')
print(f'N. of epochs set at {config.epochs}')

optimizer = map_optimizer(params['optimizer'], model.parameters(), params['learning_rate'], params['momentum'])

print('Starting Training process')
trainer.fit(model, optimizer)