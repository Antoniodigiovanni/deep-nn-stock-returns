import config
import pandas as pd
from trainer.trainer import GeneralizedTrainer
import torch
import torch.nn as nn
from tuning.tuning_utils import *
from models.neural_net.Optimize_Net import OptimizeNet
import base.set_random_seed
import time
start_time = time.time()

torch.use_deterministic_algorithms(True)
params = {
        'hidden_layer1':    1024,
        'hidden_layer2':    512,
        'hidden_layer3':    32,
        'hidden_layer4':    16,
        'hidden_layer5':    0,
        'act_func':         "Tanh",
        'learning_rate':    4.6366142454431495e-06,
        'optimizer':        {"_name": "Nadam"}, #, "momentum": 0},
        'use_l1_reg':       {"_name": "False"}#, "lambda": 1e-5}
    }

class Sequential_Net(nn.Module):
    def __init__(self, n_inputs):
        super(Sequential_Net, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(n_inputs, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            )

    def forward(self, x):
        x = self.fc(x)
        return x


# Third model test
class Not_Sequential_Net(nn.Module):
    def __init__(self, n_inputs) -> None:
        super(Not_Sequential_Net, self).__init__()

        self.linear1 = nn.Linear(n_inputs, 64)
        self.activation1 = nn.LeakyReLU()
        self.linear2 = nn.Linear(64, 8)
        self.activation2 = nn.LeakyReLU()
        self.linear3 = nn.Linear(8,8)
        self.activation3 = nn.LeakyReLU()
        self.linear4 = nn.Linear(8,16)
        self.activation4 = nn.LeakyReLU()
        self.out = nn.Linear(16,1)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.activation3(x)
        x = self.linear4(x)
        x = self.activation4(x)
        x = self.out(x)
        return x


loss_fn = nn.MSELoss()
crsp = pd.read_csv(config.paths['ProcessedDataPath']+'/dataset.csv', index_col=0)

crsp['ret'] = crsp['ret']/100
crsp.drop('melag', axis=1, inplace=True)
crsp.drop('prc', axis=1, inplace=True)
crsp.drop('me', axis=1, inplace=True)
print('Time until the dataset is loaded')
print("--- %s seconds ---" % (time.time() - start_time))

# #print(crsp.head(10))

trainer = GeneralizedTrainer(crsp, params, loss_fn, methodology='normal', l1_reg=False, nni_experiment=False, train_window_years=config.n_train_years, val_window_years=config.n_val_years)
n_inputs = trainer.n_inputs

model = Sequential_Net(n_inputs).to(config.device)

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

"""
print('Training of the first model completed, now running the second one:')


model = OptimizeNet(n_inputs, params).to(config.device)
print(f'Printing the model:')
print(model)
print(f'Device from config: {config.device}')
print(f'N. of epochs set at {config.epochs}')
trainer = GeneralizedTrainer(crsp, params, loss_fn, methodology='normal', l1_reg=True, nni_experiment=False)

optimizer = map_optimizer(params['optimizer'], model.parameters(), params['learning_rate'], params['momentum'])

print('Starting Training process')
trainer.fit(model, optimizer)


print('Gu Network Tuning, to check if something changes in this new methodology')
# 20220730-19_43:30:170587- trial_full.json 
# alpha: NaN
# IR: NaN
# R2: 0.45
params = {
    'learning_rate': 0.001,
    'l1_lambda1': 1e-05,
    'patience': 10,
    'adam_beta_1': 0.9,
    'adam_beta_2': 0.999
}

loss_fn = nn.L1Loss()
trainer = GeneralizedTrainer(crsp, params, loss_fn, methodology='normal', l1_reg=True, nni_experiment=False)

n_inputs = trainer.n_inputs
from models.neural_net.gu_et_al_NN4 import GuNN4

model = GuNN4(n_inputs).to(config.device)
optimizer = optim.Adam(model.parameters(),
            params['learning_rate'],
            # Uncomment when will be using them as parameters
            # betas=(
            #     params['adam_beta_1'], 
            #     params['adam_beta_2'])
                )

print('Starting Training process')
trainer.fit(model, optimizer)
"""