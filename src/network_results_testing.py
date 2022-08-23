import config
import pandas as pd
from trainer.trainer import GeneralizedTrainer
import torch
import torch.nn as nn
from tuning.tuning_utils import *
from models.neural_net.Optimize_Net import OptimizeNet


torch.manual_seed(0)
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

class Sequential_Net(nn.Module):
    def __init__(self, n_inputs):
        super(Sequential_Net, self).__init__()
        
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
            # nn.LeakyReLU() # There should be no activation function in the last layer
        )
    # Is this forward method necessary?
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

# Use this to check if the models are equal
# for param in model.parameters():
    # print(param)



loss_fn = map_loss_func(params['loss'])
crsp = pd.read_csv(config.paths['ProcessedDataPath']+'/dataset.csv', index_col=0)
# Dividing by 100 helps?
crsp['ret'] = crsp['ret']/100
crsp.drop('melag', axis=1, inplace=True)
crsp.drop('prc', axis=1, inplace=True)
crsp.drop('me', axis=1, inplace=True)

# #print(crsp.head(10))

trainer = GeneralizedTrainer(crsp, params, loss_fn, methodology='normal', l1_reg=True, nni_experiment=False, train_window_years=config.n_train_years, val_window_years=config.n_val_years)
n_inputs = trainer.n_inputs

model = OptimizeNet(n_inputs, params).to(config.device)

def initialize_weights(m):
    print(m)
    if isinstance(m, nn.Linear):
        if params['act_func'] == 'LeakyReLU':
            print('Activation Function is LeakyReLU')
            nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        elif params['act_func'] == 'ReLU':
            print('Activation Function is ReLU')
            nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        else:
            print('Xavier Uniform for other activation functions')
            nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(initialize_weights)
print(f'Device from config: {config.device}')
print(f'N. of epochs set at {config.epochs}')

optimizer = map_optimizer(params['optimizer'], model.parameters(), params['learning_rate'], params['momentum'])

print('Starting Training process of the first network')
trainer.fit(model, optimizer)


"""
print('Training of the first model completed, now running the second one:')


torch.manual_seed(0)

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