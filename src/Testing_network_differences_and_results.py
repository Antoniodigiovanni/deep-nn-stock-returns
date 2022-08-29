import config
import pandas as pd
from trainer.trainer import GeneralizedTrainer
import torch
import torch.nn as nn
from tuning.tuning_utils import *
from models.neural_net.Optimize_Net import OptimizeNet_v2
# import base.set_random_seed
import time
start_time = time.time()
torch.manual_seed(21)
#torch.use_deterministic_algorithms(True)
params = {
        'hidden_layer1':    1024,
        'hidden_layer2':    128,
        'hidden_layer3':    32,
        'hidden_layer4':    32,
        'hidden_layer5':    0,
        'act_func':         "ReLU",
        'learning_rate':    0.0005124117523666067,
        'optimizer':        "Adam", #, "momentum": 0},
        }

loss_fn = nn.MSELoss()
crsp = pd.read_csv(config.paths['ProcessedDataPath']+'/dataset.csv', index_col=0)
crsp = crsp.iloc[:,0:-3]
crsp = crsp.loc[(crsp.iloc[:,7:]!=0).any(axis=1)]
crsp.drop(['melag', 'prc', 'me', 'me_nyse20'], axis=1, inplace=True)
# crsp = crsp.iloc[:,0:16]

print('Time until the dataset is loaded')
print("--- %s seconds ---" % (time.time() - start_time))

# #print(crsp.head(10))

trainer = GeneralizedTrainer(crsp, params, loss_fn, methodology='normal', l1_reg=False, nni_experiment=False, train_window_years=config.n_train_years, val_window_years=config.n_val_years)
n_inputs = trainer.n_inputs

model = OptimizeNet_v2(n_inputs, params).to(config.device)
print('Model:')
print(model)
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

