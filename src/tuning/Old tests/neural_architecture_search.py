from nni.experiment import Experiment
import pandas as pd
import config
from data.data_preprocessing import *
from data.custom_dataset import CustomDataset
from nni.retiarii import model_wrapper


search_space = {
    "hidden_size_1": {"_type": "choice", "_value": [4, 8, 16, 32, 64, 128]},
    "hidden_size_2": {"_type": "choice", "_value": [0, 4, 8, 16, 32]},
    "hidden_size_3": {"_type": "choice", "_value": [0, 4, 8, 16, 32]}
}

import nni.retiarii.nn.pytorch as nn
import torch.nn.functional as F

""" class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.dropout1 = nn.Dropout(0.0)

        self.fc2 = nn.Linear(16,16)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(0.0)

        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = F.sigmoid(self.fc3(x))
        return x
        

 """
@model_wrapper 
class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.hidden_dim1 = nn.ValueChoice(
            [4,8,16,32,64,128,256,512,1024], label='hidden_dim1')
        self.hidden_dim2 = nn.ValueChoice(
            [0,4,8,16,32,64,128,256,512], label='hidden_dim2')
        
        self.fc1 = nn.Linear(input_size, self.hidden_dim1)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim1)
        self.dropout1 = nn.Dropout(nn.ValueChoice([0.0,0.25,0.5]))

        self.fc2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim2)
        self.dropout2 = nn.Dropout(nn.ValueChoice([0.0,0.25,0.5]))

        self.fc3 = nn.Linear(self.hidden_dim2, 1)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = F.sigmoid(self.fc3(x))
        return x


import nni.retiarii.strategy as strategy

simple_strategy = strategy.TPEStrategy()

import nni.retiarii.evaluator.pytorch.lightning as pl

crsp = pd.read_csv(config.paths['ProcessedDataPath']+'/dataset.csv', index_col=0)
train, val, _ = split_data(crsp)    
X_train, Y_train = sep_target(train)
X_val, Y_val = sep_target(val)

train_dataset = CustomDataset(X_train, Y_train)
val_dataset = CustomDataset(X_val, Y_val)

model_space = Net(train_dataset.data.shape[1])


trainer = pl.Regression(train_dataloader=pl.DataLoader(train_dataset, batch_size=16),
val_dataloaders=pl.DataLoader(val_dataset, batch_size=16),
max_epochs=20)

from nni.retiarii.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment
#experiment = Experiment("local")

# Configure the instance
""" experiment.config.experiment_name = 'test1'
experiment.config.trial_concurrency = 2
experiment.config.max_trial_number = 5
experiment.config.max_experiment_duration = '10m'
experiment.config.search_space = search_space
experiment.config.trial_command = 'python main.py'
experiment.config.trial_code_directory='./'

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.run(8077)
 """

exp = RetiariiExperiment(model_space, trainer, [], simple_strategy)

exp_config = RetiariiExeConfig('local')
exp_config.experiment_name = 'test1'
exp_config.trial_concurrency = 2
exp_config.max_trial_number = 5
exp_config.max_experiment_duration = '10m'
exp_config.trial_gpu_number = 0
#exp_config.search_space = search_space
#exp_config.trial_command = 'python main.py'
#exp_config.trial_code_directory='./'

#exp_config.config.tuner.name = 'TPE'
#exp_config.config.tuner.class_args['optimize_mode'] = 'maximize'


exp.run(exp_config, 8175)

print('Final model:')
for model_code in exp.export_top_models():
    print(model_code)