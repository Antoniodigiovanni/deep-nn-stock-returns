import torch.nn as nn
import numpy as np
from tuning.tuning_utils import *

class RegressionNet(nn.Module):
    """
        Modify the class to implement further steps needed in the code
        (i.e. flexibility in the architecture, dropout, ...) 

    """

     # define model elements
    def __init__(self, n_inputs, params):
        super(RegressionNet, self).__init__()
        
        self.act_func = map_act_func(params['act_func'])
        self.last_layer = n_inputs
        self.dropout_prob = params['dropout_prob']
        
        # Batch norm is currently not included in the search space
        # self.batch_norm = params['batch_norm']
        self.batch_norm = 0

        layers = []
        
        if params['hidden_layer1'] != 0:
            layers.extend(self.create_layer(n_inputs, params['hidden_layer1']))
        if params['hidden_layer2'] != 0:
            layers.extend(self.create_layer(params['hidden_layer1'], params['hidden_layer2']))
        if params['hidden_layer3'] != 0:
            layers.extend(self.create_layer(params['hidden_layer2'], params['hidden_layer3']))
        if params['hidden_layer4'] != 0:
            layers.extend(self.create_layer(params['hidden_layer3'], params['hidden_layer4']))
        if params['hidden_layer5'] != 0:
            layers.extend(self.create_layer(params['hidden_layer4'], params['hidden_layer5']))
        if params['hidden_layer6'] != 0:
            layers.extend(self.create_layer(params['hidden_layer5'], params['hidden_layer6']))
        if params['hidden_layer7'] != 0:
            layers.extend(self.create_layer(params['hidden_layer6'], params['hidden_layer7']))
        if params['hidden_layer8'] != 0:
            layers.extend(self.create_layer(params['hidden_layer7'], params['hidden_layer8']))
        if params['hidden_layer9'] != 0:
            layers.extend(self.create_layer(params['hidden_layer8'], params['hidden_layer9']))
        if params['hidden_layer10'] != 0:
            layers.extend(self.create_layer(params['hidden_layer9'], params['hidden_layer10']))

        # Last layer
        layers.append(nn.Linear(self.last_layer, 1))
        
        self.fc = nn.Sequential(*layers)
        print(self.fc)


    def create_layer(self, size1, size2):
        self.last_layer = size2
        layers = []
        # Linear layer
        layers.append(nn.Linear(size1, size2))
        # Batch Normalization
        if self.batch_norm != 0:
            layers.append(nn.BatchNorm1d(size2))
        # Activation Function
        layers.append(self.act_func)
        # Dropout
        layers.append(nn.Dropout(self.dropout_prob))

        return layers
        
    # forward propagate input
    def forward(self, features):
        X = self.net(features)
        return X.squeeze()