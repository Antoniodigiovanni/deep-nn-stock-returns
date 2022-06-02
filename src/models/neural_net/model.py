import torch.nn as nn
import numpy as np

class RegressionNet(nn.Module):
    """
        Modify the class to implement further steps needed in the code
        (i.e. flexibility in the architecture, dropout, ...) 

    """

     # define model elements
    def __init__(self, n_inputs):
        super(RegressionNet, self).__init__()
        
        # Change to traditional method in order to set the 
        # weights and biases initialization method
        
        self.net = nn.Sequential (
            nn.Linear(n_inputs, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        
    # forward propagate input
    def forward(self, features):
        X = self.net(features)
        return X

    def save_model():
        """
            Could be implemented for ease of coding
        """
        pass