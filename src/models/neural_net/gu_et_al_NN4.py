import torch.nn as nn
import numpy as np
import torch.optim as optim

class GuNN4(nn.Module):

    # define model elements
    def __init__(self, n_inputs):
        super(GuNN4, self).__init__()
        
        self.net = nn.Sequential (
            # nn.Linear(n_inputs, 32),
            # nn.ReLU(),
            # nn.BatchNorm1d(32),
            # nn.Linear(32, 16),
            # nn.ReLU(),
            # nn.BatchNorm1d(16),
            # nn.Linear(16,8),
            # nn.ReLU(),
            # nn.BatchNorm1d(8),
            # nn.Linear(8,4),
            # nn.ReLU(),
            # nn.BatchNorm1d(4),
            # nn.Linear(4,1)
            nn.Linear(n_inputs, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8,4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Linear(4,1)
            
        )
        
    # forward propagate input
    def forward(self, features):
        X = self.net(features)
        return X.squeeze()