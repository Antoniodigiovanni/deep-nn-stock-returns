import torch.nn as nn
from tuning.tuning_utils import *

class OptimizeNet(nn.Module):
    def __init__(self, n_inputs, params):
        super(OptimizeNet, self).__init__()
        self.batch_norm = None
        if params['batch_norm'] in params:
            self.batch_norm = params['batch_norm']
        
        self.hidden_size_1 = params['hidden_layer1']
        self.hidden_size_2 = params['hidden_layer2']
        self.hidden_size_3 = params['hidden_layer3']
        self.hidden_size_4 = params['hidden_layer4']
        self.hidden_size_5 = params['hidden_layer5']
        self.hidden_size_6 = params['hidden_layer6']
        self.hidden_size_7 = params['hidden_layer7']
        self.hidden_size_8 = params['hidden_layer8']
        self.hidden_size_9 = params['hidden_layer9']
        self.hidden_size_10 = params['hidden_layer10']
          
        

        self.act_func = map_act_func(params['act_func'])

        self.fc1 = self._fc_block(n_inputs, self.hidden_size_1, self.act_func)
        if self.hidden_size_2 > 0:
            self.fc2 = self._fc_block(self.hidden_size_1, self.hidden_size_2, self.act_func)
            last_layer_size = self.hidden_size_2
        if self.hidden_size_3 > 0:
            self.fc3 = self._fc_block(self.hidden_size_2, self.hidden_size_3, self.act_func)
            last_layer_size = self.hidden_size_3
        if self.hidden_size_4 > 0:
            self.fc4 = self._fc_block(self.hidden_size_3, self.hidden_size_4, self.act_func)
            last_layer_size = self.hidden_size_4
        if self.hidden_size_5 > 0:
            self.fc5 = self._fc_block(self.hidden_size_4, self.hidden_size_5, self.act_func)
            last_layer_size = self.hidden_size_5
        if self.hidden_size_6 > 0:
            self.fc6 = self._fc_block(self.hidden_size_5, self.hidden_size_6, self.act_func)
            last_layer_size = self.hidden_size_6
        if self.hidden_size_7 > 0:
            self.fc7 = self._fc_block(self.hidden_size_6, self.hidden_size_7, self.act_func)
            last_layer_size = self.hidden_size_7
        if self.hidden_size_8 > 0:
            self.fc8 = self._fc_block(self.hidden_size_7, self.hidden_size_8, self.act_func)
            last_layer_size = self.hidden_size_8
        if self.hidden_size_9 > 0:
            self.fc9 = self._fc_block(self.hidden_size_8, self.hidden_size_9, self.act_func)
            last_layer_size = self.hidden_size_9
        if self.hidden_size_10 > 0:
            self.fc10 = self._fc_block(self.hidden_size_9, self.hidden_size_10, self.act_func)
            last_layer_size = self.hidden_size_10
        

        # self.out = self._fc_block(last_layer_size, 1, act_func)
        self.out = nn.Linear(last_layer_size, 1)
        # self.tanh = nn.Tanh()
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
        if self.hidden_size_6:
            x = self.fc6(x)
        if self.hidden_size_7:
            x = self.fc7(x)
        if self.hidden_size_8:
            x = self.fc8(x)
        if self.hidden_size_9:
            x = self.fc9(x)
        if self.hidden_size_10:
            x = self.fc10(x)
        x = self.out(x)
        return x.squeeze()
    
    def _fc_block(self, in_c, out_c, act_func):
        if self.batch_norm == 1:
            block = nn.Sequential(
                nn.Linear(in_c, out_c),
                nn.BatchNorm1d(out_c),
                act_func,
            )
        else:
            block = nn.Sequential(
                nn.Linear(in_c, out_c),
                act_func
            )
            
        return block


class OptimizeNet_v2(nn.Module):
    def __init__(self, n_inputs, params):
        super(OptimizeNet_v2, self).__init__()
        # self.hidden_sizes = []
        # self.hidden_sizes.append(params['hidden_layer1'])
        # self.hidden_sizes.append(params['hidden_layer2'])
        # self.hidden_sizes.append(params['hidden_layer3'])
        # self.hidden_sizes.append(params['hidden_layer4'])
        # self.hidden_sizes.append(params['hidden_layer5'])


        self.hidden_size_1 = params['hidden_layer1']
        self.hidden_size_2 = params['hidden_layer2']
        self.hidden_size_3 = params['hidden_layer3']
        self.hidden_size_4 = params['hidden_layer4']
        self.hidden_size_5 = params['hidden_layer5']
        self.hidden_size_6 = params['hidden_layer6']
        self.hidden_size_7 = params['hidden_layer7']
        self.hidden_size_8 = params['hidden_layer8']
        self.hidden_size_9 = params['hidden_layer9']
        self.hidden_size_10 = params['hidden_layer10']
          
        
        self.act_func = map_act_func(params['act_func'])
        # modules = []
        # for idx,layer in enumerate(self.hidden_sizes):
        #     if layer>0:
        #         if idx == 0:
        #             modules.append(nn.Linear(n_inputs, layer))
        #             modules.append(self.act_func)
        #         else:
        #             modules.append(nn.Linear(self.hidden_sizes[idx-1], layer))
        #             modules.append(self.act_func)
        # modules.append(nn.Linear(self.hidden_sizes[-1], 1))

        self.fc = torch.nn.Sequential()
        
        if self.hidden_size_1 > 0: 
            self.fc.add_module("layer1", nn.Linear(n_inputs, self.hidden_size_1))
            self.fc.add_module("act_func1", self.act_func)
            last_layer_size = self.hidden_size_1
        if self.hidden_size_2 > 0: 
            self.fc.add_module("layer2", nn.Linear(self.hidden_size_1, self.hidden_size_2))
            self.fc.add_module("act_func2", self.act_func)
            last_layer_size = self.hidden_size_2
        if self.hidden_size_3 > 0: 
            self.fc.add_module("layer3", nn.Linear(self.hidden_size_2, self.hidden_size_3))
            self.fc.add_module("act_func3", self.act_func)
            last_layer_size = self.hidden_size_3
        if self.hidden_size_4 > 0: 
            self.fc.add_module("layer4", nn.Linear(self.hidden_size_3, self.hidden_size_4))
            self.fc.add_module("act_func4", self.act_func)
            last_layer_size = self.hidden_size_4
        if self.hidden_size_5 > 0: 
            self.fc.add_module("layer5", nn.Linear(self.hidden_size_4, self.hidden_size_5))
            self.fc.add_module("act_func5", self.act_func)
            last_layer_size = self.hidden_size_5

        self.fc.add_module("last_layer", nn.Linear(last_layer_size, 1))
        # Continue with conversion from module to NN.sequential

        
    def forward(self, x):
        x = self.fc(x)
        return x.squeeze()
