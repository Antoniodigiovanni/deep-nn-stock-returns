import pandas as pd
# from trainer.overfit_test_trainer import GeneralizedTrainer
from trainer.tpuTrainer import TPUTrainer
import torch
import torch.nn as nn
from tuning.tuning_utils import *
from models.neural_net.Optimize_Net import OptimizeNet
from data.dataset import BaseDataset
from models.neural_net.gu_et_al_NN4 import GuNN4
import time
import data.data_preprocessing as dp
from data.crsp_dataset import CrspDataset

try:
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except:
    pass



def _map_fn(index, flags):
    import config

    config.epochs = 1
    try:
        xm.master_print(f'Device is: {config.device} ')
        xm.master_print('TPU works!!')

        xm.master_print(f'[Master print] World size: {xm.xrt_world_size()}')
        xm.master_print(f'[Master print] Get_ordinal: {xm.get_ordinal()}')

        xm.master_print(f'[Normal print] World size: {xm.xrt_world_size()}')
        xm.master_print(f'[Normal print] Get_ordinal: {xm.get_ordinal()}')
    except:
        pass
    start_time = time.time()
    torch.manual_seed(21)
    
    params = {
            'hidden_layer1':    256,
            'hidden_layer2':    128,
            'hidden_layer3':    64,
            'hidden_layer4':    0,
            'hidden_layer5':    0,
            'act_func':         "LeakyReLU",
            'learning_rate':    4.6366142454431495e-06,
            'optimizer':        "Adam", #, "momentum": 0},
            'batch_norm':       0,
            'dropout_prob':     0.7,
            'l1_lambda1':       0.6,
            'lambda2':          0.7

        }

    class Flexible_Sequential_Net(nn.Module):
        def __init__(self, n_inputs):
            super(Flexible_Sequential_Net, self).__init__()

            self.act_func = map_act_func(params['act_func'])
            self.last_layer = n_inputs
            self.batch_norm = params['batch_norm']
            self.dropout_prob = params['dropout_prob']
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
                        
        def forward(self, x):
            x = self.fc(x)
            return x.squeeze()


    try:
        if not xm.is_master_ordinal():
            xm.rendezvous('load_only_once')
    except:
        pass

    dataset = BaseDataset()
    crsp = dataset.df

    train = crsp.loc[crsp['yyyymm'] <= 198401].copy()
    validation = crsp.loc[(crsp['yyyymm'] >= 198501) & (crsp['yyyymm'] <= 199412)].copy()
    test = crsp.loc[crsp['yyyymm'] >= 199501].copy()
        
    train = CrspDataset(train)
    validation = CrspDataset(validation)
    test = CrspDataset(test)

    try:
        if xm.is_master_ordinal():
            xm.rendezvous('load_only_once')
    except:
        pass

    crsp.drop(['melag','prc','me'], axis=1, inplace=True, errors='ignore')

    print('Time until the dataset is loaded')
    print("--- %s seconds ---" % (time.time() - start_time))

    try:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
        train,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)

        val_sampler = torch.utils.data.distributed.DistributedSampler(
            validation,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False)

          # Creates dataloaders, which load data in batches
          # Note: test loader is not shuffled or sampled

        train_loader = torch.utils.data.DataLoader(
            train,
            batch_size=flags['batch_size'],
            sampler=train_sampler,
            num_workers=flags['num_workers'],
            drop_last=True)

        test_loader = torch.utils.data.DataLoader(
            validation,
            batch_size=flags['batch_size'],
            sampler=val_sampler,
            shuffle=False,
            num_workers=flags['num_workers'],
            drop_last=True)

    except:
        print('Error in creating distributed DataLoader - falling back to normal loaders')
        if len(test) > 0:
            bs = len(test)
        else:
            bs = 10000 

        train_loader = torch.utils.data.DataLoader(train, batch_size=10000, shuffle=True)
        val_loader = torch.utils.data.DataLoader(validation, batch_size=10000, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=bs)

    n_inputs = train.get_inputs()
    model = Flexible_Sequential_Net(n_inputs).to(config.device).train()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_start = time.time()
    num_batches = len(train_loader)
    for epoch in range(flags['num_epochs']):
        # xm.master_print('Here there may be an error with config.device')
        para_train_loader = pl.ParallelLoader(train_loader, [config.device]).per_device_loader(config.device)
        total_loss = 0
        for i, (inputs, target, labels) in enumerate(para_train_loader):
            inputs = inputs.to(config.device)
            target = target.to(config.device)
            labels = labels.to(config.device)
                
            yhat = model(inputs.float())
            # Dummy acc, as it is not needed for training
            correct = 0
        
            loss = loss_fn(yhat, target.float().squeeze())
            total_loss += loss
        
            optimizer.zero_grad()
        
            loss.backward()
            xm.optimizer_step(optimizer)
        
        total_loss /= num_batches
        xm.master_print(f'Epoch {epoch+1} \tLoss: {total_loss:.2f}')

    elapsed_train_time = time.time() - train_start
    print("Process", index, "finished training. Train time was:", elapsed_train_time) 


if __name__ == '__main__':
    # Configures training (and evaluation) parameters
    flags = {}
    flags['batch_size'] = 32
    flags['num_workers'] = 8
    flags['num_epochs'] = 1
    flags['seed'] = 1234

    xmp.spawn(_map_fn, args=(flags,), nprocs=1)

"""
    ###### Old portion ######
    trainer = TPUTrainer(crsp, params, loss_fn, methodology='normal', l1_reg=False, nni_experiment=False, train_window_years=config.n_train_years, val_window_years=config.n_val_years)
    n_inputs = trainer.n_inputs

    model = Flexible_Sequential_Net(n_inputs).to(config.device)

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

    print('Starting Training process of the network')
    trainer.fit(model, optimizer)

    print('Time to run the entire training')
    print("--- %s seconds ---" % (time.time() - start_time))
    """

