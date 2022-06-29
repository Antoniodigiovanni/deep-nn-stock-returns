from time import strftime
from tkinter.messagebox import NO
import torch
import torch.nn as nn
import numpy as np
from models.neural_net.model import RegressionNet
import models.neural_net.metric as module_metric
from data.custom_dataset import CustomDataset
from data.data_preprocessing import *
from torch.utils.data import DataLoader
import torch.optim as optim
from trainer.trainer import *
import data.data_preprocessing as dp
import sys,os
import config
from pandas.tseries.offsets import DateOffset
import datetime as dt


class ExpandingWindowTraining():
    def __init__(self, dataset, train_window_years=10, val_window_years=1) -> None:
        self.dataset = dataset
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.device = config.device
        self.best_val_loss = np.inf
        self.patience = 3
        # Add patience?

        self.train_starting_year = dataset.yyyymm.min()
        self.df_end_date = dataset.yyyymm.max()
        self.train_window = train_window_years
        self.val_window = val_window_years
        self.val_starting_year = int((dt.datetime.strptime(
            str(self.train_starting_year), '%Y%m') + 
            DateOffset(years=self.train_window)).strftime('%Y%m'))
        
        # Subtract one month from the start of the validation period to
        # get the end of the training period
        self.end_train = int((dt.datetime.strptime(
            str(self.val_starting_year), '%Y%m') - 
            DateOffset(months=1)).strftime('%Y%m'))
        
        self.train_dates = None
        self.val_dates = None

        self.train, self.val = None, None
        self.train_loader, self.val_loader = None, None

        self.__generate_training_window()
        self.__subset_df()


    def fit(self):

        while self.val_dates[-1] <= self.df_end_date:
            # Aggiusta parametri per fare esperimenti nni
            self.model = RegressionNet(self.n_inputs).to(config.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.loss_fn = torch.nn.MSELoss()
            print(f'\n\nTraining from {self.train_dates[0]} to {self.train_dates[-1]}')
            print(f'Validating from {self.val_dates[0]} to {self.val_dates[-1]}\n\n')
            
            # Reset parameters for early stopping 
            j = 0
            self.best_val_loss = np.inf

            for epoch in range(config.args.epochs):
                
                epoch_loss = self.__process_one_epoch('train')
                val_loss, val_acc = self.__process_one_epoch('val')

                # Early stopping logic:
                if val_loss < self.best_val_loss:
                    j = 0
                    self.best_val_loss = val_loss
                    print(f'j set back to 0, best val_loss is: {self.best_val_loss}')
                else:
                    j+=1
                    print(f'j incremented to {j}!')
                if epoch%config.args.ep_log_interval == 0:
                    print(f'Epoch n. {epoch+1} [of #{config.args.epochs}]')
                    print(f'Training loss: {epoch_loss} | Val loss: {val_loss}')
                    print(f'Validation accuracy: {val_acc}%')
                if j >= self.patience:
                    print(f'Early stopping at epoch {epoch+1}!')
                    break
            self.__update_years()

    def __process_one_epoch(self, mode='train'):
        """
            If mode is set to 'val', the function will perform
            validation on the val set.
        """
        if mode == 'train':
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            loader = self.val_loader
            
        total_acc = 0
        total_loss = 0

        for _, data in enumerate(loader):
            #print(data['X'].shape)
            #print(data['Y'].shape)
            loss, acc = self.__process_one_step(data, mode)
            total_loss += loss
            total_acc += acc

        total_loss = total_loss / loader.batch_size
        total_acc = total_acc / loader.batch_size * 100
        
        if mode == 'train':
            return total_loss
        else:
            return total_loss, total_acc
    
    def __process_one_step(self, data, mode):
        if mode == 'train':
            self.optimizer.zero_grad()
        
        for k,v in data.items():
            data[k] = v.to(self.device)

        if mode == 'train':
            yhat = self.model(data['X'])
            # Dummy acc, as it is not needed for training
            acc = 0
        elif mode == 'val':
            with torch.no_grad():
                yhat = self.model(data['X'])
            acc = metric.accuracy(yhat, data['Y'], 0.1)
        
        loss = self.loss_fn(yhat.ravel(), data['Y'].ravel())
        
        if mode == 'train':
            loss.backward()
            self.optimizer.step()
        return loss, acc


    def __update_years(self):

        self.val_starting_year = int((dt.datetime.strptime(
            str(self.val_starting_year), '%Y%m') + DateOffset(
                years=1
            )).strftime('%Y%m'))
        
        self.end_train = int((dt.datetime.strptime(
            str(self.val_starting_year), '%Y%m') - 
            DateOffset(months=1)).strftime('%Y%m'))

        # Update the yyyymm list
        self.__generate_training_window()
        self.__subset_df()       

    def __generate_training_window(self):
        self.train_dates = list(
            pd.period_range(
                start = dt.datetime.strptime(
                    str(self.train_starting_year), '%Y%m'),
                end=dt.datetime.strptime(str(self.end_train), '%Y%m'),
                freq='M')
                .strftime('%Y%m')
                .astype(int)
                )

        self.val_dates = list(
            pd.period_range(
                start = dt.datetime.strptime(
                    str(self.val_starting_year), '%Y%m'),
                periods=(12*self.val_window),
                freq='M')
                .strftime('%Y%m')
                .astype(int)
                )

    def __subset_df(self):
        train = self.dataset.loc[self.dataset['yyyymm'].isin(self.train_dates)].copy()
        validation = self.dataset.loc[self.dataset['yyyymm'].isin(self.val_dates)].copy()

        X_train, y_train = dp.sep_target(train)
        X_val, y_val = dp.sep_target(validation)

        self.train = CustomDataset(X_train, y_train)
        self.val = CustomDataset(X_val, y_val)

        self.n_inputs = self.train.data.shape[1]
        # Modify batch size to make it trainable with higher values
        # Modify dataloader args
        self.train_loader = DataLoader(self.train, batch_size=1000)
        self.val_loader = DataLoader(self.val, batch_size = config.args.batch_size_validation)


    def __report_to_nni(self, experiment='False'):
        """
            Function idea to make the training agnostic to nni experiments
            meaning that it would work in the same way without using nni.
            I could use the arg in config to know if the train is an nni experiment or not
        """
        pass

from data.base_dataset import BaseDataset
from models.neural_net.model import RegressionNet
data = BaseDataset()
data.load_dataset_in_memory()
dataset = data.crsp


net = ExpandingWindowTraining(dataset)
net.fit()

"""# Categorical_cols should not be only in the following variable, as the encoding should be done in general
if os.path.exists(config.paths['ProcessedDataPath'] + '/dataset.csv') == False or config.ForcePreProcessing == True:
    print('Pre-processing data...')
   
    crsp, categorical_cols = dp.prepare_data(
        config.paths['CRSPretPath'],
        config.paths['CRSPinfoPath'],
        config.paths['FFPath'],
        config.paths['SignalsPath'],
        config.paths['ProcessedDataPath']
    )

    print(f'Pre-processing complete, the dataset has dimensions: {crsp.shape}')
    
else:
    print('Datased found in memory, loading...')
    crsp = pd.read_csv(config.paths['ProcessedDataPath']+'/dataset.csv', index_col=0)
    


# Abstract
if os.path.exists(config.currentPath +'/../../saved/models/RegressionNet_model.pt') and config.ForceTraining == False:
    print('Saved model found, breaking execution')
    sys.exit()


train, val = split_data_train_val(crsp)
X_train, Y_train = sep_target(train)
X_val, Y_val = sep_target(val)

del[train, val]

train = CustomDataset(X_train, Y_train)
del [X_train, Y_train]

val = CustomDataset(X_val, Y_val)
del[X_val, Y_val]


del crsp

# Should shuffle be set to true in my case? Check documentation
train_loader = DataLoader(train, batch_size=config.batch_size_train, num_workers=2)
val_loader = DataLoader(val, batch_size=config.batch_size_validation, num_workers=2)



n_inputs = train.data.shape[1]
model = RegressionNet(n_inputs).to(config.device)

optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

trainer = NeuralNetTrainer(model, train_loader, val_loader, optimizer).train()"""

""" tb = SummaryWriter()

for epoch in range(config.n_epochs):
    epoch_loss = train_one_epoch(model, train_loader, optimizer, config.loss_fn, config.device)
    validation_loss = validate_one_epoch(model, val_loader, optimizer, config.loss_fn, config.device)  
  
    if epoch % config.ep_log_interval == 0:
        print(f'Epoch n.{epoch} | Loss: {epoch_loss}')
        print(f'Validation Loss: {validation_loss}')

    tb.add_scalar("Training Loss", epoch_loss, epoch)
    tb.add_scalar("Validation Loss", validation_loss, epoch)
    tb.flush()
print(f'\nFinal Training Loss: {epoch_loss} at epoch n. {epoch+1}')
print(f'\nFinal Validation Loss: {validation_loss} at epoch n. {epoch+1}')
# Save final model (entire model)
print("\nSaving trained model")
fn = config.paths['modelsPath']+'/RegressionNet_model.pt'
torch.save(model, fn)
 """
