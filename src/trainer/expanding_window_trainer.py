import torch
import torch.nn as nn
import numpy as np
from models.neural_net.gu_et_al_NN4 import GuNN4
from data.custom_dataset import CustomDataset
from data.data_preprocessing import *
from torch.utils.data import DataLoader
import torch.optim as optim
from src.portfolios.ReturnsPrediction import ReturnsPrediction
from trainer.trainer import *
import data.data_preprocessing as dp
import sys,os
import config
from pandas.tseries.offsets import DateOffset
import datetime as dt


class ExpandingWindowTraining():
    def __init__(self, dataset, params, train_window_years=15, val_window_years=10) -> None:
        self.dataset = dataset
        self.params = params
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.device = config.device
        self.best_val_loss = np.inf
        self.patience = self.params['patience'] # Add patience as a parameter could be an idea
        
        # L1 Regularization
        self.l1_reg = True
        self.l1_lambda = self.params['l1_lambda1']
        
        # Prediction step to be implemented - #TODO
        if config.args.predict:
            self.prediction = True
        else:
            self.prediction = False

        self.train_starting_year = dataset.yyyymm.min()
        self.df_end_date = dataset.yyyymm.max()
        self.train_window = train_window_years
        self.val_window = val_window_years
        self.val_starting_year = int((dt.datetime.strptime(
            str(self.train_starting_year), '%Y%m') + 
            DateOffset(years=self.train_window)).strftime('%Y%m'))
        
        # Subtracting one month from the start of the validation period
        # gives us the end of the training period
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
            self.model = GuNN4(self.n_inputs).to(config.device)
            
            self.optimizer = optim.Adam(self.model.parameters(),
                self.params['learning_rate'],
                betas=(
                    self.params['adam_beta_1'], 
                    self.params['adam_beta_2'])
                    )

            self.loss_fn = nn.L1Loss()
            
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
                # I could maybe report validation loss
                nni.report_intermediate_result(val_acc)
                if epoch%config.args.ep_log_interval == 0:
                    print(f'Epoch n. {epoch+1} [of #{config.args.epochs}]')
                    print(f'Training loss: {epoch_loss} | Val loss: {val_loss}')
                    print(f'Validation accuracy: {val_acc}%')
                if j >= self.patience:
                    print(f'Early stopping at epoch {epoch+1}!')
                    break
            prediction_df = ReturnsPrediction(self.test_loader, self.model)
            print(prediction_df.shape)
            print(prediction_df.head())
            self.__update_years()

        print(f'The expanding window training is completed,\
                the last validation accuracy for\
                this trial is {val_acc} - maybe print out avg or best.')

        nni.report_final_result( val_acc)

    def __process_one_epoch(self, mode='train'):
        """,
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
            acc = metric.accuracy(data['Y'], yhat, 0.1)
        
        loss = self.loss_fn(yhat.ravel(), data['Y'].ravel())
        
        if self.l1_reg:
            l1_norm = sum(p.abs().sum()
                for p in self.model.parameters())

            loss = loss + l1_norm * self.l1_lambda


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
        self.train_loader = DataLoader(self.train, batch_size=10000)
        self.val_loader = DataLoader(self.val, batch_size = 10000)


    def __report_to_nni(self, experiment='False'):
        """
            Function idea to make the training agnostic to nni experiments
            meaning that it would work in the same way without using nni.
            I could use the arg in config to know if the train is an nni experiment or not
        """
        pass