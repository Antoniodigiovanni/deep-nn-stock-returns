import config
import pandas as pd
import nni
import torch
import numpy as np
from data.crsp_dataset import CrspDataset
from torch.utils.data import DataLoader
from portfolios.ReturnsPrediction import ReturnsPrediction
from portfolios.portfolio_new import Portfolio
import os
import json
from pandas.tseries.offsets import DateOffset
import datetime as dt
from torch.utils.tensorboard import SummaryWriter
from models.neural_net import metric
from trainer.feature_importance import IntegratedGradients_importance
import time


class GeneralizedTrainer():
    def __init__(self, dataset, params, loss_fn, methodology = 'normal', l1_reg = False, train_window_years=15, val_window_years=10, nni_experiment=True) -> None:
        self.dataset = dataset
        self.params = params
        self.model = None
        self.optimizer = None
        self.loss_fn = loss_fn
        self.device = config.device
        self.best_val_loss = np.inf
        self.patience = 10
        print(f'Patience is {self.patience}')
        self.nni_experiment = nni_experiment

        # print(f'Epochs in Training class: {config.epochs}')
        self.methodology = methodology
        if self.methodology == 'expanding':
            print('Expanding window training')
        elif self.methodology == 'normal':
            print('Normal training')

        # L1 Regularization
        self.l1_reg = l1_reg
        if self.l1_reg:
            print('Change Line 48 in trainer when using a nested search space')
            self.l1_lambda = self.params['l1_lambda1']
            # self.l1_lambda = self.params['use_l1_reg']['lambda']
        
        
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
        

        self.train_dates, self.val_dates, self.test_dates = None, None, None

        self.train, self.val, self.test = None, None, None
        self.train_loader, self.val_loader, self.test_loader = None, None, None

        self.prediction_df = pd.DataFrame()

        self.__generate_training_window()
        self.subset_df()

        # Tensorboard
        if self.nni_experiment == True:
            self.log_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard')
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.writer = SummaryWriter()


    def fit(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        
        timeStamp = int(time.time()*10000000)
        print(f'Save Dir is: {config.saveDir}')
        # Prepare for saving trained model
        if os.path.exists(config.saveDir + '/models') == False:
            os.makedirs(config.saveDir + '/models')
        PATH = config.saveDir + '/models/model_'+str(timeStamp)+'.pt'
        print(f'PATH is: {PATH}')


        if (self.train_dates[-1] > self.df_end_date | self.val_dates[-1] > self.df_end_date | self.test_dates[-1] > self.df_end_date):
            print(self.train_dates[-1])
            print(self.val_dates[-1])
            print(self.test_dates[-1])
            print('Dates out of bounds, exiting training...')
            keep_training = 0
            import sys
            sys.exit()
        else:
            keep_training = 1


        while keep_training == 1:
            print(f'\n\nTraining from {self.train_dates[0]} to {self.train_dates[-1]}')
            print(f'Validating from {self.val_dates[0]} to {self.val_dates[-1]}')
            print(f'Testing from {self.test_dates[0]} to {self.test_dates[-1]}')
            
            # Reset parameters for early stopping 
            j = 0
            self.best_val_loss = np.inf
            
            for epoch in range(config.epochs):
                
                epoch_loss = self.__process_one_epoch('train')
                val_loss, val_acc = self.__process_one_epoch('val')

                # Early stopping logic:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch_loss': epoch_loss,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'params': self.params,
                        'n_inputs': self.n_inputs
                        }, PATH)
                    # print('Saved new best model')
                    # if j != 0:
                        #print(f'j set back to 0, best val_loss is: {self.best_val_loss}')
                    j = 0
                    
                else:
                    j+=1
                    # print(f'j incremented to {j}!')
                # print(f'Epoch loss is of type: {type(epoch_loss)}, Validation loss is of type: {type(val_loss)},  Validation accuracy is of type: {type(val_acc)}')
                epoch_loss = epoch_loss.item()


                if epoch%config.ep_log_interval == 0:
                    print(f'Epoch n. {epoch+1} [of #{config.epochs}]')
                    print(f'Training loss: {round(epoch_loss, 4)} | Val loss: {round(val_loss,4)}')
                    print(f'Validation accuracy: {round(100*val_acc,2)}%')
                
                if j >= self.patience:
                    print(f'Early stopping at epoch {epoch+1}!')
                    

                    checkpoint = torch.load(PATH)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    epoch = checkpoint['epoch']
                    print(f'Loading last good epoch checkpoint, epoch #{epoch+1}')
                    epoch_loss = checkpoint['epoch_loss']
                    val_loss = checkpoint['val_loss']
                    val_acc = checkpoint['val_acc']
                    self.model.train()

                    break
                
        
            if self.methodology == 'expanding':
                self.__update_years()
                if self.test_dates[-1] > self.df_end_date:
                    keep_training = 0
                    print(f'The expanding window training is completed.')
            elif self.methodology == 'normal':
                keep_training = 0
                print('Normal training completed.')
            
        
    def __process_one_epoch(self, mode='train'):
        """,
            If mode is set to 'val', the function will perform
            validation on the val set.
        """
        if mode == 'train':
            self.model.train()
            size = 5
            num_batches = 1
        else:
            self.model.eval()
            size = 5
            num_batches = 1
            
        total_correct = 0
        total_loss = 0
        val_loss = 0

        if mode == 'train': 
           loss, correct = self.__process_one_step(self.train_inputs, self.train_target, self.train_labels, mode)
           total_loss += loss
           total_loss /= num_batches
        
        # total_acc = total_acc / loader.batch_size * 100
        
        elif mode == 'val':
            loss, correct = self.__process_one_step(self.val_inputs, self.val_target, self.val_labels, mode)
            val_loss += loss.item()
            total_correct += correct
            
            val_acc = total_correct/size
            val_loss /= num_batches
            
        if mode == 'train':
            return total_loss
        else:
            return val_loss, val_acc


    def __process_one_step(self, inputs, target, labels, mode):
        if mode == 'train':
            self.optimizer.zero_grad()
        
        inputs = inputs.to(config.device)
        target = target.to(config.device)
        labels = labels.to(config.device)
                
        if mode == 'train':
            yhat = self.model(inputs.float())
            # Dummy acc, as it is not needed for training
            correct = 0
        elif mode == 'val':
            with torch.no_grad():
                yhat = self.model(inputs.float())
                correct = metric.accuracy(target.float().squeeze(), yhat, 0.2)
        
        loss = self.loss_fn(yhat, target.float().squeeze())
        
        if self.l1_reg:
            l1_norm = sum(p.abs().sum()
                for p in self.model.parameters())
            loss = loss.squeeze() + l1_norm.squeeze() * self.l1_lambda


        if mode == 'train':
            loss.backward()
            self.optimizer.step()

        return loss, correct


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
            self.subset_df()       


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
        if self.methodology == 'normal':
            self.test_dates = list(
                pd.period_range(
                    start= (dt.datetime.strptime(str(self.val_dates[-1]), '%Y%m') + DateOffset(months=1)),
                    end=dt.datetime.strptime(str(self.df_end_date), '%Y%m'),
                    freq='M'
                    )
                    .strftime('%Y%m')
                    .astype(int)
                    )
        elif self.methodology == 'expanding':
              self.test_dates = list(
            pd.period_range(
                start= (dt.datetime.strptime(str(self.val_dates[-1]), '%Y%m') + DateOffset(months=1)),
                periods=12,
                freq='M'
                )
                .strftime('%Y%m')
                .astype(int)
                )
        

    def subset_df(self):
        train = self.dataset.loc[self.dataset['yyyymm'].isin(self.train_dates)].copy()
        validation = self.dataset.loc[self.dataset['yyyymm'].isin(self.val_dates)].copy()
        test = self.dataset.loc[self.dataset['yyyymm'].isin(self.test_dates)].copy()
        
        self.train = CrspDataset(train)
        self.validation = CrspDataset(validation)
        self.test = CrspDataset(test)

        print('Train/Val/Test split updated')
        
        if len(self.test) > 0:
            bs = len(self.test)
        else:
            bs = 10000 
        self.n_inputs = self.train.get_inputs()

        self.train_loader = DataLoader(self.train, batch_size=5, shuffle=True)
        
        iterLoader = iter(self.train_loader)
        self.train_inputs, self.train_target, self.train_labels = iterLoader.next()
        
        self.val_loader = DataLoader(self.validation, batch_size=5, shuffle=True)
        
        iterLoader = iter(self.val_loader)
        self.val_inputs, self.val_target, self.val_labels = iterLoader.next()

    
        self.test_loader = DataLoader(self.test, batch_size=bs)

        iterLoader = iter(self.test_loader)
        self.test_inputs, self.test_target, self.test_labels = iterLoader.next()

        print('Train:')
        df = pd.DataFrame({'yyyymm': self.train_labels[:,0].numpy(), 'permno': self.train_labels[:,1].numpy(), 'ret': self.train_target[:,0].numpy(), 'feature1': self.train_inputs[:,0].numpy(), 'feature2': self.train_inputs[:,1].numpy()})
        print(df)
        print(df.columns)
        del self.test_loader, self.train_loader, self.val_loader