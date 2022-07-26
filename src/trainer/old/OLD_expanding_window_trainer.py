import torch
import torch.nn as nn
import numpy as np
from models.neural_net.gu_et_al_NN4 import GuNN4
from data.custom_dataset import CustomDataset, TestDataset
from data.data_preprocessing import *
from torch.utils.data import DataLoader
import torch.optim as optim
from portfolios.ReturnsPrediction import ReturnsPrediction
from portfolios.PortfolioCreation import Portfolio
from trainer.trainer import *
import data.data_preprocessing as dp
import sys,os
from pandas.tseries.offsets import DateOffset
import datetime as dt


class ExpandingWindowTraining():
    def __init__(self, dataset, params, loss_fn, l1_reg=False, train_window_years=15, val_window_years=10) -> None:
        self.dataset = dataset
        self.params = params
        self.model = None
        self.optimizer = None
        self.loss_fn = loss_fn
        self.device = config.device
        self.best_val_loss = np.inf
        self.patience = self.params['patience'] 
        
        # L1 Regularization
        self.l1_reg = l1_reg
        if self.l1_reg:
            self.l1_lambda = self.params['l1_lambda1']
        
        
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
        self.__subset_df()


    def fit(self, model, optimizer):
        # Trying with moving the model instantiation outside of the loop, so every time training is repeated we are not 
        # re-instantiating the model.
        self.model = model
        self.optimizer = optimizer

        while self.test_dates[-1] <= self.df_end_date:
            
            print(f'\n\nTraining from {self.train_dates[0]} to {self.train_dates[-1]}')
            print(f'Validating from {self.val_dates[0]} to {self.val_dates[-1]}')
            print(f'Testing from {self.test_dates[0]} to {self.test_dates[-1]}')
            
            # Reset parameters for early stopping 
            j = 0
            self.best_val_loss = np.inf
            
            for epoch in range(config.args.epochs):
                
                epoch_loss = self.__process_one_epoch('train')
                val_loss, val_acc = self.__process_one_epoch('val')

                # Early stopping logic:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

                    if j != 0:
                        pass
                        #print(f'j set back to 0, best val_loss is: {self.best_val_loss}')
                    j = 0
                    
                else:
                    j+=1
                    #print(f'j incremented to {j}!')

                results = {'default': float(val_loss), 'val_acc': val_acc}
                nni.report_intermediate_result(results)#(val_loss)
            
                if epoch%config.args.ep_log_interval == 0:
                    print(f'Epoch n. {epoch+1} [of #{config.args.epochs}]')
                    print(f'Training loss: {epoch_loss} | Val loss: {val_loss}')
                    print(f'Validation accuracy: {val_acc}%')
                
                if j >= self.patience:
                    print(f'Early stopping at epoch {epoch+1}!')
                    break
                
            pred_df = ReturnsPrediction(self.test_loader, self.model).pred_df
            self.prediction_df = pd.concat([self.prediction_df, pred_df], ignore_index=True)
            
            #To delete this part until next #
            print(f'Check for  prediction df, min:{self.prediction_df.yyyymm.min()}, max: {self.prediction_df.yyyymm.max()}\n shape:{self.prediction_df.shape}')
            print(f'\n Tail:\n {self.prediction_df.tail()}')
            #
            self.__update_years()

        print(f'The expanding window training is completed.')

                
        timeStamp_id = dt.datetime.now().strftime('%Y%m%d-%H_%M:%S:%f')
        # with open(timeStamp_id + '- trial.json', 'w') as fp:
        #     json.dump(dict, fp)

        print('Calculating portfolios')
        print('Prediction df:')
        print(self.prediction_df.describe())

        self.prediction_df.sort_values(['permno','yyyymm'], inplace=True)
        print('NaNs in prediction_df:')
        count = self.prediction_df.isna().sum()
        percentage = self.prediction_df.isna().mean()
        null_values = pd.concat([count, percentage], axis=1, keys=['count', '%'])
        print(null_values)



        portfolio = Portfolio(pred_df=self.prediction_df)
        information_ratio = portfolio.information_ratio
        alpha = portfolio.alpha
        returns = portfolio.returns
        
        
        print('\nNaNs in Portfolio returns:')
        count = returns.isna().sum()
        percentage = returns.isna().mean()
        null_values = pd.concat([count, percentage], axis=1, keys=['count', '%'])
        print(null_values)
        print(f'IR: {information_ratio}\nAlpha: {alpha}')
        print('Portfolio Returns:')
        print(returns.describe())
        

        # Saving files
        if os.path.exists(config.paths['guTuningResultsPath'] + '/predicted_returns') == False:
            os.makedirs(config.paths['guTuningResultsPath'] + '/predicted_returns')
        if os.path.exists(config.paths['guTuningResultsPath'] + '/portfolio_returns') == False:
            os.makedirs(config.paths['guTuningResultsPath'] + '/portfolio_returns')
        if os.path.exists(config.paths['guTuningResultsPath'] + '/trial_info') == False:
            os.makedirs(config.paths['guTuningResultsPath'] + '/trial_info')

        from csv import DictWriter, writer

        field_names = ['timeStamp','information_ratio','alpha']
        summary_dict = {'timeStamp': timeStamp_id, 'information_ratio': information_ratio, 'alpha': alpha}
        with open(config.paths['guTuningResultsPath'] + '/experiment_summary.csv', 'a') as fp:          
            writer_obj = writer(fp)
            if fp.tell() == 0:
                writer_obj.writerow(field_names)
            dictwriter_object = DictWriter(fp, fieldnames=field_names)
            dictwriter_object.writerow(summary_dict)
  
        self.prediction_df.to_csv(config.paths['guTuningResultsPath'] + '/predicted_returns/' + timeStamp_id + ' - predicted_returns.csv')
        returns.to_csv(config.paths['guTuningResultsPath'] + '/portfolio_returns/' + timeStamp_id + ' - portfolio_returns.csv')
        final_dict = {'params': self.params, 'information_ratio': information_ratio, 'alpha':alpha}
        with open(config.paths['guTuningResultsPath'] + '/trial_info/' + timeStamp_id + '- trial_full.json', 'w') as fp:
            json.dump(final_dict, fp, indent=4)
        
        # Old version
        # returns.to_csv(config.paths['resultsPath'] + '/' + timeStamp_id + ' - portfolio_returns.csv')
        # final_dict = {'params': self.params, 'information_ratio': information_ratio, 'alpha':alpha}
        # with open(config.paths['resultsPath'] + '/' + timeStamp_id + '- trial_full.json', 'w') as fp:
        #     json.dump(final_dict, fp)

        print('Portfolio returns calculation completed.')
       
        results = {
            'default': float(val_loss), 
            'val_acc': val_acc,
            'alpha': float(alpha),
            'information_ratio': float(information_ratio)}
        
        nni.report_final_result(results) #(val_loss)



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
            
        total_correct = 0
        total_loss = 0
        total = 0

        for batch_idx, data in enumerate(loader):
            #print(data['X'].shape)
            #print(data['Y'].shape)
            loss, correct = self.__process_one_step(data, mode)
            total += data['X'].size(0)
            total_loss += loss.item()
            total_correct += correct
            

        total_loss /= batch_idx
        total_acc = 100.*correct/total

        
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
            correct = 0
        elif mode == 'val':
            with torch.no_grad():
                yhat = self.model(data['X'])
            correct = metric.accuracy(data['Y'], yhat, 0.1)
        
        loss = self.loss_fn(yhat.ravel(), data['Y'].ravel())
        
        if self.l1_reg:
            l1_norm = sum(p.abs().sum()
                for p in self.model.parameters())

            loss = loss + l1_norm * self.l1_lambda


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

        self.test_dates = list(
            pd.period_range(
                start= (dt.datetime.strptime(str(self.val_dates[-1]), '%Y%m') + DateOffset(months=1)),
                periods=12,
                freq='M'
                )
                .strftime('%Y%m')
                .astype(int)
                )

    def __subset_df(self):
        train = self.dataset.loc[self.dataset['yyyymm'].isin(self.train_dates)].copy()
        validation = self.dataset.loc[self.dataset['yyyymm'].isin(self.val_dates)].copy()
        test = self.dataset.loc[self.dataset['yyyymm'].isin(self.test_dates)].copy()

        X_train, y_train = dp.sep_target(train)
        X_val, y_val = dp.sep_target(validation)
        X_test, y_test, test_index = dp.sep_target_idx(test)

        self.train = CustomDataset(X_train, y_train)
        self.val = CustomDataset(X_val, y_val)
        self.test = TestDataset(X_test, y_test, test_index)

        self.n_inputs = self.train.data.shape[1]
        # Modify batch size to make it trainable with higher values
        # Modify dataloader args
        self.train_loader = DataLoader(self.train, batch_size=10000)
        self.val_loader = DataLoader(self.val, batch_size=10000)
        self.test_loader = DataLoader(self.test, batch_size=10000)


    def __report_to_nni(self, experiment='False'):
        """
            Function idea to make the training agnostic to nni experiments
            meaning that it would work in the same way without using nni.
            I could use the arg in config to know if the train is an nni experiment or not
        """
        pass