import nni
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
import data.data_preprocessing as dp
import os
import json
from pandas.tseries.offsets import DateOffset
import datetime as dt
from torch.utils.tensorboard import SummaryWriter
from models.neural_net import metric


class GeneralizedTrainer():
    def __init__(self, dataset, params, loss_fn, methodology = 'normal', l1_reg = False, train_window_years=15, val_window_years=10) -> None:
        self.dataset = dataset
        self.params = params
        self.model = None
        self.optimizer = None
        self.loss_fn = loss_fn
        self.device = config.device
        self.best_val_loss = np.inf
        self.patience = self.params['patience'] # Add patience as a parameter could be an idea
        
        print(f'Epochs in Training class: {config.epochs}')
        self.methodology = methodology
        if self.methodology == 'expanding':
            print('Expanding window training')
        elif self.methodology == 'normal':
            print('Normal training')

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
        self.subset_df()

        # Tensorboard
        self.log_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard')
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def fit(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        
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

                    if j != 0:
                        pass
                        #print(f'j set back to 0, best val_loss is: {self.best_val_loss}')
                    j = 0
                    
                else:
                    j+=1
                    #print(f'j incremented to {j}!')
                # print(f'Epoch loss is of type: {type(epoch_loss)}, Validation loss is of type: {type(val_loss)},  Validation accuracy is of type: {type(val_acc)}')
                epoch_loss = epoch_loss.item()
                self.writer.add_scalar("Loss/train", float(epoch_loss), epoch)
                self.writer.add_scalar("Loss/validation", float(val_loss), epoch)
                self.writer.add_scalar("Loss/divergence", float(val_loss) - epoch_loss, epoch)
                self.writer.add_scalar("Accuracy/Validation", val_acc, epoch)


                results = {'default': float(val_loss), 'val_acc': val_acc}
                nni.report_intermediate_result(results)#(val_loss)
                
                if epoch%config.ep_log_interval == 0:
                    print(f'Epoch n. {epoch+1} [of #{config.epochs}]')
                    print(f'Training loss: {round(epoch_loss, 4)} | Val loss: {round(val_loss,4)}')
                    print(f'Validation accuracy: {round(100*val_acc,2)}%')
                
                if j >= self.patience:
                    print(f'Early stopping at epoch {epoch+1}!')
                    break
                
            pred_df = ReturnsPrediction(self.test_loader, self.model).pred_df
            self.prediction_df = pd.concat([self.prediction_df, pred_df], ignore_index=True)
            
            self.writer.flush()


            #To delete this part until next #
            print(f'Check for  prediction df, min:{self.prediction_df.yyyymm.min()}, max: {self.prediction_df.yyyymm.max()}\n shape:{self.prediction_df.shape}')
            print(f'\n Tail:\n {self.prediction_df.tail()}')
            #

            if self.methodology == 'expanding':
                self.__update_years()
                if self.test_dates[-1] > self.df_end_date:
                    keep_training = 0
                    print(f'The expanding window training is completed.')
            elif self.methodology == 'normal':
                keep_training = 0
                print('Normal training completed.')

        r2 = metric.r2_metric_calculation(self.prediction_df)
        print(r2)
        self.writer.add_scalar("R2/All", r2['R2'])
        self.writer.add_scalar("R2/Top1000", r2['R2_top_1000'])
        self.writer.add_scalar("R2/Bottom1000", r2['R2_bottom_1000'])
        
        self.writer.flush()    

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
        if os.path.exists(config.paths['hpoResultsPath'] + '/predicted_returns') == False:
            os.makedirs(config.paths['hpoResultsPath'] + '/predicted_returns')
        if os.path.exists(config.paths['hpoResultsPath'] + '/portfolio_returns') == False:
            os.makedirs(config.paths['hpoResultsPath'] + '/portfolio_returns')
        if os.path.exists(config.paths['hpoResultsPath'] + '/trial_info') == False:
            os.makedirs(config.paths['hpoResultsPath'] + '/trial_info')

        from csv import DictWriter, writer

        field_names = ['timeStamp','information_ratio','alpha', 'r2', 'val_loss']
        summary_dict = {'timeStamp': timeStamp_id, 'information_ratio': information_ratio, 'alpha': alpha, 'r2': r2, 'val_loss': val_loss}
        with open(config.paths['hpoResultsPath'] + '/experiment_summary.csv', 'a') as fp:          
            writer_obj = writer(fp)
            if fp.tell() == 0:
                writer_obj.writerow(field_names)
            dictwriter_object = DictWriter(fp, fieldnames=field_names)
            dictwriter_object.writerow(summary_dict)
  
        self.prediction_df.to_csv(config.paths['hpoResultsPath'] + '/predicted_returns/' + timeStamp_id + ' - predicted_returns.csv')
        returns.to_csv(config.paths['hpoResultsPath'] + '/portfolio_returns/' + timeStamp_id + ' - portfolio_returns.csv')
        final_dict = {'params': self.params, 'information_ratio': information_ratio, 'alpha':alpha, 'r2': r2}
        with open(config.paths['hpoResultsPath'] + '/trial_info/' + timeStamp_id + '- trial_full.json', 'w') as fp:
            json.dump(final_dict, fp, indent=4)
        

        print('Portfolio returns calculation completed.')
       
        # results = {
        #     'default': float(val_loss), 
        #     'val_acc': val_acc,
        #     'alpha': float(alpha),
        #     'information_ratio': float(information_ratio),
        #     'R2': r2['R2']}
        
        # nni.report_final_result(results) #(val_loss)

        results = {
            'default': float(val_loss), 
            'val_acc': val_acc,
            'alpha': float(alpha),
            'information_ratio': float(information_ratio),
            'R2': r2['R2']}

        print(r2)
        
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
        val_loss = 0

        size = len(loader.dataset)
        num_batches = len(loader)

        if mode == 'train':
            # print(f'num_batches = len(dataloader): {len(loader)}')
            # print(f'Size = len(dataloader.dataset):{len(loader.dataset)}')
            # print(f'Batch size: {loader.batch_size}')
                
            for batch, data in enumerate(loader):
                
                loss, correct = self.__process_one_step(data, mode)
                # if batch % 1 == 0:
                #     loss, current = loss.item(), batch * len(data['X'])
                #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                total_loss += loss
            total_loss /= num_batches
        
        # total_acc = total_acc / loader.batch_size * 100
        
        elif mode == 'val':
            for data in loader:
                loss, correct = self.__process_one_step(data, mode)
                val_loss += loss.item()
                total_correct += correct
            # print(f'Total correct in validation:{total_correct}')              
            val_acc = total_correct/size
            val_loss /= num_batches
            # print(f"Validation Error: \n Accuracy: {(100*val_acc):>0.1f}%, Avg loss: {val_loss:>8f} \n")

        if mode == 'train':
            return total_loss
        else:
            return val_loss, val_acc


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
                correct = metric.accuracy(data['Y'], yhat, 0.2)
        
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
