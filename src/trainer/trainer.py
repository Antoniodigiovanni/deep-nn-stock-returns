from tabnanny import check
import nni
import torch
import torch.nn as nn
import numpy as np
from models.neural_net.gu_et_al_NN4 import GuNN4
from data.custom_dataset import CrspDataset, CustomDataset, TestDataset
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
        
        # Prepare for saving trained model
        if os.path.exists(config.saveDir + '/models') == False:
            os.makedirs(config.saveDir + '/models')
        PATH = config.saveDir + '/models/model_'+str(timeStamp)+'.pt'
        
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
                        'val_acc': val_acc
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
                self.writer.add_scalar("Loss/train", float(epoch_loss), epoch)
                self.writer.add_scalar("Loss/validation", float(val_loss), epoch)
                self.writer.add_scalar("Loss/divergence", float(val_loss) - epoch_loss, epoch)
                self.writer.add_scalar("Accuracy/Validation", val_acc, epoch)


                results = {'default': float(val_loss), 'val_acc': val_acc}
                if self.nni_experiment == True:
                    nni.report_intermediate_result(results)

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

        timeStamp_id = dt.datetime.now().strftime('%Y%m%d-%H:%M:%S:%f')
        
        print('Calculating portfolios')
        print('Prediction df:')
        print(self.prediction_df.describe())

        self.prediction_df.sort_values(['permno','yyyymm'], inplace=True)
        print('NaNs in prediction_df:')
        count = self.prediction_df.isna().sum()
        percentage = self.prediction_df.isna().mean()
        null_values = pd.concat([count, percentage], axis=1, keys=['count', '%'])
        print(null_values)

        print(f'Rebalancing is: {config.rebalancing_frequency}')
        portfolio = Portfolio(pred_df=self.prediction_df, n_cuts=config.n_cuts, rebalancing_frequency=config.rebalancing_frequency, weighting = config.weighting)
        information_ratio = portfolio.information_ratio
        alpha = portfolio.alpha
        returns = portfolio.returns
        portfolio_weights = portfolio.portfolio_weights

        

        print('\nNaNs in Portfolio returns:')
        count = returns.isna().sum()
        percentage = returns.isna().mean()
        null_values = pd.concat([count, percentage], axis=1, keys=['count', '%'])
        print(null_values)
        print(f'IR: {information_ratio}\nAlpha: {alpha}')
        print('Portfolio Returns:')
        print(returns.describe())
        
        
        sharpe_ratio = metric.calc_sharpe_ratio(returns)
        print(f'Sharpe Ratio of {config.rebalancing_frequency} {config.weighting} portfolio:')
        print(sharpe_ratio)

        # Modify folder structure for when not doing the nni_experiment.
        predictedRetDir = config.saveDir + '/predicted_returns'
        portfolioRetDir = config.saveDir + '/portfolio_returns'
        trialInfoDir = config.saveDir + '/trial_info'
        portfolioWeightsDir = config.saveDir + '/portfolio_weights'

        # Saving files
        """
        if os.path.exists(config.saveDir + '/predicted_returns') == False:
            os.makedirs(config.saveDir + '/predicted_returns')
        if os.path.exists(config.saveDir + '/portfolio_returns') == False:
            os.makedirs(config.saveDir + '/portfolio_returns')
        if os.path.exists(config.saveDir + '/trial_info') == False:
            os.makedirs(config.saveDir + '/trial_info')
        if os.path.exists(config.saveDir + '/portfolio_weights') == False:
            os.makedirs(config.saveDir + '/portfolio_weights')
        # if os.path.exists(config.saveDir + '/models') == False:
        #     os.makedirs(config.saveDir + '/models')
        """
        if os.path.exists(predictedRetDir) == False:
            os.makedirs(predictedRetDir)
        if os.path.exists(portfolioRetDir) == False:
            os.makedirs(portfolioRetDir)
        if os.path.exists(trialInfoDir) == False:
            os.makedirs(trialInfoDir)
        if os.path.exists(portfolioWeightsDir) == False:
            os.makedirs(portfolioWeightsDir)
        

        from csv import DictWriter, writer

        field_names = ['timeStamp','information_ratio','alpha', 'r2', 'val_loss', 'Sharpe_Ratio']
        summary_dict = {
            # 'timeStamp': timeStamp_id,
            'timeStamp': timeStamp,
            'information_ratio': information_ratio, 
            'alpha': alpha, 
            'r2': r2, 
            'val_loss': val_loss, 
            # 'Sharpe_Ratio': sharpe_ratio
            }

        with open(config.saveDir + '/experiment_summary.csv', 'a') as fp:          
            writer_obj = writer(fp)
            if fp.tell() == 0:
                writer_obj.writerow(field_names)
            dictwriter_object = DictWriter(fp, fieldnames=field_names)
            dictwriter_object.writerow(summary_dict)
  
        self.prediction_df.to_csv(predictedRetDir + '/' + timeStamp + '_predicted_returns.csv')
        returns.to_csv(portfolioRetDir + '/' + timeStamp + '_portfolio_returns.csv')
        
        final_dict = {
            'params': self.params,
            'information_ratio': information_ratio,
            'alpha':alpha, 
            'r2': r2,
            'Others': 'Think about adding other metrics (SR,Max_DD,...)'}

        with open(trialInfoDir + '/' + timeStamp + '_trial_full.json', 'w') as fp:
            json.dump(final_dict, fp, indent=4)
        
        portfolio_weights.to_csv(portfolioWeightsDir + '/' + timeStamp + '_portfolio_weights.csv')

        # Saving model
        # torch.save(self.model, config.paths['hpoResultsPath'] + '/models/' + timeStamp_id + ' - model.pt')
        # torch.save(self.model.state_dict(), config.paths['hpoResultsPath'] + '/models/' + timeStamp_id + ' - model_state_dict.pt')

        print('Portfolio returns calculation completed.')
       
        results = {
            'default': float(val_loss), 
            'val_acc': val_acc,
            'alpha': float(alpha),
            'information_ratio': float(information_ratio),
            'R2': r2['R2']}

        print(r2)
        os.remove(PATH)
        if self.nni_experiment == True:
            nni.report_final_result(results)

 

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
                
            for i, (inputs, target, labels) in enumerate(loader):
                
                loss, correct = self.__process_one_step(inputs, target, labels, mode)
                # if batch % 1 == 0:
                #     loss, current = loss.item(), batch * len(data['X'])
                #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                total_loss += loss
            total_loss /= num_batches
        
        # total_acc = total_acc / loader.batch_size * 100
        
        elif mode == 'val':
            for i, (inputs, target, labels) in enumerate(loader):
                loss, correct = self.__process_one_step(inputs, target, labels, mode)
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
        
        if len(self.test) > 0:
            bs = len(self.test)
        else:
            bs = 10000 
        self.n_inputs = self.train.get_inputs()
        # Modify batch size to make it trainable with higher values
        # Modify dataloader args
        self.train_loader = DataLoader(self.train, batch_size=10000)
        self.val_loader = DataLoader(self.validation, batch_size=10000)
        self.test_loader = DataLoader(self.test, batch_size=bs)