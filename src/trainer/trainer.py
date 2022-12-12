import config
import pandas as pd
import nni
import torch
import torch.nn as nn
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
    def __init__(self, dataset, params, loss_fn, methodology = 'normal', l1_reg = False, l2_reg = False, train_window_years=15, val_window_years=10, epochs = 100, nni_experiment=True) -> None:
        self.dataset = dataset
        self.params = params
        self.model = None
        self.optimizer = None
        self.loss_fn = loss_fn
        self.device = config.device
        self.best_val_loss = np.inf
        if 'patience' in self.params:
            self.patience = self.params['patience']
        else:
            self.patience = 10
        print(f'Patience is {self.patience}')
        self.nni_experiment = nni_experiment
        self.epochs = epochs

        self.methodology = methodology
        if self.methodology == 'expanding':
            print('Expanding window training')
        elif self.methodology == 'normal':
            print('Normal training')

        # L1 Regularization
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        if self.l1_reg:
            if 'l1_lambda1' in self.params:
                self.l1_lambda = self.params['l1_lambda1']
            elif 'L1' in self.params:
                self.l1_lambda = self.params['L1']['lambda']

        if self.l2_reg:
            if 'l2_lambda' in self.params:
                self.l2_lambda = self.params['l2_lambda']
            elif 'L2' in self.params:
                self.l2_lambda = self.params['L2']['lambda']
      
        
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

        # Saving non-trained model to re-load it every iteration
        # initial_path = config.saveDir + '/models/model_'+str(timeStamp)+'_not-trained.pt'
        # print(f'Initial model saved at: {initial_path}')
        # torch.save({
        #                 'model_state_dict': self.model.state_dict(),
        #                 'optimizer_state_dict': self.optimizer.state_dict(),
        #                 }, initial_path)

        iteration_r2 = []
        cum_epoch = 0
        val_iteration_losses = []
        iteration_val_spearman = []
        iteration_train_spearman = []

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

        # self.best_global_spearman = -np.inf #To delete when going back to normal epochs
        # self.best_val_spearman = -np.inf
        # self.best_val_r2 = -np.inf    
        while keep_training == 1:
            # To re-initialize model weights each iteration
            # self.model.apply(self.initialize_weights)

            print(f'\n\nTraining from {self.train_dates[0]} to {self.train_dates[-1]}')
            print(f'Validating from {self.val_dates[0]} to {self.val_dates[-1]}')
            print(f'Testing from {self.test_dates[0]} to {self.test_dates[-1]}')
            
            # # Re-load initial model checkpoint
            # initial_model = torch.load(initial_path)
            # print('Loading untrained model...')
            # self.model.load_state_dict(initial_model['model_state_dict'])
            # self.optimizer.load_state_dict(initial_model['optimizer_state_dict'])
            # self.model.train()
            
            # Reset parameters for early stopping 
            j = 0
            self.best_val_loss = np.inf
            self.best_val_spearman = -np.inf
            self.best_val_r2 = -np.inf
            epochs_train_spearman = []
            epochs_val_spearman = []
            val_epoch_losses = [] # Used to keep track of all the losses of the epoch (only the ones passing early stopping test)
 
            for epoch in range(self.epochs):
            
                start_time_epoch = time.time()
                cum_epoch +=1 # Used for Tensorboard charts, in order to have all the epochs of all the iterations with different numbers.
                
                epoch_loss, epoch_train_spearman, epoch_train_r2 = self.__process_one_epoch('train')    
                val_loss, val_acc, epoch_val_spearman, epoch_val_r2 = self.__process_one_epoch('val')
                for idx,_ in enumerate(epoch_train_spearman):
                    epoch_train_spearman[idx] = epoch_train_spearman[idx].item()
                for idx,_ in enumerate(epoch_val_spearman):
                    epoch_val_spearman[idx] = epoch_val_spearman[idx].item()
                for idx,_ in enumerate(epoch_train_r2):
                    epoch_train_r2[idx] = epoch_train_r2[idx].item()
                for idx,_ in enumerate(epoch_val_r2):
                    epoch_val_r2[idx] = epoch_val_r2[idx].item()
                
                epoch_val_spearman = np.mean(epoch_val_spearman)
                epoch_train_spearman = np.mean(epoch_train_spearman)
                epoch_train_r2 = np.mean(epoch_train_r2)
                epoch_val_r2 = np.mean(epoch_val_r2)

                epochs_train_spearman.append(epoch_train_spearman)
                epochs_val_spearman.append(epoch_val_spearman)
                
                
                # Early stopping logic:
                if (val_loss < self.best_val_loss) and (epoch_val_spearman > self.best_val_spearman) and (epoch_val_r2 >= self.best_val_r2):# and (epoch_train_spearman > self.best_global_spearman):
                                        
                    self.best_val_loss = val_loss
                    self.best_val_r2 = epoch_val_r2
                    if epoch_val_spearman > self.best_val_spearman:
                        self.best_val_spearman = epoch_val_spearman
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch_loss': epoch_loss,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'train_spearman': epoch_train_spearman,
                        'val_spearman': epoch_val_spearman,
                        'params': self.params,
                        'n_inputs': self.n_inputs
                        }, PATH)
                    j = 0
                    val_epoch_losses.append(val_loss)
                    
                else:
                    j+=1
                    # print(f'j incremented to {j}!')
                # print(f'Epoch loss is of type: {type(epoch_loss)}, Validation loss is of type: {type(val_loss)},  Validation accuracy is of type: {type(val_acc)}')
                epoch_loss = epoch_loss.item()
                # self.writer.add_scalars('loss', {'train': float(epoch_loss)}, cum_epoch)
                # self.writer.add_scalars('loss', {'validation': float(val_loss)}, cum_epoch)
                self.writer.add_scalar("Loss/train", float(epoch_loss), cum_epoch)
                self.writer.add_scalar("Loss/validation", float(val_loss), cum_epoch)
                self.writer.add_scalar("Loss/divergence", float(val_loss) - epoch_loss, cum_epoch)
                self.writer.add_scalar("Accuracy/Validation", val_acc, cum_epoch)

                # print(f'Last val loss: {val_loss:.4f}\tLast Val Spearman: {epoch_val_spearman:.4f}')
                # print(f'Best Val loss: {self.best_val_loss}, Best Val spearman: {self.best_val_spearman}')

                results = {
                    'default': epoch_val_spearman,
                    # 'default': float(val_loss), # For minimization problems
                    'val_loss': float(val_loss),
                    'val_acc': val_acc} 
                if self.nni_experiment == True:
                    nni.report_intermediate_result(results)

                if epoch%config.ep_log_interval == 0:
                    print(f'Epoch n. {epoch+1} [of #{self.epochs}]')
                    print(f'Training loss: {round(epoch_loss, 4)} | Val loss: {round(val_loss,4)}')
                    print(f'Spearman:\tTrain: {np.mean(epoch_train_spearman)}\tVal:  {np.mean(epoch_val_spearman)}')
                    print(f'R2:\tTrain: {(100*epoch_train_r2):.3f}%\tVal: {(100*epoch_val_r2):.3f}%')
                    print(f'Validation accuracy: {round(100*val_acc,2)}%')
                    try:
                        print(f'Elapsed time for last epoch is: {elapsed_epoch_time:.2f} seconds')
                    except:
                        first_epoch_time = time.time() - start_time_epoch
                        print(f'First epoch time is: {first_epoch_time:.2f}')
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
                elapsed_epoch_time = time.time() - start_time_epoch
                # epoch+=1 # Logic to delete when going back to normal epochs
            mean_train_spearman = np.mean(epochs_train_spearman)
            mean_val_spearman = np.mean(epochs_val_spearman)
            iteration_val_spearman.append(mean_val_spearman)
            iteration_train_spearman.append(mean_train_spearman)

            print(f'The average spearman across epochs in this iteration are:\nTrain: {mean_train_spearman} | \tVal: {mean_val_spearman}')
            train_r2 = metric.normal_r2_calculation(ReturnsPrediction(self.train_loader, self.model).pred_df)
            val_r2 = metric.normal_r2_calculation(ReturnsPrediction(self.val_loader, self.model).pred_df)
            print(f'\n\nR2:\tTrain: {train_r2:.2f}\tVal: {val_r2:.2f}')    
            pred_df = ReturnsPrediction(self.test_loader, self.model).pred_df
            r2_temp = metric.normal_r2_calculation(pred_df)
            oos_spearman_corr = metric.calc_spearman(torch.tensor(pred_df['predicted_ret'], device='cpu', dtype=torch.float32), torch.tensor(pred_df['ret'], device='cpu', dtype=torch.float32))
            oos_spearman_corr.to('cpu').item()
            mda = metric.mean_directional_accuracy(pred_df['predicted_ret'], pred_df['ret'])
            print(f'Out-of-sample stats for this last iteration:\nR2:{(100*r2_temp):.3f}%\tSpearman Correlation Coefficient:{oos_spearman_corr:.3f}\tMean Directional Accuracy: {(100*mda):.3f}')
            iteration_r2.append(r2_temp)
            self.prediction_df = pd.concat([self.prediction_df, pred_df], ignore_index=True)
            
            
            self.writer.flush()



            print(f'Prediction df tail:\n {self.prediction_df.tail(5)}')
            
            val_iteration_losses.append(np.mean(val_epoch_losses))
            if self.methodology == 'expanding':
                self.__update_years()
                if self.test_dates[-1] > self.df_end_date:
                    keep_training = 0
                    print(f'The expanding window training is completed.')
            elif self.methodology == 'normal':
                keep_training = 0
                print('Normal training completed.')

        # print(f'This is again the self.prediction_df columns on line 185: {self.prediction_df.columns}')
        
        print('Training is over...')
        
        print('Re-loading best model...')
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Generating folders for saving
        predictedRetDir = config.saveDir + '/predicted_returns'
        portfolioRetDir = config.saveDir + '/portfolio_returns'
        trialInfoDir = config.saveDir + '/trial_info'
        portfolioWeightsDir = config.saveDir + '/portfolio_weights'
        cumulativeReturnsDir = config.saveDir + '/cumulative_log_returns'
        plotsDir = config.saveDir + '/plots'

        # Saving files
        
        if os.path.exists(predictedRetDir) == False:
            os.makedirs(predictedRetDir)
        if os.path.exists(portfolioRetDir) == False:
            os.makedirs(portfolioRetDir)
        if os.path.exists(trialInfoDir) == False:
            os.makedirs(trialInfoDir)
        if os.path.exists(portfolioWeightsDir) == False:
            os.makedirs(portfolioWeightsDir)
        if os.path.exists(cumulativeReturnsDir) == False:
            os.makedirs(cumulativeReturnsDir)
        if os.path.exists(plotsDir) == False:
            os.makedirs(plotsDir)
        

        self.prediction_df.to_csv(predictedRetDir + '/' + str(timeStamp) + '_predicted_returns.csv')
        print('\Prediction Dataframe Saved...\n')

        val_spearman_correlation = np.mean(iteration_val_spearman)
        train_spearman_correlation = np.mean(iteration_train_spearman)
        
        out_of_sample_spearman = metric.calc_spearman(torch.tensor(self.prediction_df['predicted_ret'], device='cpu', dtype=torch.float32), torch.tensor(self.prediction_df['ret'], device='cpu', dtype=torch.float32))
        out_of_sample_spearman.to('cpu').item()

        print(f'Mean Validation losses for each iteration are:\n{val_iteration_losses}')
        r2 = metric.r2_metric_calculation(self.prediction_df)
        print(f'R2 calculated using metric.r2_metric_calculation is {r2}')

        print(f'The average R2 of the ones calculated for each iteration is {np.mean(iteration_r2)}')

        self.writer.add_scalar("R2/All", r2['R2'])
        self.writer.add_scalar("R2/Top1000", r2['R2_top_1000'])
        self.writer.add_scalar("R2/Bottom1000", r2['R2_bottom_1000'])
        
        self.writer.flush()    

        timeStamp_id = dt.datetime.now().strftime('%Y%m%d-%H:%M:%S:%f')
        
        print('Calculating portfolios...')
        # print('Prediction df:')
        # print(self.prediction_df.describe())

        self.prediction_df.sort_values(['permno','yyyymm'], inplace=True)
        # print('NaNs in prediction_df:')
        count = self.prediction_df.isna().sum()
        percentage = self.prediction_df.isna().mean()
        null_values = pd.concat([count, percentage], axis=1, keys=['count', '%'])
        # print(null_values)

        print(f'Rebalancing is: {config.rebalancing_frequency}')
        
        portfolio = Portfolio(pred_df=self.prediction_df, rebalancing_frequency=config.rebalancing_frequency, weighting = config.weighting)
        information_ratio_regression = portfolio.information_ratio_regression
        alpha = portfolio.alpha
        returns = portfolio.returns
        information_ratio = metric.information_ratio(returns)
        # portfolio_weights = portfolio.portfolio_weights

        

        # print('\nNaNs in Portfolio returns:')
        # count = returns.isna().sum()
        # percentage = returns.isna().mean()
        # null_values = pd.concat([count, percentage], axis=1, keys=['count', '%'])
        print(null_values)
        print(f'IR: {information_ratio}\nAlpha: {alpha}')
        # print('Portfolio Returns:')
        # print(returns.describe())
        
        
        sharpe_ratio = metric.calc_sharpe_ratio(returns)
        print(f'Sharpe Ratio of {config.rebalancing_frequency} {config.weighting} portfolio:')
        print(sharpe_ratio)
        # print(type(sharpe_ratio))

    
 

        from csv import DictWriter, writer

        field_names = [
            'timeStamp',
            # 'information_ratio',
            'alpha', 
            'r2_oos', 
            'train_spearman',
            'val_spearman',
            'val_loss',
             'long_alpha_on_market'
            # 'Sharpe_Ratio'
            ]
        summary_dict = {
            # 'timeStamp': timeStamp_id,
            'timeStamp': timeStamp,
            # 'information_ratio': information_ratio, 
            'alpha': alpha, 
            'r2_oos': r2['R2'], 
            'train_spearman': train_spearman_correlation,
            'val_spearman': val_spearman_correlation,
            'val_loss': val_loss,
            'long_alpha_on_market': portfolio.alpha_market_long
            # 'Sharpe_Ratio': sharpe_ratio
            }

        with open(config.saveDir + '/experiment_summary.csv', 'a') as fp:          
            writer_obj = writer(fp)
            if fp.tell() == 0:
                writer_obj.writerow(field_names)
            dictwriter_object = DictWriter(fp, fieldnames=field_names)
            dictwriter_object.writerow(summary_dict)
  
        returns.to_csv(portfolioRetDir + '/' + str(timeStamp) + '_portfolio_returns.csv')
        
        final_dict = {
            'params': self.params,
            'alpha':alpha, 
            'r2': r2,
            'Information Ratio': information_ratio,
            'Regression-based Information Ratio': information_ratio_regression,
            'Sharpe Ratio': sharpe_ratio,
            'Train Spearman': train_spearman_correlation,
            'Val Spearman': val_spearman_correlation,
            'Others': 'Think about adding other metrics (Max_DD, turnover, ...)'}

        with open(trialInfoDir + '/' + str(timeStamp) + '_trial_full.json', 'w') as fp:
            json.dump(final_dict, fp, indent=4)
        
        # Saving portfolio weights
        # portfolio_weights.to_csv(portfolioWeightsDir + '/' + timeStamp + '_portfolio_weights.csv')

        # Saving model
        # torch.save(self.model, config.paths['hpoResultsPath'] + '/models/' + timeStamp_id + ' - model.pt')
        # torch.save(self.model.state_dict(), config.paths['hpoResultsPath'] + '/models/' + timeStamp_id + ' - model_state_dict.pt')

        cumulative_log_returns = portfolio.cum_returns
        cumulative_log_returns.to_csv(cumulativeReturnsDir + '/' + str(timeStamp) + '_cumulative_log_returns.csv')
        plot_cum_ret_path = plotsDir + '/' + str(timeStamp) + '_cumulative_log_returns.png'
        portfolio.plot_cumulative_returns(plot_cum_ret_path)

        plot_feature_importance = plotsDir + '/' + str(timeStamp)
        try:
            IntegratedGradients_importance(self.model, plot_feature_importance)
        except:
            pass
            
        print('Portfolio returns calculation completed.')
       
        best_val_loss = min(val_iteration_losses)
        results = {
            'default': val_spearman_correlation,
            # 'default': float(best_val_loss), # Standard nni reporting for minimization
            'val_loss': float(best_val_loss), 
            'long_alpha_on_market': portfolio.alpha_market_long,
            'val_acc': val_acc,
            'Train Spearman': train_spearman_correlation,
            # 'Val Spearman': val_spearman_correlation,
            'alpha': float(alpha),
            'information_ratio': float(information_ratio_regression),
            'R2': r2['R2']}

        # print(r2)
        # os.remove(PATH)
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
        epoch_train_spearman = []
        epoch_val_spearman = []
        epoch_train_r2 = []
        epoch_val_r2 = []
        if mode == 'train':
            # print(f'num_batches = len(dataloader): {len(loader)}')
            # print(f'Size = len(dataloader.dataset):{len(loader.dataset)}')
            # print(f'Batch size: {loader.batch_size}')
                
            for i, (inputs, target, labels) in enumerate(loader):
                
                loss, correct, train_spearman, train_r2 = self.__process_one_step(inputs, target, labels, mode)
                total_loss += loss
                train_spearman = train_spearman.to('cpu')
                epoch_train_spearman.append(train_spearman)
                epoch_train_r2.append(train_r2)
            total_loss /= num_batches
        
        # total_acc = total_acc / loader.batch_size * 100
        
        elif mode == 'val':
            for i, (inputs, target, labels) in enumerate(loader):
                loss, correct, val_spearman, val_r2 = self.__process_one_step(inputs, target, labels, mode)
                val_spearman = val_spearman.to('cpu')
                epoch_val_spearman.append(val_spearman)
                epoch_val_r2.append(val_r2)
                val_loss += loss.item()
                total_correct += correct
            # print(f'Total correct in validation:{total_correct}')              
            val_acc = total_correct/size
            val_loss /= num_batches
            # print(f"Validation Error: \n Accuracy: {(100*val_acc):>0.1f}%, Avg loss: {val_loss:>8f} \n")

        if mode == 'train':
            return total_loss, epoch_train_spearman, epoch_train_r2
        else:
            return val_loss, val_acc, epoch_val_spearman, epoch_val_r2


    def __process_one_step(self, inputs, target, labels, mode):
        if mode =='train':
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
            l1 = l1_norm.squeeze() * self.l1_lambda
        else:
            l1 = 0

        if self.l2_reg:
            l2_norm = sum(p.pow(2.0).sum()
                for p in self.model.parameters())
            l2 = l2_norm.squeeze() * self.l2_lambda
        else:
            l2 = 0
        
        loss = loss.squeeze() + l1 + l2

        spearman = metric.calc_spearman(yhat, target.float().squeeze())
        spearman.to('cpu')

        r2 = metric.r2_in_training(yhat, target.float().squeeze())


        if mode == 'train':
            loss.backward()
            self.optimizer.step()

        return loss, correct, spearman, r2


    def __update_years(self):

            print('Updating Train starting year for rolling window training')
            self.train_starting_year = int((dt.datetime.strptime(
                str(self.train_starting_year), '%Y%m') + DateOffset(
                    years=1
                )).strftime('%Y%m'))

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
        # Modify batch size to make it trainable with higher values
        # Modify dataloader args
        try:
            self.train_loader = DataLoader(self.train, batch_size=self.params['batch_size'], shuffle=True)
            self.val_loader = DataLoader(self.validation, batch_size=self.params['batch_size'], shuffle=True)
        except:
            self.train_loader = DataLoader(self.train, batch_size=10000, shuffle=True)
            self.val_loader = DataLoader(self.validation, batch_size=10000, shuffle=True)
        self.test_loader = DataLoader(self.test, batch_size=bs)

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.params['act_func'] == 'LeakyReLU':
                # print('Activation Function is LeakyReLU')
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif self.params['act_func'] == 'ReLU':
                # print('Activation Function is ReLU')
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            else:
                # print('Xavier Uniform for other activation functions')
                nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    