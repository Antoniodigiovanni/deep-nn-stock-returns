import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.api as sm
import numpy as np
import config
from pandas.tseries.offsets import DateOffset
import datetime as dt
import pandas as pd
from data.dataset import BaseDataset

class PooledOLS():

    def __init__(self, x_list, y, df, train_window_years=20, test_window_years = 5):
        # if df == None:
        #     dataset = BaseDataset()
        #     self.df = dataset.df
        # else:
        self.df = df 
        self.methodology = 'expanding'
        self.y = y
        self.x_list = x_list
        self.train_starting_yyyymm = df.yyyymm.min()
        self.last_yyyymm = df.yyyymm.max()
        # self.df_end_date = dataset.yyyymm.max()
        

        self.train_window = train_window_years
        self.test_window = test_window_years
        
        self.test_starting_year = int((dt.datetime.strptime(
            str(self.train_starting_yyyymm), '%Y%m') + 
            DateOffset(years=self.train_window)).strftime('%Y%m'))
        
        # Subtracting one month from the start of the validation period
        # gives us the end of the training period
        self.end_train = int((dt.datetime.strptime(
            str(self.test_starting_year), '%Y%m') - 
            DateOffset(months=1)).strftime('%Y%m'))
        

        self.train_dates, self.test_dates = None, None
        self.train, self.test =None, None

        self.df = self._shift_ret_and_yyyymm(self.df)
        self.__generate_training_window()
        self.subset_df()
        self.regress()


    def __update_years(self):
        self.test_starting_year = int((dt.datetime.strptime(
            str(self.test_starting_year), '%Y%m') + DateOffset(
                years=1
            )).strftime('%Y%m'))
        
        self.end_train = int((dt.datetime.strptime(
            str(self.test_starting_year), '%Y%m') - 
            DateOffset(months=1)).strftime('%Y%m'))

        # Update the yyyymm list
        self.__generate_training_window()
        self.subset_df()       


    def __generate_training_window(self):
        self.train_dates = list(
            pd.period_range(
                start = dt.datetime.strptime(
                    str(self.train_starting_yyyymm), '%Y%m'),
                end=dt.datetime.strptime(str(self.end_train), '%Y%m'),
                freq='M')
                .strftime('%Y%m')
                .astype(int)
                )

        if self.methodology == 'normal':
            self.test_dates = list(
                pd.period_range(
                    start= (dt.datetime.strptime(str(self.train_dates[-1]), '%Y%m') + DateOffset(months=1)),
                    end=dt.datetime.strptime(str(self.df_end_date), '%Y%m'),
                    freq='M'
                    )
                    .strftime('%Y%m')
                    .astype(int)
                    )
        elif self.methodology == 'expanding':
              self.test_dates = list(
            pd.period_range(
                start= (dt.datetime.strptime(str(self.train_dates[-1]), '%Y%m') + DateOffset(months=1)),
                periods=(12*self.test_window),
                freq='M'
                )
                .strftime('%Y%m')
                .astype(int)
                )
        

    def subset_df(self):
        train = self.df.loc[self.df['yyyymm'].isin(self.train_dates)].copy()
        test = self.df.loc[self.df['yyyymm'].isin(self.test_dates)].copy()

        self.train_y = train[self.y]
        self.train_x = train[self.x_list]
        self.train_labels = train[['permno','yyyymm']]

        self.test_y = test[self.y]
        self.test_x = test[self.x_list]
        self.test_labels = test[['permno','yyyymm']]


    def _shift_ret_and_yyyymm(self, df):
        cols_to_shift = df[['yyyymm','ret']].columns
        df['date'] = df['yyyymm'].apply(lambda x: dt.datetime.strptime(str(x), '%Y%m'))
        df['tmp'] = (df['date']  + pd.DateOffset(months=1))

        s = df.groupby('permno').apply(lambda x: x['tmp'].isin(x['date']))
        df['present_following_month'] = s.reset_index(level=0)['tmp']

        # Shift features to next month
        shifted_df = df.copy()
        shifted_df[cols_to_shift] = shifted_df.sort_values(by='yyyymm').groupby('permno')[cols_to_shift].shift(-1)

        shifted_df = shifted_df.loc[shifted_df['present_following_month'] == True].copy()
        shifted_df.drop(['tmp','date', 'present_following_month'], axis=1, inplace=True)

        shifted_df = shifted_df.dropna()

        return shifted_df


    def regress(self):
        keep_training = 1
        self.prediction_df = pd.DataFrame()
        while keep_training == 1:
            print(f'Training from {self.train_dates[0]} to {self.train_dates[-1]}')
            print(f'Forecasting from {self.test_dates[0]} to {self.test_dates[-1]}')

            self.pooled_x = sm.add_constant(self.train_x)
            self.pooled_y = self.train_y
            pooled_osr_model = sm.OLS(endog=self.pooled_y.astype(float), exog=self.pooled_x.astype(float))
            pooled_osr_model_results = pooled_osr_model.fit()

            # with open(config.paths['resultsPath']+'/PooledOLSsummary.txt', 'w') as fh:
                # fh.write(pooled_osr_model_results.summary().as_text())
            
            # with open(config.paths['resultsPath']+'/PooledOLSsummary.csv', 'w') as fh:
                # fh.write(pooled_osr_model_results.summary().as_csv())
            
            self.summary = pooled_osr_model_results.summary()            
            # print(self.summary)

            self.test_x = sm.add_constant(self.test_x)
            predicted_ret = (pooled_osr_model_results.params * self.test_x).sum(axis=1).rename('predicted_ret')
            pred_df = self.test_labels.join(predicted_ret)
            pred_df = pred_df.join(self.test_y)

            self.prediction_df = pd.concat([self.prediction_df, pred_df], ignore_index=True)
            
            if self.methodology == 'expanding':
                self.__update_years()
                if self.test_dates[-1] > self.last_yyyymm:
                    keep_training = 0
                    print(f'The expanding window training is completed.')
            elif self.methodology == 'normal':
                keep_training = 0
                print('Normal training completed.')

        self.prediction_df.yyyymm = self.prediction_df.yyyymm.astype(int)
        print('Training is over')

    def plot_df(self, x):
            
            sns.scatterplot(
                x=self.df[x], y=self.df[self.y], hue=self.df['permno'],).\
                    set(title= str(x) + 'v. Stock returns' )
            plt.show()


