import numpy as np
import config
import pandas as pd
import os
import datetime as dt
import statsmodels.api as sm
from portfolios.FF5FM_Mom import FF5FM_Mom


class Portfolio():
    def __init__(self, n_cuts=10, pred_df = None, rebalancing_frequency='yearly', weighting="VW", verbose = True):
        self.n_cuts = n_cuts
        self.weighting = weighting
        self.rebalancing_frequency = rebalancing_frequency
        self.verbose = verbose

        self.portfolio_weights = None
        self.alpha = None
        self.t_value_alpha = None
        self.information_ratio = None
        self.returns = None
        self.__pred_df = pred_df

        self.__load_pred_df()
        self.calculate_portfolio_weights()
        self.calculate_portfolio_monthly_returns()
        self.calculate_information_ratio()


    def __load_pred_df(self):
        crsp = pd.read_csv(config.paths['CRSPretPath'])
        # crsp['ret'] = (crsp['ret']/100)

        # if self.__pred_df == None:        
            # if os.path.exists(config.paths['PredictedRetPath']):
            #     self.__pred_df = pd.read_csv(config.paths['PredictedRetPath'], index_col=0)
            
            # else:
            #     print('Predicted Returns file not in memory, predicting returns...')
            #     ReturnsPrediction()
        
        # Dropping returns from pred_df, as they are winsorized there.
        self.__pred_df.drop('ret', axis=1, inplace=True, errors='ignore')
        self.__pred_df['permno'] = self.__pred_df.permno.astype(int)

        self.__pred_df = self.__pred_df.merge(
                crsp[['permno','yyyymm','melag', 'ret']], 
                on=['permno','yyyymm'], 
                how='left'
            )
    
    
    def calculate_information_ratio(self):
        FFMom = FF5FM_Mom().returns
        """
            Finish implementing the information ratio calculation.
            Regress the long and long-short

        """
        columns_to_keep = [self.returns.columns[0]]
        columns_to_keep.extend(self.returns.columns[-2:])
        df = self.returns[columns_to_keep]
        if self.verbose == True:
            print('IR & alpha calculation:')
            print(df.head())
        df = df.merge(FFMom, on=['yyyymm'], how='left')
        
        
        # On long-short returns        
        X = df.iloc[:, 3:]

        # Column 1 is long returns on max quantile, 
        # Column 2 is long-short returns
        y = df.iloc[:,2]

        X = sm.add_constant(X)
        lm = sm.OLS(y, X).fit()

        if self.verbose == True:
            print('Params')
            print(lm.params)
            print('tValues')
            print(lm.tvalues)
        self.alpha = lm.params[0]
        self.t_value_alpha = lm.tvalues[0]
        
        year_min = self.returns['yyyymm'].min()//100
        month_min = self.returns['yyyymm'].min()%100

        year_max = self.returns['yyyymm'].max()//100
        month_max = self.returns['yyyymm'].max()%100

        n_months =  ((year_max - year_min) + (month_max-month_min+1)/12)*12 
        self.information_ratio = self.t_value_alpha / np.sqrt(n_months)


    def list_rebalancing_dates(self):
        """
            Portfolio rebalancing dates based on the desired  frequency.
            The rebalancing_map contains the Pandas code for rebalancing yearly, quarterly, ...
            and the length of the series generated (ballpark figures) which varies based on
            the frequency.
        
        """
        rebalancing_map = {'yearly': ['A-JUN', 120], 'quarterly': ['Q', 700], 'monthly':['M', 1500]}
        

        rebalancing_months = list(
            (pd.period_range(dt.datetime.strptime(
                str(self.__pred_df['yyyymm'].min()),'%Y%m'),
                periods=rebalancing_map[self.rebalancing_frequency][1],
                freq=rebalancing_map[self.rebalancing_frequency][0]))
                .strftime('%Y%m')
                .astype(int)
                )

        return rebalancing_months
            
    def calculate_portfolio_weights(self):
        
        rebalancing_months = self.list_rebalancing_dates()

        # Select rows of the rebalancing date (yearly, monthly or quarterly)
        df_rebalancing_months = self.__pred_df.loc[self.__pred_df['yyyymm'].isin(rebalancing_months)].copy()

        
        # Moreover, I will be adding a small infinitesimal number to avoid the case in which some
        # predicted returns are equal causing the presence of less than 10 unique predicted_ret in the df
        # (with some architectures it happens in 3 rows on 86000+, but still messes up all calculation)
        # import random
        # df_rebalancing_months['predicted_ret'] = df_rebalancing_months['predicted_ret'].apply(lambda x: x + random.uniform(1e-30, 1e-15))
        
        """ Original method in the next 4 lines"""
        # rebalancing_month_bins = df_rebalancing_months.set_index(['yyyymm', 'permno'])\
        #     .groupby('yyyymm')['predicted_ret']\
        #         .apply(lambda x: pd.qcut(x, q=self.n_cuts, labels=False, duplicates='drop'))\
        #             .reset_index(name='bin')
        
        # Updated method which is more concise and does not require a merge.
        # try:
        df_rebalancing_months['bin'] = (
            df_rebalancing_months.groupby('yyyymm')['predicted_ret']
            .transform(lambda x: pd.qcut(x, q=self.n_cuts, labels = False))#, duplicates='drop'))
            )
        #     df_rebalancing_months['predicted_ret_noise'] = df_rebalancing_months['predicted_ret'] + 0.0000001 * (np.random.rand(len(df_rebalancing_months)) - 0.5)
        #     df_rebalancing_months['bin_test'] = (
        #         df_rebalancing_months.groupby('yyyymm')['predicted_ret_noise']
        #         .transform(lambda x: pd.qcut(x, q=self.n_cuts, labels=False))
        #     ) 
        #     if df_rebalancing_months['bin'].equals(df_rebalancing_months['bin_test']) == False:
        #         print('The two methodologies are not equal')
        #     else:
        #         print('The two methodologies are equal')
        # except:
        #     print('\n\nError in calculating deciles, likely the bins are non-unique using alternative method:')
        #     df_rebalancing_months['predicted_ret_noise'] = df_rebalancing_months['predicted_ret'] + 0.0000001 * (np.random.rand(len(df_rebalancing_months)) - 0.5)

        #     df_rebalancing_months['bin'] = (
        #         df_rebalancing_months.groupby('yyyymm')['predicted_ret_noise']
        #         .transform(lambda x: pd.qcut(x, q=self.n_cuts, labels=False))
        #     )
        # df_rebalancing_months.drop(['predicted_ret_noise', 'bin_test'], axis=1, errors='ignore', inplace=True)
        
        # print('Are the two columns equal?')
        # print(df_rebalancing_months['bin'].equals(rebalancing_month_bins['bin']))
        
        if self.verbose == True:
            print(f'Rebalancing month which have at least one NaN as bin:')
            print(df_rebalancing_months.loc[df_rebalancing_months['bin'].isna()]['yyyymm'].unique())

        # df_rebalancing_months = df_rebalancing_months.merge(
        #     rebalancing_month_bins,
        #     on=['permno','yyyymm'],
        #     how='left'
        # )


        if self.weighting == 'VW':

            df_rebalancing_months['pweight'] =(
                df_rebalancing_months['melag']/(df_rebalancing_months.groupby(['yyyymm','bin'])['melag'].transform('sum'))
                )


        elif self.weighting == 'EW':            
            df_rebalancing_months['pweight'] =(
                1/df_rebalancing_months.groupby(['yyyymm','bin'])['melag'].transform('count'))

        if self.verbose == True:
            print('Checking weights sum:')
            df_rebalancing_months['weight_sum'] = df_rebalancing_months.groupby(['yyyymm', 'bin'])['pweight'].transform('sum').round(5)
            
            if (df_rebalancing_months['weight_sum'] == df_rebalancing_months['weight_sum'].iloc[0]).all():
                print("All values are equal in column 'Weight Sum'")
            else:
                print("Some values are not equal to the first row in column 'Weight Sum', printing them below:")
                print(df_rebalancing_months.loc[(df_rebalancing_months['weight_sum'] != df_rebalancing_months['weight_sum'].iloc[0])])
        
        self.__pred_df = self.__pred_df.merge(
            df_rebalancing_months[['permno', 'yyyymm', 'bin','pweight']],
            on=['permno','yyyymm'],
            how='left')
        
        # Saving weights in an accessible df from outside - is this the right position to do this or should I do it after the ffill?
        self.portfolio_weights = df_rebalancing_months

        if self.verbose == True:
            print('Pred_df NaNs:')
            print(self.__pred_df.isna().sum())
        
        # Idea could be to add a column which contains the split between rebalancing dates
        # such that if some - Chen and Zimmermann do not do that in their code (maybe because I have dropped
        # rows with NaN as return)
        self.__pred_df.sort_values(['permno', 'yyyymm'], ascending=[True, True]) 
        self.__pred_df['pweight'] = self.__pred_df.groupby(['permno','bin'])[['pweight']].ffill()

        if self.verbose == True:
            print('Pred_df NaNs (after ffill):')
            print(self.__pred_df.isna().sum())

        if self.verbose == True:
            print('Re-checking weights sum after ffill:')
            self.__pred_df['weight_sum'] = self.__pred_df.groupby(['yyyymm', 'bin'])['pweight'].transform('sum').round(5)
            
            if (self.__pred_df['weight_sum'] == self.__pred_df['weight_sum'].iloc[0]).all():
                print("All values are equal in column 'Weight Sum'")
            else:
                print("Some values are not equal to the first row in column 'Weight Sum', printing them below:")
                print(self.__pred_df.loc[(self.__pred_df['weight_sum'] != self.__pred_df['weight_sum'].iloc[0])])

        # Drop NaN in pweight filled, they are related to stocks that started 
        # trading before the rebalancing period, before being included 
        # in the rebalanced portfolio
        self.__pred_df.dropna(subset=['pweight'], inplace=True)
        
        if self.verbose == True:
            print('\n\nPred_df NaNs (final):')
            print(self.__pred_df.isna().sum())

        

    def calculate_portfolio_monthly_returns(self):
        
        portfolio_returns = self.__pred_df.groupby(
            ['yyyymm','bin']).apply(
                lambda x: sum(x['ret']*x['pweight']))\
                .reset_index(name='portfolio_ret')

        portfolio_returns = portfolio_returns.pivot(
            index='yyyymm',
            columns='bin',
            values='portfolio_ret')\
            .reset_index()
        
        portfolio_returns =\
            portfolio_returns.rename_axis(None, axis=1)


        portfolio_returns[(str(portfolio_returns.columns[-1]) + \
        '-' + str(portfolio_returns.columns[1]))] =\
            portfolio_returns.iloc[:,-1] -\
            portfolio_returns.iloc[:,1]
        
        self.returns = portfolio_returns
        
        if self.verbose == True:
            print('Portfolio Returns NaNs:')
            print(self.returns.isna().sum())


            cum_ret = ((self.returns.iloc[:,-1]/100+1).cumprod()-1)*100
            print(f'Cumulative return: {(cum_ret).iloc[-1]}%')
            
            avg_ret = self.returns.iloc[:,-1].mean()
            print(f'Average monthly return: {avg_ret}%') # Add *100 if using decimals
    
    def metrics_calculation(self):
        self.metrics = {}
        pass
