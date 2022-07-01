import config
import pandas as pd
import os
import datetime as dt
import statsmodels.api as sm
from portfolios.FF5FM_Mom import FF5FM_Mom
from portfolios.ReturnsPrediction import ReturnsPrediction


class Portfolio():
    def __init__(self, n_cuts=10, rebalancing_frequency='yearly', weighting="VW"):
        self.n_cuts = n_cuts
        self.weighting = weighting
        self.rebalancing_frequency = rebalancing_frequency

        self.alpha = None
        self.t_value_alpha = None
        self.information_ratio = None
        self.returns = None


        if os.path.exists(config.paths['PredictedRetPath']):
            self.__load_pred_df()

        else:
            print('Predicted Returns file not in memory, predicting returns...')
            ReturnsPrediction()
            self.__load_pred_df()

            
        self.calculate_portfolio_weights()
        self.calculate_portfolio_monthly_returns()
        self.calculate_information_ratio()


    def __load_pred_df(self):
        crsp = pd.read_csv(config.paths['CRSPretPath'])
        crsp['ret'] = (crsp['ret']/100)
        
        self.__pred_df = pd.read_csv(config.paths['PredictedRetPath'], index_col=0)
        
        # Dropping returns from pred_df, as they are winsorized there.
        self.__pred_df.drop('ret', axis=1, inplace=True)
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
        
        df = df.merge(FFMom, on=['yyyymm'], how='left')

        # On long-short returns        
        X = df.iloc[:, 3:]
        
        # Column 1 is long returns on max quantile, 
        # Column 2 is long-short returns
        y = df.iloc[:,2]

        X = sm.add_constant(X)
        lm = sm.OLS(y, X).fit()

        self.alpha = lm.params[0]
        #self.t_value_alpha = lm.tvalues[0]
        self.information_ratio = lm.tvalues[0]


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

        # Calculating bins..
        # Still have to check cut vs. qcut
        rebalancing_month_bins = df_rebalancing_months.set_index(['yyyymm', 'permno'])\
            .groupby('yyyymm')['predicted_ret']\
                .apply(lambda x: pd.cut(x, bins=self.n_cuts, labels=False))\
                    .reset_index(name='bin')


        df_rebalancing_months = df_rebalancing_months.merge(
            rebalancing_month_bins,
            on=['permno','yyyymm'],
            how='left'
        )


        if self.weighting == 'VW':

            df_rebalancing_months['pweight'] =\
                ((df_rebalancing_months['melag'])\
                   /(df_rebalancing_months.groupby(['yyyymm','bin'])\
                        ['melag'].transform('sum')))

        elif self.weighting == 'EW':
            
            df_rebalancing_months['pweight'] =(
                1/df_rebalancing_months.groupby(
                ['yyyymm','bin'])['melag']\
                .transform('count'))
        
        self.__pred_df = self.__pred_df.merge(
            df_rebalancing_months[['permno', 'yyyymm', 'bin','pweight']],
            on=['permno','yyyymm'],
            how='left')
        
        # Idea could be to add a column which contains the split between rebalancing dates
        # such that if some - Chen and Zimmermann do not do that in their code (maybe because I have dropped
        # rows with NaN as return)
        self.__pred_df.sort_values(['permno', 'yyyymm'], ascending=[True, True]) 
        self.__pred_df['bin'] = self.__pred_df.groupby(['permno'])[['bin']].ffill()
        self.__pred_df['pweight'] = self.__pred_df.groupby(['permno','bin'])[['pweight']].ffill()

        # Drop NaN in pweight filled, they are related to stocks that started 
        # trading before the rebalancing period, before being included 
        # in the rebalanced portfolio
        self.__pred_df.dropna(subset=['pweight'], inplace=True)


    def calculate_portfolio_monthly_returns(self):
        
        portfolio_returns = self.__pred_df.groupby(
            ['yyyymm','bin']).apply(
                lambda x: sum(x['ret']*x['pweight']))\
                .reset_index(name='portfolio_ret')
        
        portfolio_returns['bin'] = portfolio_returns.bin.astype(int)

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
