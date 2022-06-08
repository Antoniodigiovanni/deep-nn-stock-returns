import config
import pandas as pd
import os
import datetime as dt
from portfolios.FF5FM_Mom import FF5FM_Mom

# Making pred_df private could be done, like __pred_df

class Portfolio():
    def __init__(self, n_cuts=10, rebalancing_frequency='yearly', weighting="VW"):
        self.n_cuts = n_cuts
        self.weighting = weighting
        self.rebalancing_frequency = rebalancing_frequency
        
        try:
            crsp = pd.read_csv(config.paths['CRSPretPath'])
            crsp['ret'] = (crsp['ret']/100)

            # Not joining with crspminfo so not needed
            #crsp.drop(['prc','exchcd','siccd','shrcd','me_nyse10','me_nyse20'], axis=1, inplace=True)     
            assert os.path.exists(config.paths['PredictedRetPath']), 'The predicted returns df does not exist, perform the prediction!'
            
            self.pred_df = pd.read_csv(config.paths['PredictedRetPath'], index_col=0)
            
            # Dropping returns from pred_df, as they are winsorized there.
            self.pred_df.drop('ret', axis=1, inplace=True)
            self.pred_df['permno'] = self.pred_df.permno.astype(int)

            self.pred_df = self.pred_df.merge(
                crsp[['permno','yyyymm','melag', 'ret']], 
                on=['permno','yyyymm'], 
                how='left'
            )

            print('Pred_df Head:')
            print(self.pred_df.head())

            # Still have to check cut vs. qcut
            # Cutting in bins is done only in rebalancing dates.
            # pred_df = pred_df.set_index(['yyyymm', 'permno'])\
            #     .groupby('yyyymm')['predicted_ret']\
            #         .apply(lambda x: pd.cut(x, bins=n_cuts, labels=False))\
            #             .reset_index(name='bin')
            
            # self.returns = pred_df.merge(crsp, on=['permno', 'yyyymm'], how='left')
            # print('Merged df Head:')
            # print(self.df.head())

            
            self.calculate_portfolio_weights()
            self.calculate_portfolio_monthly_returns()
            self.calculate_information_ratio()

        except AssertionError as msg:
            print(msg)


    def calculate_information_ratio(self):
        FFMom = FF5FM_Mom().returns
        """
            Finish implementing the information ratio calculation.

        """
        

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
                str(self.pred_df['yyyymm'].min()),'%Y%m'),
                periods=rebalancing_map[self.rebalancing_frequency][1],
                freq=rebalancing_map[self.rebalancing_frequency][0]))
                .strftime('%Y%m')
                .astype(int)
                )

        return rebalancing_months
            
    def calculate_portfolio_weights(self):
        
        rebalancing_months = self.list_rebalancing_dates()

        # Select rows of the rebalancing date (yearly, monthly or quarterly)
        df_rebalancing_months = self.pred_df.loc[self.pred_df['yyyymm'].isin(rebalancing_months)].copy()

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
        
        self.pred_df = self.pred_df.merge(
            df_rebalancing_months[['permno', 'yyyymm', 'bin','pweight']],
            on=['permno','yyyymm'],
            how='left')
        
        # Idea could be to add a column which contains the split between rebalancing dates
        # such that if some - Chen and Zimmermann do not do that in their code (maybe because I have dropped
        # rows with NaN as return)
        self.pred_df.sort_values(['permno', 'yyyymm'], ascending=[True, True]) 
        self.pred_df['bin'] = self.pred_df.groupby(['permno'])[['bin']].ffill()
        self.pred_df['pweight'] = self.pred_df.groupby(['permno','bin'])[['pweight']].ffill()

        # Drop NaN in pweight filled, they are related to stocks that started 
        # trading before the rebalancing period, before being included 
        # in the rebalanced portfolio
        self.pred_df.dropna(subset=['pweight'], inplace=True)


    def calculate_portfolio_monthly_returns(self):
        print('Pred df head again:')
        print(self.pred_df.head())
        portfolio_returns = self.pred_df.groupby(
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

        print('Port returns head:')
        print(portfolio_returns.head())
