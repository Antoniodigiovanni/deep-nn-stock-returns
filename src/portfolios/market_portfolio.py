import config.config as config
import pandas as pd
import data.data_preprocessing as dp
import numpy as np
from data.dataset import BaseDataset


class MarketPortfolio():
    def __init__(self, weighting = 'VW', keep_micro_stocks = False, keep_medium_stocks = True, keep_big_stocks = True) -> None:
        self.weighting = weighting
        
        stocks_to_keep = []
        if keep_micro_stocks:
            stocks_to_keep.append('Micro')
        if keep_medium_stocks:
            stocks_to_keep.append('Medium')            
        if keep_big_stocks:
            stocks_to_keep.append('Big')
        self.stocks_to_keep = stocks_to_keep

        self.__load_crsp()
        self.market_ret_calculation()
        


    
    def market_ret_calculation(self):
        if self.weighting == 'VW':
            weight = pd.DataFrame(self.crsp.groupby(['yyyymm'])['melag'].sum()).reset_index(drop=False)
            weight = weight.rename(columns={'melag':'total_melag'})
            # print(f'Weight columns = {weight.columns}')
            self.crsp = self.crsp.merge(weight, on=['yyyymm'], how='inner')
            self.crsp['weight'] = self.crsp['melag']/self.crsp['total_melag']
            self.crsp = (self.crsp.sort_values(by=['permno','yyyymm'])).reset_index(drop=True)
            self.crsp['market_ret'] = self.crsp['ret'] * self.crsp['weight']
            self.crsp.loc[self.crsp['permno'] != self.crsp['permno'].shift(), 'market_ret'] = np.nan
            # self.pred_df.vw_ret[self.pred_df.permno != self.pred_df.permno.shift()] = np.nan
            monthly_vw = self.crsp.groupby(['yyyymm'])['market_ret'].sum()
            # monthly_vw = monthly_vw.unstack('decile')
            monthly_vw = monthly_vw.reset_index(drop=False)
            # print('Monthly VW before indexing:')
            # print(monthly_vw.head(5))
            self.mkt_returns = monthly_vw
            # monthly_vw = monthly_vw.loc[1:,:]
            
        elif self.weighting == 'EW':
            monthly_ew = self.crsp.groupby(['yyyymm'])['ret'].mean()
            # monthly_ew.unstack('decile')
            monthly_ew = monthly_ew.reset_index(drop=False)
            self.mkt_returns = monthly_ew
            # print(monthly_ew.head())
        else:
            print('Inserted weighting method is not valid.\nUse EW for equal weighted or VW for value weighted')

        """
        if self.weighting == 'VW':
            self.crsp['pweight'] =(
                self.crsp['melag']/(self.crsp.groupby(['yyyymm'])
                ['melag'].transform('sum')))

        elif self.weighting == 'EW':            
            self.crsp['pweight'] =(
                1/self.crsp.groupby(['yyyymm'])['melag'].transform('count'))

        self.mkt_returns = self.crsp.groupby(
            ['yyyymm']).apply(
                lambda x: sum(x['ret']*x['pweight']))\
                .reset_index(name='market_ret')
        """
    def __load_crsp(self):

        crsp = dp.load_crsp()
        
        dataset = BaseDataset()
        df = dataset.df
        df = df[['permno','yyyymm']].copy()
        # df = dp.remove_microcap_stocks(df, self.stocks_to_keep)
        # df = dp.filter_exchange_code(df)
        # df = dp.filter_share_code(df)
        # df = dp.calculate_excess_returns(df)
        
        crsp = df.merge(crsp, on=['yyyymm','permno'], how='left')
        crsp = crsp.drop(['siccd', 'me_nyse20','me_nyse50'], axis=1, errors='ignore')
        self.crsp = crsp

    def calc_metrics(self):
        self.cum_log_market_ret = self.cumulative_mkt_ret()
        

    def cumulative_mkt_ret(self):
        cum_log_market_ret = self.mkt_returns.copy()
        # print(self.mkt_returns)
        cum_log_market_ret = cum_log_market_ret.loc[cum_log_market_ret.yyyymm >= 199501].copy()
        cum_log_market_ret['market_ret'] = (np.log(cum_log_market_ret['market_ret']/100+1).cumsum())
        return cum_log_market_ret