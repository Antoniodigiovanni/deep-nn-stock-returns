import config.config as config
import pandas as pd
import data.data_preprocessing as dp
import numpy as np

class MarketPortfolio():
    def __init__(self) -> None:
        self.__load_crsp()
        self.market_ret_calculation()
        


    
    def market_ret_calculation(self, weighting = 'VW'):
        self.weighting = weighting

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

    def __load_crsp(self):
        # Load CRSP
        crsp = dp.load_crsp()
        crsp = dp.remove_microcap_stocks(crsp)
        crsp = dp.filter_exchange_code(crsp)
        crsp = dp.filter_share_code(crsp)
        crsp = dp.calculate_excess_returns(crsp)
        
        crsp = crsp.drop(['siccd', 'me_nyse20','me_nyse50'], axis=1)
        self.crsp = crsp

    def calc_metrics(self):
        self.cum_log_market_ret = self.cumulative_mkt_ret()
        

    def cumulative_mkt_ret(self):
        cum_log_market_ret = self.mkt_returns.copy()
        print(self.mkt_returns)
        cum_log_market_ret = cum_log_market_ret.loc[cum_log_market_ret.yyyymm >= 199501].copy()
        cum_log_market_ret['market_ret'] = (np.log(cum_log_market_ret['market_ret']/100+1).cumsum())
        return cum_log_market_ret