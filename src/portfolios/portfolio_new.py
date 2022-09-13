import pandas as pd
import datetime as dt
import numpy as np
import config.config as config
import data.data_preprocessing as dp
import statsmodels.api as sm
from portfolios.FF5FM_Mom import FF5FM_Mom


class Portfolio():
    def __init__(self, pred_df, rebalancing_frequency = 'monthly', weighting = 'VW') -> None:
        self.pred_df = pred_df
        self.weighting = weighting
        self.rebalancing_frequency = rebalancing_frequency
    
        self._split_predicted_returns()
        self._calculate_weights()
        self._calculate_portfolio_returns()
        self.regress_on_FF5FM()
        self._calculate_metrics()


    def _list_rebalancing_months(self):
        rebalancing_map = {
            'yearly': 'A-JUN', 
            'quarterly': 'Q', 
            'monthly':'M'}

        rebalancing_months = pd.period_range(
            start=dt.datetime.strptime(str(self.pred_df.yyyymm.min()), '%Y%m'),
            end=dt.datetime.strptime(str(self.pred_df.yyyymm.max()), '%Y%m'),
            freq=rebalancing_map[self.rebalancing_frequency]
            ).strftime('%Y%m').astype(int).tolist()
        
        return rebalancing_months

    def _split_predicted_returns(self, nyse_breakpoints=False):
        rebalancing_months = self._list_rebalancing_months()
        
        if nyse_breakpoints:
            pass
        else:
            self.rebalancing_df = self.pred_df.loc[self.pred_df['yyyymm'].isin(rebalancing_months)]
            self.rebalancing_df['rank'] = self.rebalancing_df.groupby('yyyymm')['predicted_ret'].rank(pct = True)
            self.rebalancing_df['decile'] = np.nan

            for i in range(1,10):
                # self.rebalancing_df.decile[self.rebalancing_df.decile >= float(i)/10] = int(i)+1
                self.rebalancing_df.loc[self.rebalancing_df["rank"] >= float(i)/10, "decile"] = int(i)+1
            
            # print('NaNs in deciles:')
            # print(self.rebalancing_df[pd.isnull(self.rebalancing_df.decile)])
            self.rebalancing_df.loc[pd.isnull(self.rebalancing_df['decile']), 'decile'] = 1 
            
        self.rebalancing_df = self.rebalancing_df.drop(['predicted_ret', 'ret', 'date', 'melag'], axis=1, errors='ignore')

        self.pred_df = self.pred_df.merge(self.rebalancing_df, on=['permno','yyyymm'], how='outer')
        print(self.pred_df.head())

        crsp = dp.load_crsp()
        crsp = crsp[['permno','yyyymm', 'me']]

        self.pred_df = self.pred_df.merge(crsp, on=['yyyymm','permno'], how='left')
        # Add ffill part for yearly and quarterly rebalance

    def _calculate_weights(self):
        if self.weighting == 'VW':
            weight = pd.DataFrame(self.pred_df.groupby(['yyyymm','decile'])['me'].sum()).reset_index(drop=False)
            weight = weight.rename(columns={'me':'total_me'})
            self.pred_df = self.pred_df.merge(weight, on=['yyyymm', 'decile'], how='inner')
            self.pred_df['weight'] = self.pred_df['me']/self.pred_df['total_me']
        elif self.weighting == 'EW':
            monthly_ew = self.pred_df.groupby(['yyyymm','decile'])['ret'].mean()
            monthly_ew.unstack('decile')
            monthly_ew = monthly_ew.reset_index(drop=False)
            print(monthly_ew.head())
        else:
            print('Inserted weighting method is not valid.\nUse EW for equal weighted or VW for value weighted')

    def _calculate_portfolio_returns(self):
        if self.weighting == 'VW':
            self.pred_df = (self.pred_df.sort_values(by=['permno','yyyymm'])).reset_index(drop=True)
            self.pred_df['vw_ret'] = self.pred_df['ret'] * self.pred_df['weight'].shift()
            self.pred_df.loc[self.pred_df['permno'] != self.pred_df['permno'].shift(), 'vw_ret'] = np.nan
            # self.pred_df.vw_ret[self.pred_df.permno != self.pred_df.permno.shift()] = np.nan
            monthly_vw = self.pred_df.groupby(['yyyymm','decile'])['vw_ret'].sum()
            monthly_vw = monthly_vw.unstack('decile')
            monthly_vw = monthly_vw.reset_index(drop=False)
            # print('Monthly VW before indexing:')
            # print(monthly_vw.head(2))
            monthly_vw = monthly_vw.loc[1:,:]
            
            # print('Monthly VW after indexing:')
            # print(monthly_vw.head(20))

            print(monthly_vw.iloc[:,-1].mean())
            print((monthly_vw.iloc[:,-1]-monthly_vw.iloc[:,1]).mean())
            monthly_vw['l-s'] = monthly_vw.iloc[:,-1] - monthly_vw.iloc[:,1]
            # Temp, change 
            self.returns = monthly_vw
            self.returns.to_csv('port_returns.csv')

    def regress_on_FF5FM(self):
        """ Taken from old portfolio class, change """
        FFMom = FF5FM_Mom().returns
        """
            Finish implementing the information ratio calculation.
            Regress the long and long-short

        """
        self.verbose = True
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
        self.information_ratio_regression = self.t_value_alpha / np.sqrt(n_months)

    def _calculate_metrics(self):
        self.cum_returns = self.returns.copy()
        for col in self.cum_returns.iloc[:,1:].columns:
            self.cum_returns[col] = (np.log(self.cum_returns[col]/100+1).cumsum())

        print('Cumulative Returns')
        print(self.cum_returns.tail())

    def plot_cumulative_returns(self, path):
        from cycler import cycler
        import matplotlib.pyplot as plt
        from .market_portfolio import MarketPortfolio

        mkt = MarketPortfolio()
        cum_mkt_ret = mkt.cumulative_mkt_ret()

        self.cum_returns.merge(cum_mkt_ret, on=['yyyymm'], how='left')
        self.cum_returns['date'] = self.cum_returns['yyyymm'].apply(lambda x: dt.datetime.strptime(str(x), '%Y%m'))

        linestyle_cycler = (cycler('color', ['deepskyblue','coral','magenta','royalblue', 'red','lime', 'crimson', 'cyan','springgreen','teal','gray','darkorange']) +
                            cycler('linestyle',['-','--',':','-.',':','-','-.','--','-',':','-.','--']))


        fig = plt.figure(figsize=(15, 5))
        ax = plt.gca()
        ax.set_prop_cycle(linestyle_cycler)
        plt.plot(self.cum_returns['date'], self.cum_returns.iloc[:,1:-1]) # plt.plot(l, ret.iloc[:,1:])
        plt.ylabel('Cumulative Log returns')
        plt.legend(self.cum_returns.iloc[:,1:-1].columns)
        plt.tight_layout()
        plt.savefig(path)
        