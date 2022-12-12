import pandas as pd
import datetime as dt
import numpy as np
import data.data_preprocessing as dp
import statsmodels.api as sm
from portfolios.FF5FM_Mom import FF5FM_Mom
import models.neural_net.metric as metric
from .market_portfolio import MarketPortfolio



class Portfolio():
    def __init__(self, pred_df, rebalancing_frequency = 'monthly', weighting = 'VW', verbose = 1, method='long-short') -> None:
        self.pred_df = pred_df
        self.weighting = weighting
        self.rebalancing_frequency = rebalancing_frequency
        self.verbose = verbose
        self.method = method
        
        self.mkt = MarketPortfolio(verbose=self.verbose)
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

    def _split_predicted_returns(self, use_breakpoints = True, nyse_breakpoints=False):
        rebalancing_months = self._list_rebalancing_months()
        self.rebalancing_df = self.pred_df.loc[self.pred_df['yyyymm'].isin(rebalancing_months)]

        if nyse_breakpoints:
            pass
        elif use_breakpoints:
            deciles = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
            for i,decile in enumerate(deciles):
                if i == 0:
                    breakpoints = self.rebalancing_df.groupby('yyyymm')['predicted_ret'].quantile(.1).reset_index(drop=False).copy()
                    breakpoints = breakpoints.rename({'predicted_ret': "predRet_10"}, axis=1)
                else:
                    breakpoints['predRet_'+str(int(decile*100))] = self.rebalancing_df.groupby('yyyymm')['predicted_ret'].quantile(decile).reset_index(drop=True)
            self.rebalancing_df = self.rebalancing_df.merge(breakpoints, on='yyyymm', how='left')

            conditions =[
                (self.rebalancing_df['predicted_ret'] < self.rebalancing_df['predRet_10']),
                ((self.rebalancing_df['predicted_ret'] >= self.rebalancing_df['predRet_10']) & (self.rebalancing_df['predicted_ret'] < self.rebalancing_df['predRet_20'])),
                ((self.rebalancing_df['predicted_ret'] >= self.rebalancing_df['predRet_20']) & (self.rebalancing_df['predicted_ret'] < self.rebalancing_df['predRet_30'])),
                ((self.rebalancing_df['predicted_ret'] >= self.rebalancing_df['predRet_30']) & (self.rebalancing_df['predicted_ret'] < self.rebalancing_df['predRet_40'])),
                ((self.rebalancing_df['predicted_ret'] >= self.rebalancing_df['predRet_40']) & (self.rebalancing_df['predicted_ret'] < self.rebalancing_df['predRet_50'])),
                ((self.rebalancing_df['predicted_ret'] >= self.rebalancing_df['predRet_50']) & (self.rebalancing_df['predicted_ret'] < self.rebalancing_df['predRet_60'])),
                ((self.rebalancing_df['predicted_ret'] >= self.rebalancing_df['predRet_60']) & (self.rebalancing_df['predicted_ret'] < self.rebalancing_df['predRet_70'])),
                ((self.rebalancing_df['predicted_ret'] >= self.rebalancing_df['predRet_70']) & (self.rebalancing_df['predicted_ret'] < self.rebalancing_df['predRet_80'])),
                ((self.rebalancing_df['predicted_ret'] >= self.rebalancing_df['predRet_80']) & (self.rebalancing_df['predicted_ret'] < self.rebalancing_df['predRet_90'])),
                (self.rebalancing_df['predicted_ret'] >= self.rebalancing_df['predRet_90'])
            ]
            values = [1,2,3,4,5,6,7,8,9,10]

            self.rebalancing_df['decile'] = np.select(conditions, values)

        else:
            self.rebalancing_df['rank'] = self.rebalancing_df.groupby('yyyymm')['predicted_ret'].rank(pct = True)
            self.rebalancing_df['decile'] = np.nan

            for i in range(1,10):
                # self.rebalancing_df.decile[self.rebalancing_df.decile >= float(i)/10] = int(i)+1
                self.rebalancing_df.loc[self.rebalancing_df["rank"] >= float(i)/10, "decile"] = int(i)+1
            
            # print('NaNs in deciles:')
            # print(self.rebalancing_df[pd.isnull(self.rebalancing_df.decile)])
            self.rebalancing_df.loc[pd.isnull(self.rebalancing_df['decile']), 'decile'] = 1 
            
        self.rebalancing_df = self.rebalancing_df.drop(['predicted_ret', 'ret', 'date'], axis=1, errors='ignore')
        
        self.pred_df = self.pred_df.merge(self.rebalancing_df, on=['permno','yyyymm'], how='outer')

        crsp = dp.load_crsp()
        crsp = crsp[['permno','yyyymm', 'me', 'melag']]

        self.pred_df = self.pred_df.merge(crsp, on=['yyyymm','permno'], how='left')
        # Add ffill part for yearly and quarterly rebalance

    def _calculate_weights(self):
        if self.weighting == 'VW':
            weight = pd.DataFrame(self.pred_df.groupby(['yyyymm','decile'])['melag'].sum()).reset_index(drop=False)
            weight = weight.rename(columns={'melag':'total_melag'})
            self.pred_df = self.pred_df.merge(weight, on=['yyyymm', 'decile'], how='inner')
            self.pred_df['weight'] = self.pred_df['melag']/self.pred_df['total_melag']
        elif self.weighting == 'EW':
            weight = pd.DataFrame(self.pred_df.groupby(['yyyymm','decile'])['permno'].count()).reset_index(drop=False) 
            weight = weight.rename(columns={'permno':'count'})
            self.pred_df = self.pred_df.merge(weight, on=['yyyymm', 'decile'], how='inner')
            self.pred_df['weight'] = 1/self.pred_df['count']
            
            # weights_df = self.pred_df.copy()
            # count_df = weights_df.groupby(['yyyymm', 'decile']).size().reset_index().rename({0:'count'}, axis=1)
            # weights_df = weights_df.merge(count_df, on=['yyyymm', 'decile'], how='left')
            # weights_df['weight'] = 1/weights_df['count']
            # weights_df['weighted_ret'] = weights_df['weight'] * weights_df['ret']
            # weights_df = weights_df[['permno', 'yyyymm', 'ret','predicted_ret', 'decile', 'weight', 'weighted_ret']]
            # self.weights_df = weights_df.copy()

        else:
            print('Inserted weighting method is not valid.\nUse EW for equal weighted or VW for value weighted')

    def _calculate_portfolio_returns(self):
        if self.weighting == 'VW':
            self.pred_df = (self.pred_df.sort_values(by=['permno','yyyymm'])).reset_index(drop=True)
            self.pred_df['weighted_ret'] = self.pred_df['ret'] * self.pred_df['weight']
            self.pred_df.loc[self.pred_df['permno'] != self.pred_df['permno'].shift(), 'weighted_ret'] = np.nan
            # self.pred_df.vw_ret[self.pred_df.permno != self.pred_df.permno.shift()] = np.nan
            self.weights_df = self.pred_df.copy()
        
            monthly_vw = self.pred_df.groupby(['yyyymm','decile'])['weighted_ret'].sum()
            monthly_vw = monthly_vw.unstack('decile')
            monthly_vw = monthly_vw.reset_index(drop=False)
            # print('Monthly VW before indexing:')
            # print(monthly_vw.head(2))
            monthly_vw = monthly_vw.loc[1:,:]
            
            # print('Monthly VW after indexing:')
            # print(monthly_vw.head(20))

            monthly_vw['l-s'] = monthly_vw.iloc[:,-1] - monthly_vw.iloc[:,1]
            # Temp, change 
            self.returns = monthly_vw
        
        elif self.weighting == 'EW':
            self.pred_df = (self.pred_df.sort_values(by=['permno','yyyymm'])).reset_index(drop=True)
            self.pred_df['weighted_ret'] = self.pred_df['ret'] * self.pred_df['weight']
            self.pred_df.loc[self.pred_df['permno'] != self.pred_df['permno'].shift(), 'weighted_ret'] = np.nan
            self.weights_df = self.pred_df.copy()
            monthly_ew = self.pred_df.groupby(['yyyymm','decile'])['weighted_ret'].sum()
                
            monthly_ew = monthly_ew.unstack('decile')
            monthly_ew = monthly_ew.reset_index(drop=False)
            monthly_ew = monthly_ew.loc[1:,:]
            # print(monthly_ew)
            self.returns = monthly_ew
            
            
    def regress_on_FF5FM(self):
        """ Taken from old portfolio class, change """
        FFMom = FF5FM_Mom().returns
        """
            Finish implementing the information ratio calculation.
            Regress the long and long-short

        """
        method = self.method
        columns_to_keep = [self.returns.columns[0]]
        columns_to_keep.extend(self.returns.columns[-2:])
        df = self.returns[columns_to_keep]
        if self.verbose:
            print('IR & alpha calculation...')
            # print(df.head())
        df = df.merge(FFMom, on=['yyyymm'], how='left')
        
        
        # On long-short returns        
        X = df.iloc[:, 3:]

        # Column 1 is long returns on max quantile, 
        # Column 2 is long-short returns
        if method == 'long-short':
            y = df.iloc[:,2]
        elif method == 'long':
            y = df.iloc[:,1]
        else:
            print('Error, regression methodology should be either "long-short" or "long"')

        X = sm.add_constant(X)
        lm = sm.OLS(y, X).fit()

        if self.verbose:
            print('Params')
            print(lm.params)
            print('tValues')
            print(lm.tvalues)
        self.alpha = lm.params[0]
        self.t_value_alpha = lm.tvalues[0]
        self.regression_rsquared = lm.rsquared
        if self.verbose:
            try:
                print('Linear Regression summary:')
                print(lm.summary())
            except:
                print("An exception when priting summary")

            try:
                print('\nLinear Regression R-squared:')
                print(lm.rsquared)
            except:
                print("An exception occurred")
        
        year_min = self.returns['yyyymm'].min()//100
        month_min = self.returns['yyyymm'].min()%100

        year_max = self.returns['yyyymm'].max()//100
        month_max = self.returns['yyyymm'].max()%100

        n_months =  ((year_max - year_min) + (month_max-month_min+1)/12)*12 
        self.information_ratio_regression = self.t_value_alpha / np.sqrt(n_months)
        if self.verbose:
            print(f'Informatio Ratio based on FF5FM+Mom regression: {self.information_ratio_regression:.2f} ({method})')

    def _calculate_metrics(self):
        
        self.cum_returns = self.returns.copy()
        for col in self.cum_returns.iloc[:,1:].columns:
            self.cum_returns[col] = (np.log(self.cum_returns[col]/100+1).cumsum())

        T = self.returns['yyyymm'].count()

        if self.verbose:
            print(f'Average return:\n\tLong: {self.returns.iloc[:,-2].mean():.2f}%\tLong-short: {(self.returns.iloc[:,-1]).mean():.2f}%')
            print(f'T is {T}')        


        self.average_long_short_ret = self.returns.iloc[:,-1].mean()
        self.annualized_return_long = np.prod((1+self.returns.iloc[:,-2]/100)**(12/T))-1
        self.annualized_return_long_short = np.prod((1+self.returns.iloc[:,-1]/100)**(12/T))-1
        self.annualized_return_long = self.annualized_return_long * 100
        self.annualized_return_long_short = self.annualized_return_long_short * 100
        
        market_returns = self.mkt.mkt_returns
        returns = self.returns.merge(market_returns, on=['yyyymm'],how='left')
        
        self.annualized_market_return = np.prod((returns.iloc[:,-1]/100+1)**(12/T))-1
        self.alpha_market_long = ((((returns.iloc[:,-3] - returns.iloc[:,-1])/100+1)**(12/T)).prod())-1
        alpha_market_long_short = ((((returns.iloc[:,-2] - returns.iloc[:,-1])/100+1)**(12/T)).prod())-1
                
    
        self.std_dev_long = np.std(self.returns.iloc[:,-2])     
        self.std_dev_long_short = np.std(self.returns.iloc[:,-1])

        # self.sharpe_ratio_long = self.annualized_return_long/self.std_dev_long
        # self.sharpe_ratio_long_short = self.annualized_return_long_short/self.std_dev_long_short
        
        # IR = metric.information_ratio(self.returns)

        self.long_short_sharpe_ratio = (self.average_long_short_ret / self.std_dev_long_short)

        self.turnover = metric.turnover(self.weights_df)
        # print(self.returns.iloc[:,-1].min()) 
        self.max1MLoss_long_short = self.returns.iloc[:,-1].min() * -1
        # print(f'Max1MLoss: {self.max1MLoss_long_short}')
        # print(self.returns.head())
        self.maxDD_long_short = metric.max_drawdown_long_short(self.returns)
        
        if self.verbose:
            print(f'Annualized returns:\n\tLong: {self.annualized_return_long:.2f}%\tLong-short:{self.annualized_return_long_short:.2f}%')
            print(f'Annualized Market Returns {self.annualized_market_return:.2f}%')
            print(f'Standard deviation on returns:\n\tLong: {self.std_dev_long:.2f}\tLong-short:{self.std_dev_long_short:.2f}')
            print(f'Alpha over market:\tLong:{self.alpha_market_long:.2f}%\tLong-short:{alpha_market_long_short:.2f}%')
            # print(f'Sharpe ratio:\n\tLong: {self.sharpe_ratio_long:.2f}\tLong-short:{self.sharpe_ratio_long_short:.2f}')

            try:
                print('Average Return per decile')
                print(self.returns.iloc[:,1:].mean(axis=0))
            except:
                print('Could not calculate avg return per decile')
        

    def plot_cumulative_returns(self, path):
        from cycler import cycler
        import matplotlib.pyplot as plt
        
        mkt_ret = self.mkt.mkt_returns

        cum_returns = self.cum_returns.merge(mkt_ret, on=['yyyymm'], how='left')
        cum_returns.iloc[:,-1] = (np.log(cum_returns.iloc[:,-1]/100+1).cumsum())

        cum_returns['date'] = cum_returns['yyyymm'].apply(lambda x: dt.datetime.strptime(str(x), '%Y%m'))
        # print('Self.cum_returns columns are:')
        # print(self.cum_returns.columns)
        linestyle_cycler = (cycler('color', ['deepskyblue','coral','magenta','royalblue', 'red','lime', 'crimson', 'cyan','springgreen','teal','gray','darkorange']) +
                            cycler('linestyle',['-','--',':','-.',':','-','-.','--','-',':','-.','--']))


        fig = plt.figure(figsize=(15, 5))
        ax = plt.gca()
        ax.set_prop_cycle(linestyle_cycler)
        plt.plot(cum_returns['date'], cum_returns.iloc[:,1:-1]) # plt.plot(l, ret.iloc[:,1:])
        plt.ylabel('Cumulative Log returns')
        plt.legend(cum_returns.iloc[:,1:-1].columns)
        plt.tight_layout()
        try:
            plt.savefig(path)
        except:
            pass

