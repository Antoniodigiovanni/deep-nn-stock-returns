#import test
#from portfolios import portfolio_creation
#from portfolios import fama_french_plus_momentum
from base import set_random_seed
from portfolios.portfolio import Portfolio

#################################################################################################
#                                                                                               #
#                                      MODEL TRAINING                                           #
#                                                                                               #
#################################################################################################

portfolio = Portfolio().information_ratio()
print('Portfolio:')
print(portfolio)
#import train

### MODEL TUNING ###

#import tuning.neural_architecture_search
#print('Done')

#################################################################################################
#                                                                                               #
#                                      POOLED OLS REGRESSION                                    #
#                                                                                               #
#################################################################################################
""" from models.ols.pooled_ols import PooledOLS
import config
import pandas as pd

df = pd.read_csv(config.paths['ProcessedDataPath']+'/dataset.csv', index_col=0)
x_list = list(df.drop(['permno','yyyymm','ret'], axis=1).columns)
y = 'ret'
pooled_ols = PooledOLS(df, x_list, y)
pooled_ols.regress()
 """