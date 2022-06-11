#import test
#from portfolios import portfolio_creation
#from portfolios import fama_french_plus_momentum
from sqlalchemy import except_
from base import set_random_seed
from portfolios.portfolio import Portfolio
from data.base_dataset import BaseDataset
#################################################################################################
#                                                                                               #
#                                      MODEL TRAINING                                           #
#                                                                                               #
#################################################################################################

data = BaseDataset(mode='test')
try:
    print(data.crsp.shape)
except:
    print('CRSP error')
try:
    print(data.X_train.shape)
except:
    print('Train X error')
try:
    print(data.X_test.columns)
except:
    print('Test X error')

data.load_dataset_in_memory()
print('Printing CRSP after calling method to load in memory')
try:
    print(data.crsp.columns)
except:
    print('CRSP error')


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