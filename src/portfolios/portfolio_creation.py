import pandas as pd
import os, sys
from portfolio_functions import *


currentPath = os.getcwd()#os.path.dirname(sys.argv[0])
dataPath = (currentPath+'/../../data')
ProcessedDataPath = (dataPath + '/processed')

pred_df = pd.read_csv(ProcessedDataPath+'/predicted_ret.csv', index_col=0)
crsp = pd.read_csv(ProcessedDataPath+'/../external/crspmret.csv')

n_cuts = 3

# Still have to check cut vs. qcut
pred_df = pred_df.set_index(['yyyymm', 'permno'])\
    .groupby('yyyymm')['predicted_ret']\
        .apply(lambda x: pd.cut(x, bins=n_cuts, labels=False))\
            .reset_index(name='bin')

df = crsp.merge(pred_df, on=['permno', 'yyyymm'], how='right')
del [crsp, pred_df]

df = calculate_portfolio_weights(df)
df = calculate_portfolio_monthly_return(df)

df.to_csv(currentPath+'/portfolioReturns.csv')
print(currentPath)
