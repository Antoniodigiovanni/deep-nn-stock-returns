from data.dataset import BaseDataset
from models.ols.pooled_ols import PooledOLS
from portfolios.portfolio_new import Portfolio
import models.neural_net.metric as metric
import pandas as pd

dataset = BaseDataset(ret_scaling_method='standard')
df = dataset.df
start_factors_idx = df.columns.tolist().index('AbnormalAccruals')
x_list = df.columns.tolist()[start_factors_idx:]
ols = PooledOLS(x_list=x_list, y='ret', df=df)

r2 = metric.calc_r2(ols.prediction_df)
print(f'\n\nOOS R2 is: {r2}\n\n')
mda = metric.mean_directional_accuracy(ols.prediction_df['predicted_ret'], ols.prediction_df['ret'])
print(f'OOS Mean Directional Accuracy is: {mda}\n\n')

portfolio = Portfolio(ols.prediction_df, weighting='VW')
portfolio.plot_cumulative_returns('/home/ge65cuw/thesis/saved/linear/linear_plot.png')
ols.prediction_df.to_csv('/home/ge65cuw/thesis/saved/linear/predicted_ret.csv')