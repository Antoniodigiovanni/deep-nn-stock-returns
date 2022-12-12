import pandas as pd
from os import listdir
from os.path import isfile, join
import pandas as pd
import torchmetrics as tm
import numpy as np
import torch
import models.neural_net.metric as metric
from portfolios.portfolio_new import Portfolio
import data.data_preprocessing as dp

predicted_ret_path = '/home/ge65cuw/thesis/saved/final_results/predicted_returns/'
predicted_ret_files = [f for f in listdir(predicted_ret_path) if isfile(join(predicted_ret_path, f))]

# Measures to calculate:
df_columns = columns=[
        'trial_id',
        'oosSpearman',
        'MDA', 
        'Turnover_long_VW', 
        'Turnover_short_VW', 
        'Turnover_long-short_VW', 
        'Max1MLoss_VW',
        'MaxDD_VW', 
        'Avg_ret_VW', 
        'Std_VW', 
        'Sharpe_ratio_VW', 
        'FF5_Mom_STRev_alpha_VW',
        'alpha_t-stat_VW',
        'regression_R2_VW', 
        'regression_Information_ratio_VW',
        'Turnover_long_EW', 
        'Turnover_short_EW', 
        'Turnover_long-short_EW', 
        'Max1MLoss_EW',
        'MaxDD_EW', 
        'Avg_ret_EW', 
        'Std_EW', 
        'Sharpe_ratio_EW', 
        'FF5_Mom_STRev_alpha_EW',
        'alpha_t-stat_EW', 
        'regression_R2_EW', 
        'regression_Information_ratio_EW']

import warnings
# print('Remove MaxDD printing and implement logic to not save networks which fail at creating portfolios')
results_df = pd.DataFrame(columns=df_columns)
i=0
for file in predicted_ret_files:
        i+=1
        trial_id = file.split('_')[0] 
        results_dict = {}
        df = pd.read_csv(predicted_ret_path + file, index_col=0)
        for column in df_columns:
                results_dict[column] = np.nan
        
        results_dict['trial_id'] = trial_id
        # if 'ret' not in df.columns:
        #         crsp = dp.load_crsp()
        #         crsp = crsp[['permno','yyyymm','ret']]
        #         df = df.merge(crsp, on=['permno','yyyymm'], how='left')
        with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results_dict['oosSpearman'] = metric.calc_spearman(torch.tensor(df['predicted_ret']), torch.tensor(df['ret'])).item()

        results_dict['MDA'] = metric.mean_directional_accuracy(df['predicted_ret'], df['ret'])
        try:
                portfolioVW = Portfolio(df, weighting='VW', verbose = 0)
                results_dict['Turnover_long_VW'] = portfolioVW.turnover['long']
                results_dict['Turnover_short_VW'] = portfolioVW.turnover['short']
                results_dict['Turnover_long-short_VW'] = portfolioVW.turnover['long-short']
                results_dict['Max1MLoss_VW'] = portfolioVW.max1MLoss_long_short
                results_dict['MaxDD_VW'] = portfolioVW.maxDD_long_short
                results_dict['Avg_ret_VW'] = portfolioVW.average_long_short_ret
                results_dict['Std_VW'] = portfolioVW.std_dev_long_short
                results_dict['Sharpe_ratio_VW'] = portfolioVW.long_short_sharpe_ratio
                results_dict['FF5_Mom_STRev_alpha_VW'] = portfolioVW.alpha
                results_dict['alpha_t-stat_VW'] = portfolioVW.t_value_alpha
                results_dict['regression_R2_VW'] = portfolioVW.regression_rsquared
                results_dict['regression_Information_ratio_VW'] = portfolioVW.information_ratio_regression
        except Exception as e:
                print('Error in Value Weighted Portfolio')
                print(e)
        try:
                portfolioEW = Portfolio(df, weighting='EW', verbose = 0)
                results_dict['Turnover_long_EW'] = portfolioEW.turnover['long']
                results_dict['Turnover_short_EW'] = portfolioEW.turnover['short']
                results_dict['Turnover_long-short_EW'] = portfolioEW.turnover['long-short']
                results_dict['Max1MLoss_EW'] = portfolioEW.max1MLoss_long_short
                results_dict['MaxDD_EW'] = portfolioEW.maxDD_long_short
                results_dict['Avg_ret_EW'] = portfolioEW.average_long_short_ret
                results_dict['Std_EW'] = portfolioEW.std_dev_long_short
                results_dict['Sharpe_ratio_EW'] = portfolioEW.long_short_sharpe_ratio
                results_dict['FF5_Mom_STRev_alpha_EW'] = portfolioEW.alpha
                results_dict['alpha_t-stat_EW'] = portfolioEW.t_value_alpha
                results_dict['regression_R2_EW'] = portfolioEW.regression_rsquared
                results_dict['regression_Information_ratio_EW'] = portfolioEW.information_ratio_regression
        except Exception as e:
                print('Error with Equal Weighted Portfolio')
                print(e)
        print(f'Trial no.{i} [of {len(predicted_ret_files)}] completed')     
        results = pd.DataFrame(results_dict, index=[0])
        results_df = pd.concat([results_df, results], ignore_index=True)
        results_df.to_csv('/home/ge65cuw/thesis/saved/final_results/results_df_temp.csv')

results_df.to_csv('/home/ge65cuw/thesis/saved/final_results/results_df.csv')

