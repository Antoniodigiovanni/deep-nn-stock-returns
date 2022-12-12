import torch
import pandas as pd
import numpy as np
from portfolios.market_portfolio import MarketPortfolio
from portfolios.FF5FM_Mom import FF5FM_Mom
import config

def accuracy(truth, yhat, pct):
  
  # yhat = yhat.reshape(yhat.shape[0])
  # truth = truth.reshape(truth.shape[0])
  abs_delta = torch.abs(yhat-truth)
  max_allow = torch.abs(pct * truth)

  
  correct = torch.sum(abs_delta  <= max_allow).item()
  return correct

def calc_accuracy_and_predict(model, data, pct):
  model.eval()

  with torch.no_grad():
    output = model(data['X'])
    prediction = {
      'permno': data['permno'].squeeze().tolist(), 
      'yyyymm':data['yyyymm'].squeeze().tolist(), 
      'predicted_ret':output.squeeze().tolist()}
    
    correct = accuracy(data['Y'], output, pct)

  return correct, prediction


def r2_metric_calculation(pred_df):
    df = pred_df.copy()
    if 'ret' in df.columns:
      df.drop('ret', axis=1, inplace=True, errors='ignore')
      print('RET is dropped')
    crsp = pd.read_csv(config.paths['CRSPretPath'])
    
    crspinfo = pd.read_csv(config.paths['CRSPinfoPath'])
    crsp = crsp.merge(crspinfo[['permno','yyyymm','me']], on=['permno','yyyymm'], how='left')
    
    
    crsp.drop(['date'], errors='ignore',axis=1, inplace=True)
    crsp = crsp[['yyyymm', 'permno', 'me', 'ret']]
    df_std = df.merge(crsp, on=['yyyymm', 'permno'], how='left')
    
    df_std['rank'] = df_std.groupby('yyyymm')['me'].rank(method= 'max', ascending=False)
    df_top = df_std.loc[df_std['rank'] <= 1000].copy()
    df_top.drop(['rank', 'me'], axis=1, inplace=True)

    r2_top = calc_r2(df_top)
    del df_top

    df_std['rank'] = df_std.groupby('yyyymm')['me'].rank(method= 'max', ascending=True)
    df_bottom = df_std.loc[df_std['rank'] <= 1000].copy()
    df_bottom.drop(['rank', 'me'], axis=1, inplace=True)

    r2_bottom = calc_r2(df_bottom)
    del df_bottom

    
    df_std.drop('rank', axis=1, inplace=True)

    r2 = calc_r2(df_std)
    del df_std

    r2_dict = {'R2': r2, 'R2_top_1000': r2_top, 'R2_bottom_1000': r2_bottom}
    return r2_dict

def calc_r2(df):
    num = sum((df['ret'] - df['predicted_ret'])**2)
    den = sum(df['ret']**2)
    r2 = 1-(num/den)
    
    return r2

def alternative_r2(df):
  num = sum((df['ret'] - df['predicted_ret'])**2)
  den = sum((df['ret'] - np.mean(df['ret']))**2)

  r2 = 1-(num/den)
  return r2

def r2_in_training(prediction, target):
  num = sum((target - prediction)**2)
  den = sum(target**2)
  r2 = 1-(num/den)
  
  return r2
  

def normal_r2_calculation(pred_df):
  """
    R2 is calculated as 1-(unexplained variation/total variation)
    unexplained variation is the sum over i of (y_i - yPred_i)**2.
    The total variation is the sum over i of the (y_i - the mean of the depeendent variable)**2
  """

  SSR = np.sum((pred_df['ret'] - pred_df['predicted_ret'])**2)

  mean_ret = np.mean(pred_df['ret'])
  SST = np.sum((pred_df['ret'] - mean_ret)**2)

  r2 = 1-(SSR/SST)

  annualized_r2 = r2*12
  
  # Try to groupby year and then annualize (either with *12 or **12) 
  # and then take the mean

  # r2_dict = {'r2': r2, 'annualized_r2': annualized_r2}
  # return r2_dict

  return r2

def information_ratio(portfolio_returns):
    mp = MarketPortfolio()
    # mp.market_ret_calculation()
    mkt_ret = mp.mkt_returns[['yyyymm','market_ret']]

    print(mkt_ret)
    portfolio_cols = portfolio_returns.iloc[:,1:].columns.tolist()
    portfolio_returns = portfolio_returns.merge(mkt_ret, how='left', on=['yyyymm'])

    portfolio_diff_rets = portfolio_returns.copy()
    IR = {}
    for col in portfolio_cols:
        IR[col] = (
            (portfolio_diff_rets[col].mean() - portfolio_diff_rets['market_ret'].mean())
            / np.std(portfolio_diff_rets[col] - portfolio_diff_rets['market_ret'])
            )
    return IR

def calc_sharpe_ratio(df, already_excess_returns=False):
  RF = FF5FM_Mom().RF
  # RF['RF'] = RF['RF'] / 100
  
  SR = {}
  for column in df.iloc[:,1:].columns:
    if already_excess_returns == False:
      df[column] = df[column] - RF['RF']
    mean_ret = df[column].mean()
    std_dev = df[column].std()
    SR[column] = mean_ret/std_dev

  return SR

def calc_spearman(preds, target):
  from torchmetrics import SpearmanCorrCoef
  spearman = SpearmanCorrCoef()

  return spearman(preds, target)

def mean_directional_accuracy(preds, target):
  size = preds.shape[0]
  mda = np.sum(np.sign(target) == np.sign(preds))/size
  
  return mda
  # return np.mean((np.sign(actual[1:]-actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))

def max_drawdown_long_short(portfolio_returns):
  cumulative_returns = (portfolio_returns.iloc[:,1:]/100+1).cumprod()-1
  highwatermarks = cumulative_returns.cummax()

  drawdowns = (1 + highwatermarks)/(1 + cumulative_returns) - 1

  max_dd_long_short = drawdowns.iloc[:,-1].max()
  #print(f'Max_DD: {max_dd_long_short}')
  
  return max_dd_long_short

def turnover(weights_df):
  df = weights_df[['permno', 'yyyymm','ret','decile','weight', 'weighted_ret']].copy()
  df = df.sort_values(by=['permno','yyyymm'])
  df['ret'] = df['ret']/100
  turnover_long, turnover_short, turnover_ls = np.nan, np.nan, np.nan

  # Select portfolio
  df = df.loc[(df['decile'] == 10) | (df['decile'] == 1)]

  # Calculate needed metrics
  df['w_t+1'] = df.groupby('permno')['weight'].shift(-1)
  df['ret_t+1'] = df.groupby('permno')['ret'].shift(-1)
  df['w_t-times-r_t+1'] = df.groupby(['permno','decile'], as_index=False).apply(lambda x: x['weight']*x['ret_t+1']).reset_index(drop=True)

  # Calculate the sum over all stocks in the denominator
  sum_over_j = df.groupby('yyyymm', as_index=False)['w_t-times-r_t+1'].sum()
  sum_over_j = sum_over_j.rename({'w_t-times-r_t+1': 'sum_over_j'}, axis=1)
  df = df.merge(sum_over_j, on=['yyyymm'])
  df['sum_over_j'] = df['sum_over_j'] + 1

  long_df = df.loc[df['decile'] == 10].copy()
  turnover_long = np.abs(long_df['w_t+1'] - ((long_df['weight'] * (1+long_df['ret_t+1'])/long_df['sum_over_j']))).sum() 
  # Final calculation of turnover
  short_df = df.loc[df['decile'] == 1].copy()
  turnover_short = np.abs(short_df['w_t+1'] - ((short_df['weight'] * (1+short_df['ret_t+1'])/short_df['sum_over_j']))).sum() 
  turnover_ls = turnover_long + turnover_short

  return {'long': turnover_long, 'short': turnover_short, 'long-short': turnover_ls}