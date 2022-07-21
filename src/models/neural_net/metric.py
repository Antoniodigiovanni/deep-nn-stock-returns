import torch
import pandas as pd
import numpy as np
from portfolios.PortfolioCreation import Portfolio
import config

def accuracy(truth, yhat, pct):
  
  yhat = yhat.reshape(yhat.shape[0])
  truth = truth.reshape(truth.shape[0])
  abs_delta = np.abs(yhat-truth)
  max_allow = np.abs(pct * truth)

  
  correct = torch.sum(abs_delta  <= max_allow).item()
  return correct

def calc_accuracy_and_predict(model, data, pct):
  # assumes model.eval()


  with torch.no_grad():
    output = model(data['X'])
    #print(type(data['permno']))
    prediction = {'permno': data['permno'].squeeze().tolist(), 'yyyymm':data['yyyymm'].squeeze().tolist(), 'predicted_ret':output.squeeze().tolist()}
    correct = accuracy(data['Y'], output, pct)

  return correct, prediction


def r2_metric_calculation(df):
    crsp = pd.read_csv(config.paths['CRSPretPath'])
    crsp.drop(['date'], axis=1, inplace=True)
    df_std = df.merge(crsp, on=['yyyymm', 'permno'], how='left')
    
    df_std['rank'] = df_std.groupby('yyyymm')['melag'].rank(method= 'max', ascending=False)
    df_top = df_std.loc[df_std['rank'] <= 1000].copy()
    df_top.drop(['rank', 'melag'], axis=1, inplace=True)

    r2_top = calc_r2(df_top)
    del df_top

    df_std['rank'] = df_std.groupby('yyyymm')['melag'].rank(method= 'max', ascending=True)
    df_bottom = df_std.loc[df_std['rank'] <= 1000].copy()
    df_bottom.drop(['rank', 'melag'], axis=1, inplace=True)

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

def calc_portfolio_alpha():
  """
    This function returns the portfolio alpha when the Long-Short returns are regressed on the Fama French
    5 Factors Model + Momentum

    Does it make sense to calculate the portfolio alpha during training? Maybe it is better to calculate
    the accuracy only, and leave the loss function as measure for NNI.

    Currently, the function loads a prediction df, this because the function was used only after testing and making
    predictions. This has to be changed a bit. In order to take it from here.
    
  """
  
  portfolio = Portfolio(config.n_cuts, config.rebalancing_frequency, config.weighting)

  return portfolio.alpha, portfolio.information_ratio