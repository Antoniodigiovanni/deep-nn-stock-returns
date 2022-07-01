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

  acc = torch.sum(abs_delta <= max_allow).numpy() / yhat.shape[0]
  print(f'Correct: {torch.sum(abs_delta <= max_allow).numpy()} on {yhat.shape[0]}')
  return acc

def calc_accuracy(model, data, pct):
  # assumes model.eval()

  # Percent correct within pct of true returns
  n_correct = 0; n_wrong = 0

  for i in range(len(data['X'])):
    with torch.no_grad():
      output = model (data['X'][i])
      
      #prediction = pd.DataFrame({'permno': data['permno'][i], 'yyyymm':data['yyyymm'][i], 'ret':data['Y'][i], 'predicted_ret':output})
      prediction = {'permno': data['permno'][i], 'yyyymm':data['yyyymm'][i], 'ret':data['Y'][i], 'predicted_ret':output}
      abs_delta = np.abs(output - data['Y'][i])

      max_allow = np.abs(pct * data['Y'][i])
      if abs_delta < max_allow:
        n_correct +=1
        
      else:
        n_wrong += 1
        #print(f'i: {i} | target: {target} | prediction: {output}')

  acc = (n_correct * 1.0) / (n_correct + n_wrong)
  stats = {'Correct': n_correct, 'Wrong': n_wrong}

  #return acc, prediction
  return acc, stats, prediction

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