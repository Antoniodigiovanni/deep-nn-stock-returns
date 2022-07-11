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
  #print(f'Correct: {torch.sum(abs_delta <= max_allow).numpy()} on {yhat.shape[0]}')
  return acc

def calc_accuracy_and_predict(model, data, pct):
  # assumes model.eval()


  with torch.no_grad():
    output = model(data['X'])
    print(type(data['permno']))
    prediction = {'permno': data['permno'].squeeze().tolist(), 'yyyymm':data['yyyymm'].squeeze().tolist(), 'predicted_ret':output.squeeze().tolist()}
    acc = accuracy(data['Y'], output, pct)

  return acc, prediction

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