import torch
import pandas as pd
import numpy as np


def accuracy(output, target, pct):
    with torch.no_grad():
      
        assert output.shape[0] == len(target)
        n_correct = 0
        n_wrong = 0
        #correct = 0
        #print(f'output shape is: {output.shape}\ntarget shape is:{target.shape}')
        for i in range(len(target)):
          abs_delta = np.abs(output[i] - target[i])
          #print(f'abs_delta: {abs_delta.item()}')
          max_allow = np.abs(pct * target[i])
          if abs_delta < max_allow:
            n_correct +=1
          
          else:
            n_wrong += 1
          #print(f'Target: {target} | prediction: {output}')

        #acc = (n_correct * 1.0) / (n_correct + n_wrong)
        #correct += torch.sum(output == target).item()
        return n_correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

# After this line the functions are the ones I am currently using

def calc_accuracy(model, data, pct):
  # assumes model.eval()
  # percent correct within pct of true house price
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