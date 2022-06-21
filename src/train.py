import torch
import torch.nn as nn
import numpy as np
from models.neural_net.model import RegressionNet
import models.neural_net.metric as module_metric
from data.custom_dataset import CustomDataset
from data.data_preprocessing import *
from torch.utils.data import DataLoader
import torch.optim as optim
from trainer.trainer import *
import data.data_preprocessing as dp
from asyncio import create_subprocess_exec
import sys,os
import config

class ExpandingWindowTraining():
    def __init__(self, dataset, window='yearly') -> None:
        self.starting_year = dataset.yyyymm.min()
        self.window = window





"""# Categorical_cols should not be only in the following variable, as the encoding should be done in general
if os.path.exists(config.paths['ProcessedDataPath'] + '/dataset.csv') == False or config.ForcePreProcessing == True:
    print('Pre-processing data...')
   
    crsp, categorical_cols = dp.prepare_data(
        config.paths['CRSPretPath'],
        config.paths['CRSPinfoPath'],
        config.paths['FFPath'],
        config.paths['SignalsPath'],
        config.paths['ProcessedDataPath']
    )

    print(f'Pre-processing complete, the dataset has dimensions: {crsp.shape}')
    
else:
    print('Datased found in memory, loading...')
    crsp = pd.read_csv(config.paths['ProcessedDataPath']+'/dataset.csv', index_col=0)
    


# Abstract
if os.path.exists(config.currentPath +'/../../saved/models/RegressionNet_model.pt') and config.ForceTraining == False:
    print('Saved model found, breaking execution')
    sys.exit()


train, val = split_data_train_val(crsp)
X_train, Y_train = sep_target(train)
X_val, Y_val = sep_target(val)

del[train, val]

train = CustomDataset(X_train, Y_train)
del [X_train, Y_train]

val = CustomDataset(X_val, Y_val)
del[X_val, Y_val]


del crsp

# Should shuffle be set to true in my case? Check documentation
train_loader = DataLoader(train, batch_size=config.batch_size_train, num_workers=2)
val_loader = DataLoader(val, batch_size=config.batch_size_validation, num_workers=2)



n_inputs = train.data.shape[1]
model = RegressionNet(n_inputs).to(config.device)

optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

trainer = NeuralNetTrainer(model, train_loader, val_loader, optimizer).train()"""

""" tb = SummaryWriter()

for epoch in range(config.n_epochs):
    epoch_loss = train_one_epoch(model, train_loader, optimizer, config.loss_fn, config.device)
    validation_loss = validate_one_epoch(model, val_loader, optimizer, config.loss_fn, config.device)  
  
    if epoch % config.ep_log_interval == 0:
        print(f'Epoch n.{epoch} | Loss: {epoch_loss}')
        print(f'Validation Loss: {validation_loss}')

    tb.add_scalar("Training Loss", epoch_loss, epoch)
    tb.add_scalar("Validation Loss", validation_loss, epoch)
    tb.flush()
print(f'\nFinal Training Loss: {epoch_loss} at epoch n. {epoch+1}')
print(f'\nFinal Validation Loss: {validation_loss} at epoch n. {epoch+1}')
 """
# Save final model (entire model)
print("\nSaving trained model")
fn = config.paths['modelsPath']+'/RegressionNet_model.pt'
torch.save(model, fn)
