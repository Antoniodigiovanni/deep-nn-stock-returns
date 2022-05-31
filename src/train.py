import torch
import torch.nn as nn
import numpy as np
from model.model import RegressionNet
import model.metric as module_metric
from data.custom_dataset import CustomDataset
from data.data_preprocessing import *
from torch.utils.data import DataLoader
import torch.optim as optim
from trainer.trainer import *
import data.data_preprocessing as dp
from asyncio import create_subprocess_exec
import sys,os
from torch.utils.tensorboard import SummaryWriter
import config

#currentPath = os.path.dirname(sys.argv[0]) # does not work in Jupyter notebooks and linux shell, use os.getcwd() there instead
""" currentPath = os.getcwd()
dataPath = (currentPath+'/../data')
# Alternative is: os.path.dirname(__file__)

CRSPretPath = (dataPath + '/external/crspmret.csv')
CRSPinfoPath=(dataPath+'/external/crspminfo.csv')
FFPath = (dataPath + '/external/FamaFrenchData.csv')
SignalsPath = (dataPath + '/external/signed_predictors_all_wide.csv')
ProcessedDataPath = (dataPath + '/processed')
ForcePreProcessing = True
ForceTraining = True
 """
currentPath = config.currentPath
dataPath = config.dataPath
# Alternative is: os.path.dirname(__file__)

CRSPretPath = config.paths['CRSPretPath']
CRSPinfoPath= config.paths['CRSPinfoPath']
FFPath = config.paths['FFPath']
SignalsPath = config.paths['SignalsPath']
ProcessedDataPath = config.paths['ProcessedDataPath']
ForcePreProcessing = config.ForcePreProcessing
ForceTraining = config.ForceTraining
batch_size_train = 64
batch_size_validation = 128
n_epochs = 200
ep_log_interval = 15
learning_rate = 0.001
loss_fn = nn.MSELoss()

# Select GPU or CPU based on availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Categorical_cols should not be only in the following variable, as the encoding should be done in general
if os.path.exists(ProcessedDataPath + '/dataset.csv') == False or ForcePreProcessing == True:
    print('Pre-processing data...')
    crsp, categorical_cols = dp.prepare_data(CRSPretPath, CRSPinfoPath, FFPath, SignalsPath, ProcessedDataPath)
    print(f'Pre-processing complete, the dataset has dimensions: {crsp.shape}')
    #print('\n\nThe columns of the complete df are:\n\n')
    #print(list(crsp.columns))
else:
    print('Datased found in memory, loading...')
    crsp = pd.read_csv(ProcessedDataPath+'/dataset.csv', index_col=0)
    
# Should returns be scaled?


if os.path.exists(currentPath+'/../saved/models/RegressionNet_model.pt') and ForceTraining == False:
    print('Saved model found, breaking execution')
    sys.exit()


train, val, test = split_data(crsp)
X_train, Y_train = sep_target(train)
X_val, Y_val = sep_target(val)
# Test split will be moved in the prediction part
#X_test, Y_test = sep_target(test)

del[train, val]#, test]

# Add encoding of categorical columns
train = CustomDataset(X_train, Y_train)
del [X_train, Y_train]

val = CustomDataset(X_val, Y_val)
del[X_val, Y_val]

#test = CustomDataset(X_test, Y_test)
#del[X_test, Y_test]


del crsp

# Should shuffle be set to true in my case? Check documentation
train_loader = DataLoader(train, batch_size=batch_size_train, num_workers=2)
val_loader = DataLoader(val, batch_size=batch_size_validation, num_workers=2)


tb = SummaryWriter()

n_inputs = train.data.shape[1]
model = RegressionNet(n_inputs).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)



for epoch in range(n_epochs):
    epoch_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
    validation_loss = validate_one_epoch(model, val_loader, optimizer, loss_fn, device)  
  
    if epoch % ep_log_interval == 0:
        print(f'Epoch n.{epoch} | Loss: {epoch_loss}')
        print(f'Validation Loss: {validation_loss}')

    tb.add_scalar("Training Loss", epoch_loss, epoch)
    tb.add_scalar("Validation Loss", validation_loss, epoch)
#    tb.add_histogram("Validation Loss", validation_loss, epoch)
#    for name, weight in model.named_parameters():
#        tb.add_histogram(name,weight, epoch)
#        tb.add_histogram(f'{name}.grad',weight.grad, epoch)
    tb.flush()
print(f'\nFinal Training Loss: {epoch_loss} at epoch n. {epoch+1}')
print(f'\nFinal Validation Loss: {validation_loss} at epoch n. {epoch+1}')

# Save final model (entire model)
print("\nSaving trained model")
fn = currentPath+'/../saved/models/RegressionNet_model.pt'
torch.save(model, fn)
