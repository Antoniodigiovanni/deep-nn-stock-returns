import torch
import config
from models.neural_net import metric
import nni
import json
from json.decoder import JSONDecodeError
from pathlib import Path
import nni.assessor
import numpy as np

class NeuralNetTrainer():
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, params, nni_experiment=False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = config.device
        self.nni_experiment = nni_experiment
        self.params = params
        self.patience = 5 # Include patience in variables
        self.best_val_loss = np.inf

    def train(self):
        j = 0
     
        for epoch in range(self.params['epochs']):
            while j < self.patience:
                epoch_loss = self.train_one_epoch()
                validation_loss, validation_acc = self.validate_one_epoch()
                if validation_loss < self.best_val_loss:
                    j = 0
                    self.best_val_loss = validation_loss
                    # Gu et al would not update the model if the validation loss 
                    # was not lower than the best validation loss (should I?)
                else:
                    j += 1
                if self.nni_experiment:
                    nni.report_intermediate_result(validation_acc)
                    nni.assessor.AssessResult(validation_acc)
                
                # Logging
                with open(config.paths['logsPath'] + config.logFileName, 'a') as fh:
                    fh.write('Network parameters:')
                    fh.write(str(self.params))
                if epoch % config.ep_log_interval == 0:
                    with open(config.paths['logsPath'] + config.logFileName, 'a') as fh:
                        fh.write(f'\nEpoch n.{epoch} | Loss: {epoch_loss}\nValidation Loss: {validation_loss} | Validation Accuracy: {validation_acc}%\n')
                    print(f'Epoch n.{epoch} | Loss: {epoch_loss}\nValidation Loss: {validation_loss} | Validation Accuracy: {round(validation_acc,4)}%')
                #tb.add_scalar("Training Loss", epoch_loss, epoch)
                #tb.add_scalar("Validation Loss", validation_loss, epoch)
                #tb.flush()
                if j >= self.patience:
                    print(f'Stopping early at epoch {epoch}!')
       
        # Logging
        with open(config.paths['logsPath'] +config.logFileName, 'a') as fh:
            fh.write('\n\nTraining complete\n\n')
            fh.write(f'Training epochs: {epoch+1}')
            fh.write(f'\nFinal Training Loss: {epoch_loss}')
            fh.write(f'\nFinal Validation Loss: {validation_loss}')
            fh.write(f'\nFinal Validation Accuracy: {round(validation_acc,2)}%\n')

        if self.nni_experiment:
            nni.report_final_result(validation_acc)
            self.update_best_params(validation_acc)


    def update_best_params(self, validation_acc):
        
        # Best parameters update is performed at the end of the trial only.
        bestParamsFilepath = Path(config.paths['resultsPath']+config.bestParamsFileName)
        bestParamsFilepath.touch(exist_ok=True)
        
        with open(config.paths['resultsPath']+config.bestParamsFileName, 'r') as pf:
            try:
                file_params = json.load(pf)
            except JSONDecodeError:
                file_params = {}
                pass
        if 'metric' in file_params.keys():
            if validation_acc >= file_params['metric']:
                with open(config.paths['resultsPath']+config.bestParamsFileName, 'w') as pf:
                    self.params['metric'] = validation_acc
                    json.dump(self.params, pf)
        else:
            with open(config.paths['resultsPath']+config.bestParamsFileName, 'w') as pf:
                    self.params['metric'] = validation_acc
                    json.dump(self.params, pf)

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0

        for _, data in enumerate(self.train_loader):
            loss = self.train_one_step(data)
            #scheduler.step()
            total_loss += loss
            
        # Make loss batch_size-agnostic (correct?)
        total_loss = total_loss / self.train_loader.batch_size
        return total_loss
      

    def train_one_step(self, data):
        self.optimizer.zero_grad()
        
        for k,v in data.items():
            data[k] = v.to(self.device)
        
        yhat = self.model(data['X'])

        loss = self.loss_fn(yhat.ravel(), data['Y'].ravel())

        loss.backward()
        self.optimizer.step()
        return loss

    
    def validate_one_epoch(self):
        self.model.train(False) # Same as model.eval()
        total_loss = 0
        total_acc = 0

        for batch_index, data in enumerate(self.val_loader):
            loss, acc = self.validate_one_step(data)
            total_loss += loss
            total_acc += acc

        # Make loss batch_size-agnostic (correct?)
        total_loss = total_loss / self.val_loader.batch_size
        total_acc = total_acc /self.val_loader.batch_size * 100
        return total_loss, total_acc
    

    def validate_one_step(self, data):
        #self.optimizer.zero_grad() #Is this needed for validation?
        
        for k,v in data.items():
            data[k] = v.to(self.device)
        
        with torch.no_grad():
            yhat = self.model(data['X'])
        
        loss = self.loss_fn(yhat.ravel(), data['Y'].ravel())
        acc = metric.accuracy(yhat, data['Y'], 0.1)
        return loss, acc
