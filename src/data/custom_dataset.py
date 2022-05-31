import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        current_sample = self.data[idx, :]
        current_target = self.targets[idx]
        
        return {
            "X" : torch.tensor(current_sample, dtype=torch.float), 
            "Y" : torch.tensor(current_target, dtype=torch.float)
        }

class TestDataset(Dataset):
    def __init__(self, data, targets, stock_index):
        self.data = data
        self.targets = targets
        self.stock_index = stock_index
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        current_sample = self.data[idx, :]
        current_target = self.targets[idx]
        permno = self.stock_index.iloc[idx, 0]
        yyyymm = self.stock_index.iloc[idx, 1]
        
        return {
            'permno': permno,
            'yyyymm': yyyymm,
            "X" : torch.tensor(current_sample, dtype=torch.float), 
            "Y" : torch.tensor(current_target, dtype=torch.float)
        }