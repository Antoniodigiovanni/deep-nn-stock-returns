import torch
from torch.utils.data import Dataset

class CrspDataset(Dataset):
    def __init__(self, df):
        self.df = df

        self.X = df.drop(['ret','yyyymm','permno'], axis=1).values
        self.y = df[['ret']].values
        self.label = df[['yyyymm','permno']].values.astype(int)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X[idx], self.y[idx], self.label[idx]]
    
    def get_inputs(self):
        return self.X.shape[1]

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
            "X": torch.tensor(current_sample, dtype=torch.float),
            "Y": torch.tensor(current_target, dtype=torch.float)
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
            "X": torch.tensor(current_sample, dtype=torch.float),
            "Y": torch.tensor(current_target, dtype=torch.float)
        }
