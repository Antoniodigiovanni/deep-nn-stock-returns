from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
import datetime as dt
import data.data_preprocessing as dp
import pandas as pd

class CrspDataset(Dataset):
    def __init__(self, df, scale_target = False):
        self.df = df.copy()
        if self.df.empty == False:
            self.df = self.shift_ret_and_yyyymm(self.df)
            # self.df = self.shift_features(self.df)
        
        # To test with less features
        # self.df = self.df.iloc[:,:50]
        self.X = self.df.drop(['ret','yyyymm','permno'] + ['me','prc','melag','me_nyse10','me_nyse20','me_nyse50'], axis=1, errors='ignore').values
        
        if scale_target:
            self.X = dp.scale_returns(self.X)
            # X = StandardScaler().fit_transform(X)

        self.y = self.df[['ret']].values
        self.label = self.df[['yyyymm','permno']].values.astype(int)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X[idx], self.y[idx], self.label[idx]]
    
    def get_inputs(self):
        return self.X.shape[1]

    def shift_features(self, df):
        features = df.iloc[:,3:].columns
        df['date'] = df['yyyymm'].apply(lambda x: dt.datetime.strptime(str(x), '%Y%m'))
        df['tmp'] = (df['date']  - pd.DateOffset(months=1))

        s = df.groupby('permno').apply(lambda x: x['tmp'].isin(x['date']))
        df['present_previous_month'] = s.reset_index(level=0)['tmp']

        # Shift features to next month
        shifted_df = df.copy()
        shifted_df[features] = shifted_df.sort_values(by='yyyymm').groupby('permno')[features].shift()
        
        shifted_df = shifted_df.loc[shifted_df['present_previous_month'] != False].copy()
        shifted_df.drop(['tmp','date', 'present_previous_month'], axis=1, inplace=True)

        shifted_df = shifted_df.dropna()

        return shifted_df

    def shift_ret_and_yyyymm(self,df):
        cols_to_shift = df[['yyyymm','ret']].columns
        df['date'] = df['yyyymm'].apply(lambda x: dt.datetime.strptime(str(x), '%Y%m'))
        df['tmp'] = (df['date']  + pd.DateOffset(months=1))

        s = df.groupby('permno').apply(lambda x: x['tmp'].isin(x['date']))
        df['present_following_month'] = s.reset_index(level=0)['tmp']

        # Shift features to next month
        shifted_df = df.copy()
        shifted_df[cols_to_shift] = shifted_df.sort_values(by='yyyymm').groupby('permno')[cols_to_shift].shift(-1)

        shifted_df = shifted_df.loc[shifted_df['present_following_month'] == True].copy()
        shifted_df.drop(['tmp','date', 'present_following_month'], axis=1, inplace=True)

        shifted_df = shifted_df.dropna()

        return shifted_df


    """def shift_features(self, df):
        df['date'] = df['yyyymm'].apply(lambda x: dt.datetime.strptime(str(x), '%Y%m'))

        df['tmp'] = (df['date']  - pd.DateOffset(months=1))
        s = df.groupby('permno').apply(lambda x: x['date'].isin(x['tmp']))
        df['present_next_month'] = s.reset_index(level=0)['date']

        df = df.loc[df['present_next_month'] != False].copy()

        df.drop(['tmp','date', 'present_next_month'], axis=1, inplace=True)
        
        # Shift features to next month
        
        features = df.iloc[:,3:].columns
        shifted_df = df.copy()
        shifted_df[features] = shifted_df.groupby('permno')[features].shift()
        shifted_df = shifted_df.dropna() 

        return shifted_df"""