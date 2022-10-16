import os
import sys
import config.config as config
import pandas as pd
import data.data_preprocessing as dp
import datetime as dt


class BaseDataset:
    def __init__(self, mode = 'none', crsp_ret_path=config.paths['CRSPretPath'], crsp_info_path=config.paths['CRSPinfoPath']):
        self.loadingTried = 0
        self.forcePreProcessing = config.ForcePreProcessing
        self.crspRetPath = crsp_ret_path
        self.crspInfoPath = crsp_info_path
        self.processedDatasetExists = None
        self.force_crsp_download = config.ForceCrspDownload

        dataset_modes = ['none', 'train', 'test']
        if mode not in dataset_modes:
            raise ValueError("Invalid mode. Expected one of: %s" %dataset_modes)
        
        
        self.__load_dataset()

        if mode == 'train':
            self.df = self.one_month_ahead_returns(df=self.df)
            print('One-month ahead returns generated...')
        elif mode == 'test':
            pass

    def __check_processed_df_exist(self):
        if os.path.exists(config.paths['finalDatasetPath']):
            self.processedDatasetExists = 1
        else:
            self.processedDatasetExists = 0

    def __load_dataset(self):
        self.__check_processed_df_exist()
        
        self.loadingTried += 1
        
        if self.loadingTried > 2:
            print('Impossible to load the dataframe, exiting')
            sys.exit()
        if (self.processedDatasetExists == 1) & (self.forcePreProcessing == False):
            print('Loading the dataset...')
            self.df = pd.read_csv(config.paths['finalDatasetPath'], index_col=0)

            # Removing features for testing
            # self.df = self.df.drop(['STreversal', 'High52', 'MaxRet', 'Price'], axis=1)
        else:
            print('Creating the dataset...')
            self.__create_dataset()
            self.__load_dataset()

    def __create_dataset(self):
        if (os.path.exists(config.paths['CRSPretPath']) == False) | (os.path.exists(config.paths['CRSPinfoPath']) == False) | (self.force_crsp_download == True):
            df = dp.download_crsp()
        print('Data pre-processing...')
        df = dp.load_crsp(crsp_ret_path = self.crspRetPath, crsp_info_path = self.crspInfoPath)
        df = dp.remove_microcap_stocks(df)
        df = dp.filter_exchange_code(df)
        df = dp.filter_share_code(df)
        df = dp.calculate_excess_returns(df)
        df = dp.winsorize_returns(df)
        df = dp.de_mean_returns(df)
        print('Merging returns information with signals...')
        df, features = dp.merge_crsp_with_signals(df)    
        
        print('Scaling features...')
        df = dp.scale_features(df, features)
        df = dp.drop_extra_columns(df)
        
        # Save dataset
        if not os.path.exists(config.paths['ProcessedDataPath']):
            os.makedirs(config.paths['ProcessedDataPath'])
        df.to_csv(config.paths['finalDatasetPath'])

        # Changing the variable to False because Pre-processing has now been done
        self.forcePreProcessing = False

    def one_month_ahead_returns(self, df):

        """ 
            Used to predict or train on predicting one-month ahead returns 
            (shifts returns one month before)
        """
        shift_df = df.copy()

        shift_df = shift_df.sort_values(by=['permno','yyyymm']).reset_index(drop=True)
        shift_df['date'] = shift_df['yyyymm'].apply(lambda x: dt.datetime.strptime(str(x), '%Y%m')) 
        shift_df['tmp'] = (shift_df['date'] - pd.DateOffset(months=1))

        # Inverting the .isin to check if the month before exists (instead of month after as in the other file)
        s = shift_df.groupby('permno').apply(lambda x: x['date'].isin(x['tmp']))
        shift_df['exists_prev_month'] = s.reset_index(level=0)['date']


        shift_df['ret'] = shift_df.groupby('permno')['ret'].shift(-1)
        shift_df = shift_df.loc[shift_df['exists_prev_month'] != False]
        shift_df.drop(['date','tmp','exists_prev_month'], axis=1, inplace=True)

        return shift_df
