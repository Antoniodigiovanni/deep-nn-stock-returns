import data.data_preprocessing as dp
import config.config
import os
import pandas as pd


class BaseDataset:
    """
        Handles the data loading, feature engineering, and split between
        train, validation and test data, as well as the split between features
        and targets.

        The following methods are implemented:
            - __initialize_dataset:
                Internal method that returns the CRSP + Signals df for internal use only. 
            
            - load_dataset_in_memory:
                    
                Created variables:
                    - self.crsp
                Returns the CRSP + Signals df (which is otherwise loaded but not saved)
                
            - load_train_data:

                Created variables:
                    - self.train
                    - self.val

                Returns train and validation data.
                The train and validation timeframes are defined in the config file

            - load_split_train_data:

                Created variables:
                    - self.X_train
                    - self.y_train
                    - self.X_val
                    - self.y_val

                Splits the train data (train+validation) in X and y, meaning
                features (predictors) and targets. 

            - load_test_data:

                Created variables:
                    - self.test


            - load_split_test_data 
                Created variables:
                    - self.X_test
                    - self.y_test
                
    """

    def __init__(self, mode='train') -> None:

        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.train = None
        self.val = None
        self.test_index = None
        self.test = None
        self.crsp = None
        self.categorical_cols = []

        # Nothing is being done now as __init__ because otherwise the crsp would be loaded multiple times
        # An idea could be to load crsp here and then load it when other functions are called only if necessary.
        # if mode == 'train':
        #     print('Initializing dataset in train mode...')
        #     self.load_split_train_data()
        #     print('Access to X_train, X_val, y_train and y_val attributes available')

        # if mode == 'test':
        #     print('Initializing dataset in test mode...')
        #     self.load_split_test_data()
        #     print('Access to X_test and y_test attributes available')

    def __initialize_dataset(self):
        if config.ForcePreProcessing is True or (
                os.path.exists(config.paths['ProcessedDataPath'] + '/dataset.csv') is False):

            print('Data processing...')
            crsp = dp.load_crsp(config.paths['CRSPretPath'], config.paths['CRSPinfoPath'])
            crsp = dp.filter_exchange_code(crsp)
            crsp = dp.filter_share_code(crsp)

            crsp = dp.remove_microcap_stocks(crsp)

            crsp = dp.calculate_excess_returns(config.paths['FFPath'], crsp)
            crsp = dp.winsorize_returns(crsp)

            crsp = dp.de_mean_returns(crsp)

            # Should also try to merge crsp with signals without for loop
            crsp, signal_columns = dp.merge_crsp_with_signals_chunks(crsp, config.paths['SignalsPath'])


            # Trying without SIC_dummies - too many columns unfortunately...
            crsp.drop('siccd', axis=1, inplace=True)
            
            #crsp, dummy_cols = dp.SIC_dummies(crsp)
            #self.categorical_cols.extend(dummy_cols)

            # Dropping remaining NAs, check this, in theory there should be just some rows.
            print(
                f'Dropping {crsp.shape[0] - crsp.dropna().shape[0]} rows as they still contain at least a NaN number '
                f'(likely ret is missing)')
            crsp = crsp.dropna()

            print('Data preparation complete, saving...')
            crsp.to_csv(config.paths['ProcessedDataPath'] + '/dataset.csv')
        else:
            crsp = pd.read_csv(config.paths['ProcessedDataPath'] + '/dataset.csv', index_col=0)
        return crsp

    def load_dataset_in_memory(self):
        self.crsp = self.__initialize_dataset()

    def load_train_data(self):
        self.load_dataset_in_memory()

        self.train, self.val = dp.split_data_train_val(self.crsp)
        del self.crsp

    def load_split_train_data(self):
        self.load_dataset_in_memory()

        self.train, self.val = dp.split_data_train_val(self.crsp)
        del self.crsp
        self.X_train, self.y_train = dp.sep_target(self.train)
        self.X_val, self.y_val = dp.sep_target(self.val)

        del ([self.train, self.val])

    def load_test_data(self):
        self.load_dataset_in_memory()

        self.test = dp.split_data_test(self.crsp)
        del self.crsp

    def load_split_test_data(self):
        self.load_dataset_in_memory()

        self.test = dp.split_data_test(self.crsp)
        del self.crsp

        self.X_test, self.y_test, self.test_index = dp.sep_target_idx(self.test)
        del self.test
