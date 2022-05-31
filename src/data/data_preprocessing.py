from importlib.resources import path
from typing import final
import pandas as pd
import datetime as dt

# Create a class for the df - this might be a good idea to be concise in Feature Engineering (?)

def load_crsp(CRSPretPath, CRSPinfoPath, StartYear=190001):

    """ 
        The function reads the two csv files containing CRSP stocks
        data and CRSP stock returns, filters for the years considered
        in the sample (StartYear is an optional variable)
        
        The two files are merged and...
    
    """
    
    crspm = pd.read_csv(CRSPretPath)#+'/crspminfo.csv')
    crspinfo = pd.read_csv(CRSPinfoPath)#+'/crspminfo.csv')

    # Drop date column (yyyymm will be used)
    crspm = crspm.drop(columns=['date'], errors='ignore')
    
    # Filter for observations since 1967 (last part commented out)
    crspm = crspm[(crspm["yyyymm"] >= StartYear)]

    # Merge CRSPret and CRSPinfo
    crsp = crspm.set_index(['permno', 'yyyymm']).join(
        crspinfo.set_index(['permno', 'yyyymm']), how='left').reset_index()\
            .drop(['me_nyse10', 'me_nyse20'], axis=1)
    
    # TODO
    # Select columns to keep
    #crsp = crsp[['permno', 'yyyymm', 'ret', 'melag', 
    #    'exchcd', 'shrcd', 'siccd']]
    
    return crsp

    
def split_by_size(df):
    # TODO
    
    """ 
        The function splits stocks in Big, Medium and Small caps according to their market cap.
        Multiple possitiblities:
            - Follow Messmer 2017
            - Follow ...

        Messmer 2017:
            As in Fama and French (1996), stocks qualify as large if their market capitalization ranks among the top 1000, 
            accordingly mid cap stocks have to fall into the range of rank 1001-2000, 
            and small caps comprise all stocks with rank > 2000.

        ...
    
    """

    df = df.drop(['me','melag'], axis=1)
    return df


def SIC_dummies(df):

    """ 
        The function creates a dummy column for each SIC industry code,
        following ... 
    
    """
    original_cols = list(df.columns)
    # Check out NaN in 'siccd' as it seems that the last month a stock has been traded the 'siccd' is NaN 
    df = pd.get_dummies(df, columns=['siccd'], dummy_na=True) # NaNs should not be contemplated (I guess?)
    after_dummy_cols = list(df.columns)
    dummy_cols = list(set(after_dummy_cols) - set(original_cols))
    

    return df, dummy_cols


def calculate_excess_returns(FFpath, df):
    
    """ 
        The function loads data from Kenneth French's library 
        relative to the one-month treasury bill rates, used as proxy
        of risk-free rate, and uses it to calculate stock excess returns.
  
    """

    FF = pd.read_csv(FFpath, sep=',')
    
    FF = FF.rename(columns={'Unnamed: 0':'yyyymm'})
    FF = FF[['yyyymm','RF']]

    df = df.merge(FF, on='yyyymm', how='left')

    df['ret'] = df['ret']-df['RF'] 

    df = df.drop('RF', axis=1)

    return df

def filter_share_code(df, allowed_shrcd=[10,11]):
    
    """
        Takes a list of allowed Share codes 
        (ideally from a config file) and filter to keep
        only pertinent samples in the dataframe

    """
    return df.loc[df['shrcd'].isin(allowed_shrcd)].drop('shrcd', axis=1)

def filter_exchange_code(df, allowed_exchcd=[1,2,3]):
    
    """
        Takes a list of allowed Share codes 
        (ideally from a config file) and filter to keep
        only pertinent samples in the dataframe

    """

    return df.loc[df['exchcd'].isin(allowed_exchcd)].drop('exchcd', axis=1)


def scale_predictors(df): #,signal_columns)

    """ 
        The function scales the predictors in by using rank-based
        scaling, resulting in characteristics being in a 
        range [-0.5, 0.5], following  Kelly, Pruitt & Su (2019).

        Takes the df and the signal_columns from the CRSP and signals db merge, the
        latter in order to select only pertinent columns for scaling.
    
        In each month the ranks of the stocks for each characteristic
        are calculated separately. Then, the ranks are divided by the number
        of non-missing observations of that characteristic in that month and 0.5
        is subtracted. As a result, all characteristics take only values in the 
        interval [-0.5, 0.5]. Lastly, missing characteristic values are imputed using
        the cross-sectional median after rank-based scaling, i.e. 0.
    
    """

    # Rank (add cross-sectional rank)
    #df[df.columns.intersection(columns)] = df[df.columns.intersection(columns)].rank(method='max') 
    #df[signal_columns] = df[signal_columns].rank(method='max')
    
    # Divide rank by length of non-null values in column and subtract 0.5
    #df[df.columns.intersection(columns)] = (df[df.columns.intersection(columns)]/df[df.columns.intersection(columns)].count())-0.5
    #df[signal_columns] = (df[signal_columns]/df[signal_columns].count()) - 0.5
    
    # Fill NaNs with 0
    #df = df[df.columns.intersection(columns)].fillna(0)
    #df[signal_columns] = df[signal_columns].fillna(0)

    df = df.set_index(['permno','yyyymm']).groupby('yyyymm')\
        .apply(lambda x: (x.rank(method='max')/x.count())-0.5)\
            .reset_index()    
    df = df.fillna(0)
    
    return df

def scale_returns(df):
    """
        Test function, could be deleted in the future
    """
    print(f'df index: {df.index}')
    df['ret'] = df.groupby('yyyymm')['ret']\
        .apply(lambda x: (x.rank(method='max')/x.count())-0.5)

    return df

def merge_crsp_with_signals(df, SingalsPath, chunksize=50000):

    """ 
        Takes the path to the Open Asset Pricing signals csv,
        the df containing CRSP data, and the desired chunksize.

        The function performs a merge between the CRSP stock and
        ret data with the signals creating the dataset used for the
        NN model.

        The merge is performed in chunks due to the dimension of the 
        signals csv file, which can't be fully loaded in memory on 
        computers with only 8GB of RAM available.
    
     """

    crsp_columns = list(df.columns)
    final_df = pd.DataFrame()
    i = 0

    for chunk in pd.read_csv(SingalsPath, chunksize=chunksize):

        chunk = scale_predictors(chunk)
        # Evaluate whether to change the join type (does this work?)
        temp = df.merge(chunk, on=['permno','yyyymm'], how='inner')

        if i == 0:
            final_df = final_df.reindex(columns = temp.columns.tolist())
        
        final_df = pd.concat([final_df, temp], axis = 0)
        i += 1
        if i%20 == 0:
            print(f'Loop n. {i} | df Size 0: {final_df.shape[0]} | df Size 1: {final_df.shape[1]}')

    final_df = final_df.astype({'yyyymm':int})
    final_df = final_df.reset_index(drop=True)
    signal_columns = list(set(list(final_df.columns)) - set(crsp_columns))

    return final_df, signal_columns


def winsorize(df):

    """ 
        The function ... 
    
    """
    return df


def prepare_data(CRSPretpath, CRSPinfopath, FFPath, SignalsPath, ProcessedDataPath):
    
    categorical_cols = []

    print('Loading data...')
    crsp = load_crsp(CRSPretpath, CRSPinfopath)
    print('Data Loaded')
    crsp, dummy_cols = SIC_dummies(crsp)
    categorical_cols.extend(dummy_cols)
    print(f'Dummy columns for sectors created, n. of columns: {len(dummy_cols)}')
    crsp = split_by_size(crsp)
    print('Stocks split by size')
    crsp = calculate_excess_returns(FFPath, crsp)
    print('Excess returns calculated.')
    crsp = filter_exchange_code(crsp)
    crsp = filter_share_code(crsp)
    print('Share codes and Exchange codes filtered')
    print('Beginning merge process between CRSP and OpenAssetPricing signals...')
    crsp, signal_columns = merge_crsp_with_signals(crsp, SignalsPath)
    print('Merge Complete.')
    crsp = winsorize(crsp)
    

    # Dropping remainging NAs, check this, in theory there should be just some rows.
    print(f'Dropping {crsp.shape[0] - crsp.dropna().shape[0]} rows as they still contain at least a NaN number (likely ret is missing)')
    crsp = crsp.dropna()

    # Test to check if anything gets better
    crsp = scale_returns(crsp)
    
    print('Data preparation complete, saving...')
    crsp.to_csv(ProcessedDataPath + '/dataset.csv')
    print('Processed dataset saved.')

    
    return crsp, categorical_cols


def split_data(df, end_train='198512', end_val='199512'):
    
    """
        Splits data between train, validation and test
        dataframes (should an expanding window be included?)
    """
    
    end_train = dt.datetime.strptime(end_train, '%Y%m')
    start_val = end_train + pd.DateOffset(months=1)
    start_val.strftime('%Y%m') 
    

    end_val = dt.datetime.strptime(end_val, '%Y%m')
    start_test = end_val + pd.DateOffset(months=1)
    start_test.strftime('%Y%m')
    
    train = df.loc[df['yyyymm'] <= int(end_train.strftime('%Y%m'))]
    val = df.loc[(df['yyyymm'] >= int(start_val.strftime('%Y%m'))) & (df['yyyymm'] <= int(end_val.strftime('%Y%m')))]
    test = df.loc[df['yyyymm'] >= int(start_test.strftime('%Y%m'))]

    return train, val, test


# Separates the features from the target in the df
def sep_target(data):

    """
        Separates the features from the target in the
        split dfs
    """
    X = data.drop(['permno', 'yyyymm', 'ret'], axis=1).to_numpy()
    Y = data['ret'].to_numpy().ravel()
    
    return X, Y

# Separates the features from the target in the df
def sep_target_idx(data):

    """
        Used for Test purposes, keeps track of the stock and the
        relative month and year of the prediciton.
    """
    
    X = data.drop(['permno', 'yyyymm', 'ret'], axis=1).to_numpy()
    Y = data['ret'].to_numpy().ravel()
    index = data[['permno', 'yyyymm']]
    
    return X, Y, index
