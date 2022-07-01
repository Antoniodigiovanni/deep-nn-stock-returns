import pandas as pd
import datetime as dt
import numpy as np
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import os
import config


# Create a class for the df - this might be a good idea to be concise in Feature Engineering (?)

def load_crsp(CRSPretPath, CRSPinfoPath, StartYear=197001, EndYear=202112):
    """
        The function reads the two csv files containing CRSP stocks
        data and CRSP stock returns, filters for the years considered
        in the sample (StartYear is an optional variable)
        
        The two files are merged and...
    
    """
    print(f'Loading CRSP data, the starting year is set at {StartYear}')
    crspm = pd.read_csv(CRSPretPath)  # +'/crspminfo.csv')
    crspinfo = pd.read_csv(CRSPinfoPath)  # +'/crspminfo.csv')

    # Drop date column (yyyymm will be used)
    crspm = crspm.drop(columns=['date'], errors='ignore')

    # Filter for observations since 1967 (last part commented out)
    
    #crspm = crspm[(crspm["yyyymm"] >= StartYear)]
    crspm = crspm[(crspm["yyyymm"] <= EndYear)]

    # Merge CRSPret and CRSPinfo
    crsp = crspm.set_index(['permno', 'yyyymm']).join(
        crspinfo.set_index(['permno', 'yyyymm']), how='left').reset_index() \
        .drop(['me_nyse10'], axis=1)

    # TODO
    # Select columns to keep
    # crsp = crsp[['permno', 'yyyymm', 'ret', 'melag',
    #    'exchcd', 'shrcd', 'siccd']]

    return crsp


def remove_microcap_stocks(df, method=1):
    """
        The function splits stocks in Big, Medium and Small caps according to their market cap.
        Multiple possitiblities:
            1 - Follow Benjamin's suggestion from previous thesis and Fama French 2008, microcap stocks are all the ones
                with me smaller than NYSE 20th percentile market cap.

            2 - Follow Messmer 2017
            
            3 - Follow Oezturk Cem thesis, keeping only Large Cap stocks with above than median market cap 
                (I would do cross-sectionally, instead I think he does once for all the sample period)
            
            4 - Follow Gu et al. - which keep all stocks but present some analysis divided by size categories 
              (e.g. a break out predictability for large stocks (the top-1,000 stocks by market equity each month) and
              small stocks (the bottom-1,000 stocks each month) based on the full estimated model (using all stocks),
              but with a focus on predicting subsamples.
              

        Messmer 2017:
            As in Fama and French (1996), stocks qualify as large if their market capitalization ranks among the top 1000, 
            accordingly mid cap stocks have to fall into the range of rank 1001-2000, 
            and small caps comprise all stocks with rank > 2000.

        ...
    
    """

    # Method 1
    if method == 1:
        print(f'N. rows before filtering for microcap stocks: {df.shape[0]}')
        df = df.loc[df['me'] >= df['me_nyse20']]
        print(f'N. rows after filtering for microcap stocks: {df.shape[0]}')

    # Method 2
    if method == 2:
        df['SizeRank'] = df.groupby('yyyymm')['melag'].rank(method='max', ascending=False)

        conditions = [
            df['SizeRank'] <= 1000,
            (df['SizeRank'] > 1000) & (df['SizeRank'] <= 2000),
            df['SizeRank'] > 2000
        ]
        cons = ['Large', 'Medium', 'Small']

        df['MarketCapGroup'] = np.select(conditions, cons)
        df = df.loc[df['MarketCapGroup'].isin(['Large', 'Medium'])]
        df = df.drop(['melag', 'SizeRank', 'MarketCapGroup'], axis=1)

    # Method 3
    if method == 3:
        df = df.set_index('yyyymm')
        df['SizeMedian'] = df.groupby('yyyymm')['melag'].median()
        df = df.reset_index()

        df = df.loc[df['melag'] >= df['SizeMedian']]
        df = df.drop(['melag', 'SizeMedian'], axis=1)

    return df


def SIC_dummies(df):
    """
        The function creates a dummy column for each SIC industry code,
        following ... 
    
    """

    print('Info before SIC dummies included:')
    print(df.info(verbose=False, memory_usage="deep"))
    original_cols = list(df.columns)
    # Check out NaN in 'siccd' as it seems that the last month a stock has been traded the 'siccd' is NaN 
    df = pd.get_dummies(df, columns=['siccd'], dummy_na=True,
                        dtype=np.int16)  # NaNs should not be contemplated (I guess?)
    after_dummy_cols = list(df.columns)
    dummy_cols = list(set(after_dummy_cols) - set(original_cols))

    # Converting dummy cols type to int32
    df[dummy_cols] = df[dummy_cols].astype(np.int8)
    print('Info after SIC dummies included:')
    print(df.info(verbose=False, memory_usage="deep"))

    print(f'Shape after dummy columns have been included{df.shape}')

    return df, dummy_cols


def calculate_excess_returns(FFpath, df):
    """
        The function loads data from Kenneth French's library 
        relative to the one-month treasury bill rates, used as proxy
        of risk-free rate, and uses it to calculate stock excess returns.
  
    """
    FF5FMurl = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip'

    if os.path.isfile(FFpath) is False:
        download_and_unzip(FF5FMurl, config.dataPath + '/external')
        print('Fama French 5 Factor Model returns downloaded from Kenneth French\'s library')

    FF = pd.read_csv(FFpath, skiprows=3, skipfooter=60, engine='python')

    FF = FF.rename(columns={'Unnamed: 0': 'yyyymm'})
    FF = FF[['yyyymm', 'RF']]

    df = df.merge(FF, on='yyyymm', how='left')

    df['ret'] = df['ret'] - df['RF']

    df = df.drop('RF', axis=1)

    return df


def filter_share_code(df, allowed_shrcd=[10, 11]):
    """
        Takes a list of allowed Share codes 
        (ideally from a config file) and filter to keep
        only pertinent samples in the dataframe

    """
    return df.loc[df['shrcd'].isin(allowed_shrcd)].drop('shrcd', axis=1)


def filter_exchange_code(df, allowed_exchcd=[1, 2, 3]):
    """
        Takes a list of allowed Share codes 
        (ideally from a config file) and filter to keep
        only pertinent samples in the dataframe

    """

    return df.loc[df['exchcd'].isin(allowed_exchcd)].drop('exchcd', axis=1)


def scale_predictors(df):  # ,signal_columns)

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

    df = df.set_index(['permno', 'yyyymm']).groupby('yyyymm') \
        .apply(lambda x: (x.rank(method='max') / x.count()) - 0.5) \
        .reset_index()
    df = df.fillna(0)

    return df


def merge_crsp_with_signals_chunks(df, SingalsPath, chunksize=50000):
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

    # Should dummy signals be scaled? (At the moment the merge and scaling are done before creating the dummy columns)
    crsp_columns = list(df.columns)
    final_df = pd.DataFrame()
    i = 0
    colswithallnans = 0
    for chunk in pd.read_csv(SingalsPath, chunksize=chunksize):

        colswithallnans += (chunk.iloc[:, 3:][chunk.isnull().all(axis=1)].shape[0])
        chunk = scale_predictors(chunk)

        # Converting columns to float32, to save memory
        chunk[chunk.select_dtypes(np.float64).columns] = chunk.select_dtypes(np.float64).astype(np.float32)

        # Evaluate whether to change the join type (does this work?)
        temp = df.merge(chunk, on=['permno', 'yyyymm'], how='inner')

        if i == 0:
            final_df = final_df.reindex(columns=temp.columns.tolist())
            print('printing types:')
            print(chunk.dtypes)

        final_df = pd.concat([final_df, temp], axis=0)
        i += 1
        if i % 20 == 0:
            print(f'Loop n. {i} | df Size 0: {final_df.shape[0]} | df Size 1: {final_df.shape[1]}')

    final_df = final_df.astype({'yyyymm': int})
    final_df = final_df.reset_index(drop=True)
    signal_columns = list(set(list(final_df.columns)) - set(crsp_columns))
    print(f'Columns with all NaNs: {colswithallnans}')
    return final_df, signal_columns


def merge_crsp_with_signals_no_loop(df, SingalsPath):
    """
       Version without for loop of the above function
    
     """

    print('Merging CRSP data with signals from Open Asset Pricing')
    crsp_columns = list(df.columns)

    signals = pd.read_csv(SingalsPath)
    print('Signals loaded in memory...')
    signals = scale_predictors(signals)
    colswithallnans = (signals.iloc[:, 3:][signals.isnull().all(axis=1)].shape[0])

    # Converting columns to float32, to save memory
    signals[signals.select_dtypes(np.float64).columns] = signals.select_dtypes(np.float64).astype(np.float32)

    df = df.merge(signals, on=['permno', 'yyyymm'], how='inner')
    df = df.astype({'yyyymm': int})
    # final_df = final_df.reset_index(drop=True)
    signal_columns = list(set(list(df.columns)) - set(crsp_columns))
    print(f'Columns with all NaNs: {colswithallnans}')
    return df, signal_columns


def winsorize_returns(df):
    """
        The function winsorizes returns at 1% and 99% levels. 
    
    """

    ret_09 = df.groupby('yyyymm')['ret'].quantile(0.9).reset_index().rename(columns={'ret': 'ret_09'})
    ret_01 = df.groupby('yyyymm')['ret'].quantile(0.1).reset_index().rename(columns={'ret': 'ret_01'})

    df = df.merge(ret_01, on='yyyymm', how='left')
    df = df.merge(ret_09, on='yyyymm', how='left')

    df['ret'] = df['ret'].clip(df['ret_01'], df['ret_09'])
    df = df.drop(['ret_01', 'ret_09'], axis=1)

    return df


def de_mean_returns(df):
    """ 
        The function cross-sectionally subtracts the avg. return to the stock
        returns. This is done to keep cross-sectional information. 
        (Should be alternative to scaling returns - check if correct)
    
    """

    df['ret'] = df.groupby('yyyymm')['ret'].apply(lambda x: (x - np.mean(x)))

    return df


def split_data_train_val(df, end_train='198512', end_val='199512'):
    """
        Splits data between train, validation and test
        dataframes (should an expanding window be included?)
    """
    # If default values are passed to the function, gather the values from
    # the config file, which is where the actual values are decided
    # (ideally they are the same) - temporary solution before reformatting
    # config variables

    if end_train == '198512' and end_val == '199512':
        end_train = config.end_train
        end_val = config.end_val

    end_train = dt.datetime.strptime(end_train, '%Y%m')
    start_val = end_train + pd.DateOffset(months=1)
    start_val.strftime('%Y%m')

    end_val = dt.datetime.strptime(end_val, '%Y%m')

    train = df.loc[df['yyyymm'] <= int(end_train.strftime('%Y%m'))]
    val = df.loc[(df['yyyymm'] >= int(start_val.strftime('%Y%m'))) & (df['yyyymm'] <= int(end_val.strftime('%Y%m')))]

    return train, val


def split_data_test(df, end_val='199512'):
    """
        Splits test data from the initial df
        (should an expanding window be included?)
    """
    # If default values are passed to the function, gather the values from
    # the config file, which is where the actual values are decided
    # (ideally they are the same) - temporary solution before reformatting
    # config variables

    if end_val == '199512':
        end_val = config.end_val

    end_val = dt.datetime.strptime(end_val, '%Y%m')
    start_test = end_val + pd.DateOffset(months=1)
    start_test.strftime('%Y%m')

    test = df.loc[df['yyyymm'] >= int(start_test.strftime('%Y%m'))]

    return test


def sep_target(data):
    """
        Separates the features from the target in the
        split dfs
    """
    X = data.drop(['permno', 'yyyymm', 'ret'], axis=1).to_numpy()
    Y = data['ret'].to_numpy().ravel()

    return X, Y


def sep_target_idx(data):
    """
        Used for Test purposes (test df), keeps track of the stock and the
        relative month and year of the prediciton.
    """

    X = data.drop(['permno', 'yyyymm', 'ret'], axis=1).to_numpy()
    Y = data['ret'].to_numpy().ravel()
    test_index = data[['permno', 'yyyymm']]

    return X, Y, test_index


def download_and_unzip(url, extract_to='.'):
    """
        Used to download FF5FM returns from Kenneth French's library.
    """
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)
