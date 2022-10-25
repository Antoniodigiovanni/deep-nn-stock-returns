import pandas as pd
import numpy as np
from portfolios.FF5FM_Mom import FF5FM_Mom
import config.config as config
import os
from pandas.tseries.offsets import *
import datetime as dt


def load_crsp(crsp_ret_path=config.paths['CRSPretPath'], crsp_info_path=config.paths['CRSPinfoPath'], Startyyyymm = 197001, Endyyyymm = 202112):
    crspret = pd.read_csv(crsp_ret_path)
    crspinfo = pd.read_csv(crsp_info_path)

    crsp = crspret.merge(crspinfo, on=['permno','yyyymm'], how='inner')
    crsp.drop('Date', axis=1, inplace=True, errors='ignore')
    # Remove all NaNs, which are stocks missing prc, me, or stocks missing melag
    crsp = crsp.loc[crsp.notna().all(axis=1)]

    crsp = crsp.loc[(crsp['yyyymm'] >= int(Startyyyymm)) & (crsp['yyyymm'] <= int(Endyyyymm))]

    return crsp

def filter_exchange_code(df, exchcd=None):
    if exchcd is None:
        exchcd = [1, 2, 3]
    df = df.loc[df['exchcd'].isin(exchcd)].drop('exchcd', axis=1).copy()
    return df

def filter_share_code(df, shrcd=None):
    if shrcd is None:
        shrcd = [10, 11]
    df = df.loc[df['shrcd'].isin(shrcd)].drop('shrcd', axis=1).copy()
    return df

def remove_microcap_stocks(df, stocks_to_keep=['Medium','Big']):
    
    df = __calculate_stock_sizes(df)
    
    df = df.loc[df['StockSize'].isin(stocks_to_keep)]
    
    return df

def calculate_excess_returns(df):
    FF5FM = FF5FM_Mom()
    RF = FF5FM.RF

    df = df.merge(RF, how='inner', on='yyyymm')
    df['ret'] = df['ret'] - df['RF']

    return df

def winsorize_returns(df):
    ret_90 = pd.DataFrame(df.groupby('yyyymm')['ret'].quantile(0.9)).reset_index().rename({'ret':'ret90'}, axis=1)
    ret_10 = pd.DataFrame(df.groupby('yyyymm')['ret'].quantile(0.1)).reset_index().rename({'ret':'ret10'}, axis=1)
    
    df = df.merge(ret_90, how='left', on=['yyyymm'])
    df = df.merge(ret_10, how='left', on=['yyyymm'])   

    df['ret'] = df['ret'].clip(df['ret10'], df['ret90'])

    return df

def de_mean_returns(df):
    df['ret'] = df.groupby('yyyymm')['ret'].apply(lambda x: x-x.mean()) 
    
    return df
   
def merge_crsp_with_signals(df, signalspath=config.paths['SignalsPath']):
    final_df = pd.DataFrame()
    i = 0

    for chunk in pd.read_csv(signalspath, chunksize=100000):
        # chunk = scale_predictors(chunk)
        temp = df.merge(chunk, how='inner', on=['permno','yyyymm'])
        if i == 0:
            final_df = final_df.reindex(columns=temp.columns.tolist())
        i+=1
        final_df = pd.concat([final_df, temp], axis=0)
        # print(final_df)
    
    cols_to_scale = chunk.columns

    # Re-order cols_to_scale to have permno and yyyymm as first and second element and STreversal, price and size as the last ones
    features = cols_to_scale.drop(['permno', 'yyyymm'])
    cols_to_scale = ['permno', 'yyyymm'] + features.tolist() + ['STreversal', 'Price', 'Size']

    final_df = final_df.astype({'yyyymm': int, 'permno':int})
    
    return final_df, features

def rank_scale (x):
    return ((x.rank(method='max') / x.count()) - 0.5)

def standard_scale (x):
    return (((x- x.mean()) / x.std()))


def scale_returns(df, method=None):
    if method == None or method == 'rank':
        df['ret'] = df.groupby('yyyymm')['ret'].transform(lambda x: rank_scale(x))
    elif method == 'standard':
        df['ret'] = df.groupby('yyyymm')['ret'].transform(lambda x: standard_scale(x))
    return df
    
def scale_features(df, features, method=None):
    
    reduced_cols = features[:-3]
    print(reduced_cols[-5:])
    print(f'Shape of final_df before scaling: {df.shape[0]}')

    # print(features)

    if method == None or method == 'rank':
        df[features] = (
            df.groupby('yyyymm')[features]
            .transform(lambda x: rank_scale(x))
            )
    elif method == 'standard':
            df[features] = (
                df.groupby('yyyymm')[features]
                .transform(lambda x: standard_scale(x))
                )
    else:
        print('Scaling method should either be not passed as an argument, "rank" or "standard", other options are not available')
    
    # Removing rows with missing signals (where all are missing apart from STreverse, price and size)
    missing_signals = df.loc[df[reduced_cols].isna().all(axis=1)][['permno','yyyymm']]
    
    df = df.merge(
        missing_signals.drop_duplicates(), 
        on=['permno','yyyymm'], 
        how='left', indicator=True)
    df = df.loc[df['_merge'] != 'both']
    
    df.drop('_merge', axis=1, inplace=True)
    
    print(f'After scaling, the length is: {df.shape[0]}')
    
    df = df.fillna(0)

    return df

def drop_extra_columns(df, cols = ['date','melag','prc','me','siccd','me_nyse20','me_nyse50','RF','StockSize','RF','ret90','ret10']):
    df.drop(cols, axis=1, inplace=True, errors='ignore')
    return df

def __calculate_stock_sizes(df):
    df.drop(['me_nyse10', 'me_nyse20'], axis=1, inplace=True, errors='ignore')

    me_nyse_20 = pd.DataFrame(
        df.loc[df['exchcd'] == 1]
        .groupby('yyyymm')['me']
        .quantile(.2)
        ).reset_index().rename({'me':'me_nyse20'}, axis=1)

    me_nyse_50 = pd.DataFrame(
        df.loc[df['exchcd'] == 1]
        .groupby('yyyymm')['me']
        .quantile(.5)
        ).reset_index().rename({'me':'me_nyse50'}, axis=1)

    df = df.merge(me_nyse_20, how='left', on=['yyyymm'])
    df = df.merge(me_nyse_50, how='left', on=['yyyymm'])    

    # If the dataset does not start in June, the first months are not lost by selecting the 
    # min yyyymm in the rebalancing df
    crsp_june = df.loc[(df['yyyymm']%100 == 6) | (df['yyyymm'] == df['yyyymm'].min())].copy()

    conditions =[
        (crsp_june['me'] < crsp_june['me_nyse20']),
        ((crsp_june['me'] >= crsp_june['me_nyse20']) & (crsp_june['me'] < crsp_june['me_nyse50'])),
        (crsp_june['me'] >= crsp_june['me_nyse50'])
    ]
    values = ['Micro', 'Medium', 'Big']

    crsp_june['StockSize'] = np.select(conditions, values)

    crsp_june = crsp_june[['permno', 'yyyymm', 'StockSize']].copy()
    df = df.merge(crsp_june, how='left', on=['permno','yyyymm'])

    df['StockSize'] = df.groupby(['permno'])[['StockSize']].ffill()
    
    # Removing NaNs in StockSize
    df = df.loc[df['StockSize'].notna()]

    return df

def download_crsp():
    import wrds
    import configparser

    configuration = configparser.ConfigParser()
    if os.path.exists(config.dataPath + '/../src/data/credentials.ini'):
        print('Reading configuration')
        configuration.read(config.dataPath + '/../src/data/credentials.ini')
        
        username = configuration.get('credentials', 'username')
        password = configuration.get('credentials', 'password')
    
    else:
        raise FileNotFoundError()

    
    conn = wrds.Connection(
        wrds_username = username,
        wrds_password = password
    )

    crsp = conn.raw_sql(
    """
    select a.permno, a.permco, a.date, a.ret, a.retx, a.vol, a.shrout, a.prc, a.cfacshr, a.bidlo, a.askhi,
    b.comnam, b.shrcd, b.exchcd, b.siccd, b.ticker, b.shrcls,          -- from identifying info table
    c.dlstcd, c.dlret                                                  -- from delistings table
    from crsp.msf as a
    left join crsp.msenames as b
    on a.permno=b.permno
    and b.namedt<=a.date
    and a.date<=b.nameendt
    left join crsp.msedelist as c
    on a.permno=c.permno
    and date_trunc('month', a.date) = date_trunc('month', c.dlstdt)
    """, 
    date_cols=['date'])

    # Exclude entries in which a good price is missing
    if 'prc' in crsp.columns:
        crsp = crsp.dropna(subset=['prc'])

    # Exclude months in which the share code is missing
    if 'shrcd' in crsp.columns:
        crsp = crsp.dropna(subset=['shrcd'])

    # Move date to end of the month
    crsp['date']=crsp['date']+MonthEnd(0)

    # Create yearmonth column
    crsp['yyyymm'] =  crsp['date'].apply(lambda x: dt.datetime.strftime(x, '%Y%m')) 
    crsp['yyyymm'] = crsp.yyyymm.astype(int)

    # Deal with de-listing returns
    crsp['dlret'] = crsp['dlret'].fillna(0)
    crsp['ret'] = crsp['ret'].fillna(0)
    crsp = crsp.loc[crsp['dlret'] != -1]
    crsp['ret'] = (1+crsp['ret'])*(1+crsp['dlret'])-1

    crsp=crsp.drop(['dlret'], axis=1)

    # Transform ret to pct
    crsp['ret'] = crsp['ret'] * 100

    # Market equity calculation
    crsp['me']=crsp['prc'].abs()*crsp['shrout'] 
    crsp=crsp.sort_values(by=['date','permco','me'])

    ### Aggregate Market Cap ###
    # sum of me across different permno belonging to same permco in a given date
    crsp_summe = crsp.groupby(['date','permco'])['me'].sum().reset_index()

    # largest mktcap within a permco/date
    crsp_maxme = crsp.groupby(['date','permco'])['me'].max().reset_index()

    # join by jdate/maxme to find the permno
    crsp1=pd.merge(crsp, crsp_maxme, how='inner', on=['date','permco','me'])

    # drop me column and replace with the sum me
    crsp1=crsp1.drop(['me'], axis=1)

    # join with sum of me to get the correct market cap info
    crsp2=pd.merge(crsp1, crsp_summe, how='inner', on=['date','permco'])

    # sort by permno and date and also drop duplicates
    crsp2=crsp2.sort_values(by=['permno','date']).drop_duplicates()
    crsp = crsp2.copy()


    # Calculate melag and adjust for first month
    crsp['count']=crsp.groupby(['permno']).cumcount()
    crsp['melag'] = crsp.groupby('permno')['me'].shift()
    crsp['melag'] = np.where(crsp['count'] == 0, crsp['me']/(1+crsp['retx']), crsp['melag']) # ret or retx?

    crsp_info = crsp[['permno','yyyymm','prc','exchcd','me','shrcd','siccd']].to_csv(config.paths['CRSPinfoPath'], index=False)
    crsp_ret = crsp[['permno','yyyymm','ret','melag']].to_csv(config.paths['CRSPretPath'], index=False)

    print('CRSP download completed, files saved...')




