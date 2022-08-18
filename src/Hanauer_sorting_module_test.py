from portfolios.PortfolioCreation import Portfolio
import os, sys
import config
import pandas as pd

crspret = pd.read_csv(config.paths['CRSPretPath'])
crspinfo = pd.read_csv(config.paths['CRSPinfoPath'])
# This file was created by CMPT_ME_PRIOR_RETS using the 202205 CRSP database.  It
# contains a momentum factor, constructed from six value-weight portfolios formed
# using independent sorts on size and prior return of NYSE, AMEX, and NASDAQ stocks.
# Mom    is the average of the returns on two (big and small) high prior return portfolios
# minus the average of the returns on two low prior return portfolios.  The portfolios
# are constructed monthly.  Big means a firm is above the median market cap on the NYSE
# at the end of the previous month; small firms are below the median NYSE market cap.
# Prior return is measured from month -12 to - 2.  Firms in the low prior return
# portfolio are below the 30th NYSE percentile.  Those in the high portfolio are above
# the 70th NYSE percentile.
"""
    Six VW portfolios formed using independent sorts on size and prior return of NYSE, AMEX and NASDAQ stocks (exchcd = 1,2,3)
    Big firms: above median NYSE market cap in the prior month. Small firms lower than median NYSE market cap.

    Prior ret is measured from month -12 to -2.
    Low mom and high mom are respectively under the 30th NYSE prior ret percentile and above the 70th.

    1/2(Big High Mom + Small High Mom) - 1/2(Big Low Mom + Small Low Mom)
    
    
"""

df = pd.merge(crspret, crspinfo, on=['permno', 'yyyymm'])
df_filtered = df.loc[df['exchcd'].isin([1])].copy()

# print(df_filtered.groupby('yyyymm')['permno'].count())
df_filtered['Median_Me'] = df_filtered.groupby('yyyymm')['melag'].transform('median')

# def Mom_ret(df):
print(df.groupby('permno')['ret'].tail(12).cumprod())

