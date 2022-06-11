import pandas as pd
import config
import os
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
from data.data_preprocessing import download_and_unzip

# Create a class containing the joint df?

class FF5FM_Mom():
    def __init__(self):


        FF5FMurl = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip'
        FFMomurl = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip'


        if os.path.isfile(config.paths['FFPath']) == False:
            download_and_unzip(FF5FMurl, config.dataPath+'/external')    
            print('Fama French 5 Factor Model returns downloaded from Kenneth French\'s library')

        if os.path.isfile(config.paths['FFMomPath']) == False:
            download_and_unzip(FFMomurl, config.dataPath+'/external')
            print('Momentum returns downloaded from Kenneth French\'s library')

        FF5FM = pd.read_csv(config.paths['FFPath'], skiprows=3, index_col=0, skipfooter=60, engine='python')
        FFMom = pd.read_csv(config.paths['FFMomPath'], skiprows=13, index_col=0, skipfooter=100, engine='python')

        self.RF = FF5FM[['yyyymm', 'RF']]
        
        FF5FM.drop('RF', axis=1, inplace=True)
        self.returns = FF5FM.join(FFMom)
        self.returns = self.returns.reset_index().rename(columns={'index':'yyyymm'})

        del FF5FM, FFMom


""" 
    Factors are calculated as:

Rm-Rf includes all NYSE, AMEX, and NASDAQ firms. SMB, HML, RMW, and CMA for July of year t to June of t+1 
include all NYSE, AMEX, and NASDAQ stocks for which we have market equity data for December of t-1 and June of t, 
(positive) book equity data for t-1 (for SMB, HML, and RMW), non-missing revenues and at least one of the following: 
cost of goods sold, selling, general and administrative expenses, or interest expense for t-1 (for SMB and RMW), 
and total assets data for t-2 and t-1 (for SMB and CMA).
 
The Size and value factors use independent sorts of stocks into two Size groups and three B/M groups (independent 2x3 sorts). 
The Size breakpoint is the NYSE median market cap, and the B/M breakpoints are the 30th and 70th percentiles of B/M for NYSE stocks. 
The intersections of the sorts produce six VW portfolios (Fama and French, 2015)


- SMB (Small Minus Big) is the average return on the nine small stock portfolios minus the average 
return on the nine big stock portfolios
 	 	 
    SMB(B/M) = 1/3 (Small Value + Small Neutral + Small Growth) - 1/3 (Big Value + Big Neutral + Big Growth)
    #SMB(OP) = 1/3 (Small Robust + Small Neutral + Small Weak) - 1/3 (Big Robust + Big Neutral + Big Weak)
    #SMB(INV) = 1/3 (Small Conservative + Small Neutral + Small Aggressive) - 1/3 (Big Conservative + Big Neutral + Big Aggressive)

    #SMB = 1/3 (SMB(B/M) + SMB(OP) + SMB(INV))


- HML (High Minus Low) is the average return on the two value portfolios 
minus the average return on the two growth portfolios.

    HML = 1/2 (Small Value + Big Value) - 1/2 (Small Growth + Big Growth)


- RMW (Robust Minus Weak) is the average return on the two robust operating profitability 
portfolios minus the average return on the two weak operating profitability portfolios.

    RMW = 1/2 (Small Robust + Big Robust) - 1/2 (Small Weak + Big Weak)


- CMA (Conservative Minus Aggressive) is the average return on the two conservative investment 
  portfolios minus the average return on the two aggressive investment portfolios.

    1/2 (Small Conservative + Big Conservative) - 1/2 (Small Aggressive + Big Aggressive)

- MOM (Momentum) constructed by the long-short portfolio as following. Re-balanced
every month, differently from the others. Value Weighted portfolio, following Hanauer 2021

    Stocks are categorized based on the 70% and 30% percentiles
    of the past 12-2 month cumulative returns as Winner, Neutral and Loser.
    Only NYSE (big) stocks are considered for the breakpoint calculation

    1/2 (Big Win + Small Win) - 1/2 (Big Loser + Small Loser)
     
"""