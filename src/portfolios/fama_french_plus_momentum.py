import pandas as pd
import config

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

crspmret = pd.read_csv(config.paths['ProcessedDataPath'] + '/dataset.csv', index_col=0)
print(crspmret['BMdec'])
