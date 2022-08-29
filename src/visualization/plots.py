import pandas as pd
import config
import matplotlib.pyplot as plt
import seaborn as sns

"""
Add parser for files which we want to plot, or just define a function to do the plot?
IDEA: Define the function, and inside the if __name__ == __main__ insert the snippet with argparse
for a plot being done directly

from argparse import ArgumentParser
import argparse

parser = ArgumentParser()

parser.add_argument('--cumulativeReturn', action='store_true', help="")

args, unknown = parser.parse_known_args() # Using this to avoid error with notebooks

"""

def plot_cumulative_returns(predicted_ret_df):
    crsp = pd.read_csv(config.paths['CRSPretPath'])
    # crsp['ret'] = crsp['ret']/100
    crsp['pweight'] = ((crsp['melag'])/(crsp.groupby(['yyyymm'])['melag'].transform('sum')))
    crsp['pret'] = crsp['pweight'] * crsp['ret']
    df = crsp.groupby('yyyymm')['pret'].agg('sum')
    df = df.reset_index()
    df = df.loc[df['yyyymm'] >= 199501]
    df['cumret'] = ((df['pret']+1).cumprod())-1
    print(df.tail(5))
    
    plt.plot([1,2,3], [1,2,3])
    plt.show(block=True)