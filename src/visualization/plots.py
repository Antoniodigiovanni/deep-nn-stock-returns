import pandas as pd
import config
import matplotlib.pyplot as plt
import seaborn as sns


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