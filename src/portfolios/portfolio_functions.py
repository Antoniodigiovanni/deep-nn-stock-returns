def calculate_portfolio_monthly_return(df):
    #years = df['yyyymm'].unique()
    #pret = df.group 
    t = df.groupby(['yyyymm', 'permno','bin']).apply(lambda x: x['pweight'] * x['ret']).reset_index(name='pret').drop('level_3', axis=1)
    df = df.merge(t, on=['yyyymm', 'permno', 'bin'], how='left')

    portfolio = df.groupby(['yyyymm', 'bin'])['pret'].sum().reset_index(name='pret')
    
    # Convert to wide format in order to ease calculations
    portfolio = portfolio.pivot(index='yyyymm', columns='bin', values='pret').reset_index().rename_axis(None, axis=1)


    # Calculating long-short returns for quantiles
    portfolio[(str(portfolio.columns[-1])+'-'+str(portfolio.columns[1]))] = portfolio.iloc[:,-1] - portfolio.iloc[:,1]
    return portfolio



def calculate_portfolio_weights(df, weighting = 'VW'):
    if weighting == 'VW':
        if 'melag' in df.columns:
            df['pweight'] = df['melag']/df.groupby('yyyymm')['melag'].transform('sum')        
        else:
            print('melag is not present in the df')
    
    elif weighting == 'EW':
        temp = df.groupby(['yyyymm', 'bin']).size().reset_index(name='count')
        df = df.merge(temp, on=['yyyymm','bin'], how='left')
        
        df['pweight'] = 1/df['count']
        df = df.drop('count', axis=1)
    
    return df

def calculate_information_ratio():
    pass