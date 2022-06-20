import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.api as sm
import numpy as np
import config 

class PooledOLS():

    def __init__(self, df, x_list, y):
        self.df = df 
        self.y = y   
        self.pooled_y = df[y]
        self.pooled_x = df[x_list]
    
    def plot_df(self, x):
        
        sns.scatterplot(
            x=self.df[x], y=self.df[self.y], hue=self.df['permno'],).\
                set(title= str(x) + 'v. Stock returns' )
        plt.show()

    def regress(self):
        self.pooled_x = sm.add_constant(self.pooled_x)
        pooled_osr_model = sm.OLS(endog=self.pooled_y.astype(float), exog=self.pooled_x.astype(float))
        pooled_osr_model_results = pooled_osr_model.fit()

        with open(config.paths['resultsPath']+'/PooledOLSsummary.txt', 'w') as fh:
            fh.write(pooled_osr_model_results.summary().as_text())
        
        with open(config.paths['resultsPath']+'/PooledOLSsummary.csv', 'w') as fh:
            fh.write(pooled_osr_model_results.summary().as_csv())
        
        print('Pooled OLS Regression completed')
        self.summary = pooled_osr_model_results.summary()
       
        
        print(pooled_osr_model_results.summary())

