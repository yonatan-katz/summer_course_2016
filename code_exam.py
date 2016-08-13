"""Simple regression code example"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as sm



def load_data(fname):
    df = pd.DataFrame.from_csv(fname,sep='\t')    
    
    return df
    
def summary(r):
     print "Min, Mean, Max" , np.min(r), np.mean(r), np.max(r)
    
     print "1stQu, 3stQu, Median", np.percentile(r, [25, 75]), np.median(r)
    
    
def make_regression(fname="./data/prostate.data.txt"):
    
    data = load_data(fname)   
    
    print data.describe(percentiles = [.25, .5, .75],)       
    
    print "\n\n\n******************* Split data set  ******************************"
    
    train = data[data.train == 'T'].drop('train',1)
    
    test = data[data.train == 'F'].drop('train',1)          
    
    train['intercept'] = np.ones((len(train), ))

    test['intercept'] = np.ones((len(test), ))
    
    x_train = train.drop('lpsa',1)

    x_test = test.drop('lpsa',1)
    
    y_train = train['lpsa']

    y_test = test['lpsa']
    
    print "\n\n\n******************* Make regression *********************************"
    
    res = sm.OLS(y_train, x_train).fit()

    print res.summary()
    
    print "\n\n\n******************* Prediction Error Stat ******************************"
    
    predict = res.predict(x_test)

    mse = (y_test - predict)**2
    
    summary(mse)  
    
    
