# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:30:01 2016

@author: test
"""
import pandas as pd

import numpy as np

import statsmodels.formula.api as sm

from IPython import get_ipython

IPYTHON = get_ipython()

IPYTHON.magic('load_ext autoreload')
IPYTHON.magic('autoreload 2')



def load_data():
    x = pd.read_csv("./data/train_ratings_all.dat.txt", sep=" ",header=None)
    
    y = pd.read_csv("./data/train_y_rating.dat.txt", sep=" ",header=None)
    
    movies = pd.read_csv("./data/movie_titles.txt", sep=",",header=None)
    
    y_dates = pd.read_csv("./data/train_y_dates.dat.txt", sep=",",header=None)
    
    x.columns = movies.ix[:, 1]    
    
    return x, y, y_dates
    
def print_same_stat():    
    x,y,dates = load_data()
    
    g = x.iloc[:,range(14)]
    
    print g.mean()
    
    print g.apply(lambda x: x.corr(y[0]))
    
    print x.apply(lambda t: len(t[t==0].dropna()),axis=0)   
    
    print "Y vs Date corr:", np.corrcoef (y[0],dates[0])[0][1]
    
    
def regresion_test():    
    
    x,y,dates = load_data()
    
    test_sample_id = np.random.choice(range(len(x)), size=2000,replace=False)
    
    test_x = x.iloc[test_sample_id,:]
    
    test_y = y.iloc[test_sample_id,:]
    
    train_x = x.drop(test_sample_id, 0)
    
    train_y = y.drop(test_sample_id, 0)   
    
    '''
        Base line RMSE
    '''
    print "Base line RMSE:", np.sqrt(np.mean((test_y - np.mean(train_y))**2))
    
    res = sm.OLS(train_y, train_x).fit()   
    
    in_sample = res.predict(train_x)
    
    print "In sample RMSE:", np.sqrt(np.mean((train_y[0] - in_sample)**2))
    
    out_sample = res.predict(test_x)
    
    print "Out sample RMSE:", np.sqrt(np.mean((test_y[0] - out_sample)**2))
    
    out_sample[out_sample>5] = 5
    
    print "Clipped out sample RMSE:", np.sqrt(np.mean((test_y[0] - out_sample)**2))
    
    
    test_x = x.iloc[test_sample_id,range(14)]
    
    test_y = y.iloc[test_sample_id]
    
    train_x = x.drop(test_sample_id, 0).iloc[:,range(14)]
    
    train_y = y.drop(test_sample_id, 0)
    
    res = sm.OLS(train_y, train_x).fit()   
    
    out_sample = res.predict(test_x)
    
    print "Completed data (14 movies) RMSE:", np.sqrt(np.mean((test_y[0] - out_sample)**2))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    