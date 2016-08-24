# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 18:20:19 2016

@author: lirank
"""

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np

if __name__ == '__main__':
    import os
    os.getcwd()
    os.chdir(r'C:\share\everyone\SummerCourse\Public\charity')
    train_df = pd.read_csv('./charity.train.csv.gz', compression='gzip')
    valid_df = pd.read_csv('./charity.valid.csv.gz', compression='gzip')
    train_small_df = train_df[['CHARITY_AMOUNT', 'RFA_2F', 'RFA_2A']]
    train_small_df['y'] = (train_small_df['CHARITY_AMOUNT'] > 0).astype(int)
    valid_df['y'] = (valid_df['CHARITY_AMOUNT'] > 0).astype(int)
    #glm_fit = smf.glm(formula="y~RFA_2F+RFA_2A", data=train_small_df, family=sm.families.Binomial()).fit()
    glm_fit = smf.glm(formula="y~RFA_2F+RFA_2A", data=train_small_df, family=sm.families.Binomial()).fit()
    #glm_fit.summary()
    y_hat = glm_fit.predict(valid_df)
    null_p = np.mean(valid_df['y'].astype(int))
    null_deviance = -2*(null_p * np.log(null_p)+(1-null_p)*np.log(1-null_p))
    log_loss = valid_df['y']*np.log(y_hat)+(1-valid_df['y'])*np.log(1-y_hat)
    deviance = -2*np.mean(log_loss)
    deviance/null_deviance
    
    
