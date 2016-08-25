# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 18:20:19 2016

@author: lirank
"""

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import sklearn.ensemble
import sklearn.feature_extraction
import os
 
os.chdir(the_dir)
 
train_df = pd.read_csv('./charity.train.csv.gz', compression='gzip')
valid_df = pd.read_csv('./charity.valid.csv.gz', compression='gzip')
 
def get_quantiles(x, num_levels):
    quantiles = np.linspace(0, 1, num_levels)[1:]
    return x.quantile(quantiles)
 
def thermo_enc(df, col, levels):
    for n, level in enumerate(levels):
        df['{}_t{}'.format(col, n)] = np.maximum(0, df[col] - level)
    return df
 
quant = get_quantiles(train_df.MAXRAMNT, 5)
def prepare_df(df):
    df = df[['CHARITY_AMOUNT', 'MAXRAMNT', 'RFA_3']]
    df['y'] = (df['CHARITY_AMOUNT'] > 0).astype(int)
    df['RFA_3R'] = df.RFA_3.apply(lambda x: x[0] if len(x)>1 else 'X')
    df['RFA_3F'] = df.RFA_3.apply(lambda x: int(x[1]) if len(x)>1 else -1)
    df['RFA_3A'] = df.RFA_3.apply(lambda x: ord(x[2])-ord('A') if len(x)>1 else -1)
    df = thermo_enc(df, 'MAXRAMNT', quant.values)
    df = df.drop(['CHARITY_AMOUNT', 'RFA_3'], axis=1)
    return df
 
def calculate_deviance(y_hat, y):
    null_p = np.mean(y.astype(int))
    null_deviance = -2*(null_p * np.log(null_p)+(1-null_p)*np.log(1-null_p))
    log_loss = y*np.log(y_hat)+(1-y)*np.log(1-y_hat)
    deviance = -2*np.mean(log_loss)
    return deviance/null_deviance
 
train_small_df = prepare_df(train_df)
valid_small_df = prepare_df(valid_df)
 
####### Step 1: Logistic regression without interactions
glm_fit = smf.glm(formula='y~MAXRAMNT+RFA_3R+RFA_3F+RFA_3A',
                  data=train_small_df, family=sm.families.Binomial()).fit()
y_hat = glm_fit.predict(valid_small_df)
print calculate_deviance(y_hat, valid_small_df.y)
# 0.989715
 
####### Step 2: Logistic regression with thermometer encoding
cols = 'MAXRAMNT+MAXRAMNT_t0+MAXRAMNT_t1+MAXRAMNT_t2+MAXRAMNT_t3+RFA_3R+RFA_3F+RFA_3A'
glm_fit = smf.glm(formula='y~'+cols, data=train_small_df, family=sm.families.Binomial()).fit()
y_hat = glm_fit.predict(valid_small_df)
print calculate_deviance(y_hat, valid_small_df.y)
# 989925510749
 
# Sklearn input is different - need to create X and y as numpy arrays
# Also need to take care of categorical x
enc = sklearn.feature_extraction.DictVectorizer()
enc.fit([{'tmp':x} for x in train_small_df.RFA_3R.values])
def create_sklearn_compatible_x_y(df):
    x1 = df[['MAXRAMNT', 'RFA_3F', 'RFA_3A']].values
    x2 = enc.transform([{'tmp':x} for x in df.RFA_3R.values]).toarray()
    X = np.concatenate([x1, x2], axis=1)
    y = df.y.values
    return X, y
X_train, y_train = create_sklearn_compatible_x_y(train_small_df)
X_valid, y_valid = create_sklearn_compatible_x_y(valid_small_df)
 
####### Step 3: Gradient boosting
max_depth = 2
learning_rate = .1
n_estimators = 50
 
model = sklearn.ensemble.GradientBoostingClassifier(loss='deviance',
    learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
model.fit(X_train, y_train)
y_hat = model.predict_proba(X_valid)
print calculate_deviance(y_hat[:, 1], valid_small_df.y)
# 0.98908
