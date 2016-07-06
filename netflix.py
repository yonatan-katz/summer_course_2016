# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:30:01 2016

@author: test
"""
import pandas as pd

import numpy as np

import statsmodels.formula.api as sm

import matplotlib.pyplot as plt

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
    
    return x, y, y_dates, movies
    
def print_same_stat():    
    x,y,dates,movies = load_data()
    
    g = x.iloc[:,range(14)]
    
    print g.mean()
    
    print g.apply(lambda x: x.corr(y[0]))
    
    print x.apply(lambda t: len(t[t==0].dropna()),axis=0)   
    
    print "Y vs Date corr:", np.corrcoef (y[0],dates[0])[0][1]
    
def create_test_train_set(x,y):
    
    test_sample_id = np.random.choice(range(len(x)), size=2000,replace=False)
    
    test_x = x.iloc[test_sample_id,:]
    
    test_y = y.iloc[test_sample_id,:]
    
    train_x = x.drop(test_sample_id, 0)
    
    train_y = y.drop(test_sample_id, 0)    
    
    return test_x, train_x, test_y, train_y
    
    
def regression_test():    
    
    x,y,dates,movies = load_data()
    
    test_x, train_x, test_y, train_y = create_test_train_set(x, y) 
    
    #Since upper case OLS doesn't add intecept automatically
    #add it manually
    test_x.loc[:,("intercept")] = np.ones(len(test_x))
    
    train_x.loc[:,("intercept")] = np.ones(len(train_x))
    
        
    '''
        Base line RMSE
    '''
    print "Base line RMSE:", np.sqrt(np.mean((test_y[0] - np.mean(train_y[0]))**2))    
    
    res = sm.OLS(train_y, train_x).fit()  
    
    print res.summary()
    
    
    in_sample = res.predict(train_x)
    
    print "In sample RMSE:", np.sqrt(np.mean((train_y[0] - in_sample)**2))
    
    out_sample = res.predict(test_x)
    
    print "Out sample RMSE:", np.sqrt(np.mean((test_y[0] - out_sample)**2))
    
    out_sample[out_sample>5] = 5
    
    print "Clipped out sample RMSE:", np.sqrt(np.mean((test_y[0] - out_sample)**2))    
    
    #intercept column index is 99
    test_x = test_x.iloc[:, range(14) + [99]]
    
    train_x = train_x.iloc[:, range(14) + [99]]
    
    res = sm.OLS(train_y, train_x).fit() 
    
    print res.summary()
    
    out_sample = res.predict(test_x)
    
    print "Completed data (14 movies) RMSE:", np.sqrt(np.mean((test_y[0] - out_sample)**2))
    
    
def pca_regression_test():
    from sklearn.decomposition import PCA    
    
    x,y,dates,movies = load_data()
    
    x = x - np.mean(x)       
    
    x = x.ix[:, range(14)]
    
    pca = PCA(n_components=3)
    
    fitted = pca.fit(x)
    
    loadings = pd.DataFrame({"Comp1":fitted.components_[0], 
                             "Comp2":fitted.components_[1],
                             "Comp3":fitted.components_[2] })
    
    loadings = loadings.set_index(movies.ix[:,1].values[0:14])
    
    print "Loadings:", loadings
    
    stdev = pd.Series(np.sqrt(fitted.explained_variance_), index=["Comp1", "Comp2", "Comp3"])
    
    print "stdev:", stdev
    
    transformed_x = fitted.fit_transform(x)
    
    f, ax = plt.subplots(2,2)
    ax[0,0].set_title("Comp1 ~ Comp2")    
    ax[0,0].scatter(transformed_x[range(100),0], transformed_x[range(100),1])
    
    ax[0,1].set_title("Comp2 ~ Comp3")
    ax[0,1].scatter(transformed_x[range(100),1], transformed_x[range(100),2])
    
    ax[1,0].set_title("Comp1 ~ Comp3")
    ax[1,0].scatter(transformed_x[range(100),0], transformed_x[range(100),2])     
    
    f, ax = plt.subplots(2,1)  
    
    ax[0].set_title("Most Low Comp1 scores")
    ax[0].hist(y.ix[transformed_x.argsort(0)[range(100),0]].values)
    
    
    ax[1].set_title("Most high Comp1 scores")
    #get sorted values in the reversed order and take first 100 elements
    ax[1].hist(y.ix[transformed_x.argsort(0)[::-1,0]][0:100].values)
    
    plt.tight_layout()
    plt.show()    
    
    test_x, train_x, test_y, train_y = \
        create_test_train_set(\
            pd.DataFrame(\
                transformed_x,columns=["c1","c2", "c3"]), y)    
    
    tr = train_x.ix[:, ["c1","c2"]]
    tr["y"] = train_y
    
    #low case ols adds intercept automatically
    res = sm.ols(formula="y~c1+c2",data=tr).fit()
    
    print res.summary()
    
def multiple_pca_regression_test():
    
    from sklearn.decomposition import PCA
    
    x,y,dates,movies = load_data()
    
    x = x - np.mean(x)       
    
    x = x.ix[:, range(14)]
    
    pca = PCA(n_components=14)
    
    fitted = pca.fit(x)
    
    transformed_x = fitted.fit_transform(x)
    
    columns = map(lambda x: "c"+str(x), range(14))
    
    test_x, train_x, test_y, train_y = \
        create_test_train_set(\
           pd.DataFrame(\
             transformed_x,columns=columns), y) 
   
    train = train_x
    test = test_x
    train["intercept"] = np.ones(len(train_x))
    test["intercept"] = np.ones(len(test_x))

    
    for r in range(1, 14):
        columns_x = map(lambda x: "c"+str(x), range(r)) 
        tr = train.ix[:, ["intercept"] + columns_x]
        va = test.ix[:,  ["intercept"] + columns_x]
        
        res = sm.OLS(train_y, tr).fit()
    
        va.pred = res.predict(va)
    
        pca_error =  np.sqrt(np.mean((test_y.ix[:,0] -  va.pred)**2))
    
        print "PCA component:", r, ", validation error:", pca_error
        
        
def ridge_regression_test():
     from sklearn.linear_model import Ridge
    
     x,y,dates,movies = load_data()
                
     test_x, train_x, test_y, train_y = create_test_train_set(x, y)  
     
     alpha_vals = np.exp(np.arange(start=-15, stop=10,step=0.1))
     
     f, ax = plt.subplots(2,3)
     
     plot_num = 0
           
     for nuse in [50, 500, 5000]:
        
         use_id = np.random.choice(train_x.index,size=nuse,replace=False)
                               
         tx = train_x.ix[use_id,range(14)]
         
         ty = train_y.ix[use_id]
         
         to_plot_rmse = []

         to_plot_coef = []
         
         for alpha in alpha_vals:
            
            fit = Ridge(alpha=alpha).fit(X=tx,y=ty)
            
            pred = fit.predict(test_x.ix[:, range(14)])
            
            res = test_y - pred
            
            RSS = np.sum(res**2)
            
            RMSE = np.sqrt(RSS / len(test_y))
            
            to_plot_rmse.append(RMSE)
            
            to_plot_coef.append(np.sum(fit.coef_**2))
                       
         ax[0, plot_num].set_title("Ridge using %d training data" % nuse)
         ax[0, plot_num].set_xscale("log")
         ax[0, plot_num].set_xlabel("alpha")
         ax[0, plot_num].set_ylabel("RMSE")
         ax[0, plot_num].scatter(alpha_vals, to_plot_rmse)
         
         ax[1, plot_num].set_xscale("linear")
         ax[1, plot_num].set_xlabel("sum of coef")
         ax[1, plot_num].set_ylabel("RMSE")
         ax[1, plot_num].scatter(to_plot_coef, to_plot_rmse)
                     
         plot_num += 1
         
         
     plt.show()
        
     


def lasso_regression_test():
         
     from sklearn.linear_model import Lasso
    
     x,y,dates,movies = load_data()
                
     test_x, train_x, test_y, train_y = create_test_train_set(x, y)  
     
     alpha_vals = np.arange(start=10e-5, stop=1,step=0.01)

     f, ax = plt.subplots(2,3)
     
     plot_num = 0     
    
    
     for nuse in [50,500,5000]:
        
         use_id = np.random.choice(train_x.index,size=nuse,replace=False)
                               
         tx = train_x.ix[use_id,range(14)]
         
         ty = train_y.ix[use_id]
         
         to_plot_rmse = []

         to_plot_coef = []
         
         for alpha in alpha_vals:
            
            fit = Lasso(alpha=alpha).fit(X=tx,y=ty)
            
            pred = fit.predict(test_x.ix[:, range(14)])
            
            res = test_y.ix[:,0] - pred #just to align matrix shapes
            
            RSS = np.sum(res**2)
            
            RMSE = np.sqrt(RSS / len(test_y))
            
            to_plot_rmse.append(RMSE)
            
            to_plot_coef.append(np.sum(fit.coef_**2))
                       
         ax[0, plot_num].set_title("Lasso using %d training data" % nuse)
         ax[0, plot_num].set_xscale("linear")
         ax[0, plot_num].set_xlabel("alpha")
         ax[0, plot_num].set_ylabel("RMSE")
         ax[0, plot_num].scatter(alpha_vals, to_plot_rmse)
         
         ax[1, plot_num].set_xscale("linear")
         ax[1, plot_num].set_xlabel("sum of coef")
         ax[1, plot_num].set_ylabel("RMSE")
         ax[1, plot_num].scatter(to_plot_coef, to_plot_rmse)
                     
         plot_num += 1
         
         
     plt.show()
         
         
         
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
