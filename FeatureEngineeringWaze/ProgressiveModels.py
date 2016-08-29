import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as gbr
import sklearn.cross_validation
import matplotlib.pyplot as plt
import os
import itertools

########### Reading the data
the_dir = r'my\dir'

def prepare_df(file_name):
    df = pd.read_csv(file_name, compression='gzip')
    df['y'] = df.cnt.apply(np.log10)
    # 0 is sunday, 6 is saturday -> make them adjacent
    df['weekday_sep'] = df.weekday.apply(lambda x: (x+1)%7)
    return df

# Shuffle original train and test
df1 = prepare_df(os.path.join(the_dir, 'CountPrediction.train.csv.gz'))
df2 = prepare_df(os.path.join(the_dir, 'CountPrediction.test.csv.gz'))
df = pd.concat([df1, df2])
df = df.reset_index(drop=True)
df = df.reindex(np.random.permutation(df.index))
train_df = df[:len(df1)]
train_df = train_df.reset_index(drop=True)
test_df = df[len(df1)+1:]
test_df = test_df.reset_index(drop=True)

########### Weather baseline
get_l2 = lambda y, y_hat: np.sum((y - y_hat)**2) / np.sum((y - np.mean(y))**2)

def create_sklearn_compatible_x_y(df, columns):
    X = df[columns].values
    y = df.y.values
    return X, y

max_estimators = 500
tree_num = np.round(np.linspace(20, max_estimators, 50))
model = gbr(n_estimators=max_estimators, max_depth=2)

columns = ['temp', 'atemp', 'hum', 'windspeed']
X_train, y_train = create_sklearn_compatible_x_y(train_df, columns)
X_test, y_test = create_sklearn_compatible_x_y(test_df, columns)

model.fit(X_train, y_train)

def plot_l2_vs_estimator_num(X_train, y_train, X_test, y_test):
    train_staged_y_hat = model.staged_predict(X_train)
    test_staged_y_hat = model.staged_predict(X_test)
    y_hat_iter = itertools.izip(train_staged_y_hat, test_staged_y_hat)

    get_l2 = lambda y, y_hat: np.sum((y - y_hat)**2) / np.sum((y - np.mean(y))**2)

    res = {}
    for n, (train_y_hat, test_y_hat) in enumerate(y_hat_iter):
        if n+1 not in tree_num:
            continue

        train_l2 = get_l2(y_train, train_y_hat)
        test_l2 = get_l2(y_test, test_y_hat)
        res[n+1] = train_l2, test_l2

    res_df = pd.DataFrame(res).T
    res_df.columns = ['train_l2', 'test_l2']
    res_df.plot()
    print res_df.min()

plot_l2_vs_estimator_num(X_train, y_train, X_test, y_test)
#train_l2    0.686991
#test_l2     0.729482

########### day + hr as "is"
columns = ['temp', 'atemp', 'hum', 'windspeed', 'weekday', 'hr']
X_train, y_train = create_sklearn_compatible_x_y(train_df, columns)
X_test, y_test = create_sklearn_compatible_x_y(test_df, columns)

model = gbr(n_estimators=max_estimators, max_depth=3)
model.fit(X_train, y_train)
plot_l2_vs_estimator_num(X_train, y_train, X_test, y_test)
#train_l2    0.086294
#test_l2     0.105707

########### day + hr as day weekday vs weekend
columns = ['temp', 'atemp', 'hum', 'windspeed', 'weekday_sep', 'hr']
X_train, y_train = create_sklearn_compatible_x_y(train_df, columns)
X_test, y_test = create_sklearn_compatible_x_y(test_df, columns)

model = gbr(n_estimators=max_estimators, max_depth=3)
model.fit(X_train, y_train)
plot_l2_vs_estimator_num(X_train, y_train, X_test, y_test)
#train_l2    0.086507
#test_l2     0.104322

###########  day + hr as pre-computation
# Add synthetic feature from day and hour only
fold_num = 10
columns = ['weekday_sep', 'hr']
X_train, y_train = create_sklearn_compatible_x_y(train_df, columns)
X_test, y_test = create_sklearn_compatible_x_y(test_df, columns)

# Creating training values
# Note that the hyper-params should be adjusted as well
kf = sklearn.cross_validation.KFold(n=len(train_df), n_folds=fold_num)
train_df['week_hr_y'] = np.nan
for train_index, valid_index in kf:
    X_train_fold, X_valid_fold = X_train[train_index], X_train[valid_index]
    y_train_fold, y_valid_fold = y_train[train_index], y_train[valid_index]

    model = gbr(n_estimators=max_estimators, max_depth=3)
    model.fit(X_train_fold, y_train_fold)
    train_df.loc[valid_index, 'week_hr_y'] = model.predict(X_valid_fold)

# Now train over the entire training set and create the values for the test set
model = gbr(n_estimators=max_estimators, max_depth=3)
model.fit(X_train, y_train)
test_df.loc[:, 'week_hr_y'] = model.predict(X_test)

# Now re-train as usual with the new feature
columns = ['temp', 'atemp', 'hum', 'windspeed', 'week_hr_y']
X_train, y_train = create_sklearn_compatible_x_y(train_df, columns)
X_test, y_test = create_sklearn_compatible_x_y(test_df, columns)

model = gbr(n_estimators=max_estimators, max_depth=3)
model.fit(X_train, y_train)
plot_l2_vs_estimator_num(X_train, y_train, X_test, y_test)
#train_l2    0.082016
#test_l2     0.109222
