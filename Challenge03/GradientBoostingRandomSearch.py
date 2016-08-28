import pandas as pd
import numpy as np
import sklearn.ensemble
import matplotlib.pyplot as plt
import os
import itertools

the_dir = r'my\dir'
train_df = pd.read_csv(os.path.join(the_dir, 'Waves.train.csv.gz'), compression='gzip')
valid_df = pd.read_csv(os.path.join(the_dir, 'Waves.valid.csv.gz'), compression='gzip')

columns = ['passenger_count', 'trip_distance']
def create_sklearn_compatible_x_y(df):
    X = df[columns]
    y = df.y.values
    return X, y

X_train, y_train = create_sklearn_compatible_x_y(train_df)
X_valid, y_valid = create_sklearn_compatible_x_y(valid_df)

train_df.iloc[0]

####### Train gradient boosting
max_estimators = 100
tree_num = np.linspace(max_estimators/10, max_estimators, max_estimators/10)

model = sklearn.ensemble.GradientBoostingRegressor(n_estimators=max_estimators,
                                                   subsample=.5,
                                                   max_depth=2)
model.fit(X_train, y_train)

# returns an array, where the n'th cell contains the prediction with n+1 boosted trees.
train_staged_y_hat = model.staged_predict(X_train)
valid_staged_y_hat = model.staged_predict(X_valid)
y_hat_iter = itertools.izip(train_staged_y_hat, valid_staged_y_hat)

get_l2 = lambda y, y_hat: np.sum((y - y_hat)**2) / np.sum((y - np.mean(y))**2)

res = {}
for n, (train_y_hat, valid_y_hat) in enumerate(y_hat_iter):
    if n+1 not in tree_num:
        continue

    train_l2 = get_l2(y_train, train_y_hat)
    valid_l2 = get_l2(y_valid, valid_y_hat)
    res[n+1] = train_l2, valid_l2

pd.DataFrame(res).T.plot()

plt.barh([1,2], model.feature_importances_, align='center', alpha=0.4)
plt.yticks([1,2], columns)
plt.xlabel('feature_importances')


####### Random grid search example
res = {}
for n in range(20):
    print n
    max_depth = np.random.randint(1,3) # 1 or 2
    subsample = np.random.uniform(0, 1)

    model = sklearn.ensemble.GradientBoostingRegressor(n_estimators=10,
                                                       subsample=subsample,
                                                       max_depth=max_depth)

    model.fit(X_train, y_train)

    # returns an array, where the n'th cell contains the prediction with n+1 boosted trees.
    train_y_hat = model.predict(X_train)
    valid_y_hat = model.predict(X_valid)
    X_train

    train_l2 = get_l2(y_train, train_y_hat)
    valid_l2 = get_l2(y_valid, valid_y_hat)
    res[max_depth, subsample] = train_l2, valid_l2

res_df = pd.DataFrame(res).T.reset_index()
res_df.columns = ['max_depth', 'subsample', 'train_l2', 'valid_l2']
fig, ax = plt.subplots(2, 1)
ax[0].scatter(res_df.max_depth, res_df.valid_l2)
ax[1].scatter(res_df.subsample, res_df.valid_l2)
