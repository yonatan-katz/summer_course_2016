import pandas as pd
import numpy as np
import sklearn.ensemble
import matplotlib.pyplot as plt


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

res = {}
for n, (train_y_hat, valid_y_hat) in enumerate(y_hat_iter):
    if n+1 not in tree_num:
        continue

    get_l2 = lambda y, y_hat: np.sum((y - y_hat)**2) / np.sum((y - np.mean(y))**2)
    train_l2 = get_l2(y_train, train_y_hat)
    valid_l2 = get_l2(y_valid, valid_y_hat)
    res[n+1] = train_l2, valid_l2

pd.DataFrame(res).T.plot()

plt.barh([1,2], model.feature_importances_, align='center', alpha=0.4)
plt.yticks([1,2], columns)
plt.xlabel('feature_importances')

