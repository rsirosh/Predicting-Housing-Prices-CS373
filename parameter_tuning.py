
# coding: utf-8

# In[213]:

import pandas as pd
import numpy as np
import numpy.linalg as la
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import timeit
import math


# In[6]:

def pca(F, X):
    n, d = X.shape
    mu = np.zeros((d, 1))
    Z = np.zeros((d, F))
    for i in range(d):
        mu[i] = (1. / n) * np.sum(X[:, [i]])
    X = X - mu.T
    U, s, Vt = la.svd(X, False)
    g = s[:F]
    for i in range(F):
        g[i] = 1. / g[i]
    W = Vt[:F]
    Z = np.dot(W.T, np.diag(g))
    return (mu, Z)

def pca_proj(X,mu,Z):
    n, d = X.shape
    X = X - mu.T
    return np.dot(X, Z)


# In[191]:

def k_fold(k, model, X, y):
    n, d = X.shape
    z = np.zeros((k, 1))
    for i in range(k):
        T = list(range(int((i * n) / k), int((n * (i + 1) / k))))
        S = [j for j in range(n) if j not in T]
        curr_model = clone(model)
        curr_model.fit(X[S], y[S])
        # y[T] will be len(T) by 1
        # X[T] will be len(T) by d
        z[i] = (1. / len(T)) * np.sum((y[T] - curr_model.predict(X[T])) ** 2)
    return z


# In[192]:

def bootstrapping(B, model, X, y):
    n, d = X.shape
    z = np.zeros((B, 1))
    for i in range(B):
        u = np.random.choice(n, n, replace=True)
        S = np.unique(u)
        T = np.setdiff1d(np.arange(n), S, assume_unique=True)
        curr_model = clone(model)
        curr_model.fit(X[u], y[u])
        # y[T] will be len(T) by 1
        # X[T] will be len(T) by d
        # theta_hat will be d by 1
        z[i] = (1. / len(T)) * np.sum((y[T] - curr_model.predict(X[T])) ** 2)
    return z


# In[205]:

def evaluateModel(model, X, y, k=5, B=5):
    ########################KFOLD###################
    print('Evaluating K-fold with %d folds.' % k)
    start_time = timeit.default_timer()
    k_fold_z = k_fold(k, model, X, y)
    elapsed = timeit.default_timer() - start_time
    
    k_fold_mse = np.mean(k_fold_z)
    print('K-fold Mean Squared Error: ', k_fold_mse)
    
    k_fold_rmse = math.sqrt(k_fold_mse)
    print('K-fold Square Root Mean Squared Error: ', k_fold_rmse)

    print("Time elapsed for k-fold: ", elapsed)
    
    print()
    print()
    ###################BOOTSTRAPPING################
    print('Evaluating bootstrapping with %d bootstraps.' % B)
    start_time = timeit.default_timer()
    bootstrapping_z = bootstrapping(B, model, X, y)
    elapsed = timeit.default_timer() - start_time
    
    bootstrapping_mse = np.mean(bootstrapping_z)
    print('Bootstrapping Mean Squared Error: ', bootstrapping_mse)
    
    bootstrapping_rmse = math.sqrt(bootstrapping_mse)
    print('Bootstrapping Square Root Mean Squared Error: ', bootstrapping_rmse)

    print("Time elapsed for bootstrapping: ", elapsed)
    
    return (k_fold_z, k_fold_mse, k_fold_rmse, bootstrapping_z, bootstrapping_mse, bootstrapping_rmse)


# # Data Processing

# In[34]:

data = pd.read_csv("train.csv", header=0)
print(data.shape)

X = data.iloc[:,:-1]
Y = data.iloc[:,-1:]

print(X.shape)
print(Y.shape)


# In[38]:

# this just sums up how many nulls per feature and divides to find percentage of nulls per feature
# if over 50% null then print the feature
data_keys = X.keys()
for i, b in enumerate((X.isnull().sum() / X.shape[0]) > 0.5):
    if b:
        print(data_keys[i])


# In[8]:

# data = data.drop(['Alley', 'MiscFeature', 'Fence', 'PoolQC'], axis=1)


# In[7]:

# Replaces categorical value in Quality columns with numerical scale
qualityCols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
              'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']

data[qualityCols].head()

for col in qualityCols:
    # NA is never used since all NA's got converted to NaN objects when pandas read in the csv
    data[col] = data[col].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po':1, 'NA': 0})

data[qualityCols].head()


# In[39]:

# categorical columns
catCols = set(list(X))-set(list(X._get_numeric_data()))
print(catCols)

# #TRY dropping all cat cols
# data = data.drop(columns=catCols)


# In[40]:

#Perform one hot encoding on all categorical columns
frames = []
for col in catCols:
    oneHot_encoded = pd.get_dummies(X[col])
    oneHot_encoded = oneHot_encoded.add_prefix(col + '_is_')
    frames.append(oneHot_encoded)

X = X.drop(catCols, axis=1)

X = pd.concat(frames, axis=1)


# In[41]:

X.keys()


# In[47]:

X.isnull().values.any()


# In[48]:

# 80:20 train test ratio
test_size = 0.2
# This function splits the training and target sets into random train and test subsets.
# X_train and X_test are subsets of the training data
# y_train and y_test are subsets the the target data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

def rest(F, X, Y, X_train, y_train, X_test, y_test):
    best_ada_score = float('-inf')
    best_ada_rmse = float('inf')
    best_ada_score_f = -1
    best_ada_rmse_f = -1

    best_xg_score = float('-inf')
    best_xg_rmse = float('inf')
    best_xg_score_f = -1
    best_xg_rmse_f = -1

    best_svr_score = float('-inf')
    best_svr_rmse = float('inf')
    best_svr_score_f = -1
    best_svr_rmse_f = -1

    for f in F:
        print("\npca %d" % f)
        # # PCA Feature Selection

        X_mu, X_Z = pca(f, X.values)
        X_pca = pca_proj(X.values, X_mu, X_Z)

        X_train_mu, X_train_Z = pca(f, X_train.values)

        X_train_pca = pca_proj(X_train.values, X_train_mu, X_train_Z)

        X_test_pca = pca_proj(X_test.values, X_train_mu, X_train_Z)

        # # AdaBoost
        print("\nAdaBoost")
        from sklearn.ensemble import AdaBoostRegressor
        adaBoost = AdaBoostRegressor()
        k_z, k_mse, k_rmse, b_z, b_mse, b_rmse = evaluateModel(adaBoost, X_pca, Y.values.ravel(), k=5, B=5)
        if k_rmse < best_ada_rmse:
            best_ada_rmse = k_rmse
            best_ada_rmse_f = f

        adaBoost.fit(X_train_pca, y_train.values.ravel())
        ada_score = adaBoost.score(X_test_pca, y_test.values.ravel())
        print(ada_score)
        if ada_score > best_ada_score:
            best_ada_score = ada_score
            best_ada_score_f = f


        #View Predicted values
        predicted = adaBoost.predict(X_test_pca)
        ada_pred = y_test.copy()
        ada_pred['predicted'] = predicted
        ada_pred.head()


        # # XGBoost Regressor
        print("\nXGBoost")
        from xgboost import XGBRegressor

        xgb = XGBRegressor(max_depth=3, learning_rate=0.2, booster='gbtree', n_estimators=70)

        k_z, k_mse, k_rmse, b_z, b_mse, b_rmse = evaluateModel(xgb, X_pca, Y.values.ravel(), k=5, B=5)
        if k_rmse < best_xg_rmse:
            best_xg_rmse = k_rmse
            best_xg_rmse_f = f

        xgb.fit(X_train_pca, y_train)
        xgb_score = xgb.score(X_test_pca, y_test.values.ravel())
        print(xgb_score)
        if xgb_score > best_xg_score:
            best_xg_score = xgb_score
            best_xg_score_f = f


        predicted = xgb.predict(X_test_pca)
        xgb_pred = y_test.copy()
        xgb_pred['predicted'] = predicted
        xgb_pred.head()


        # # SVM (SVR)
        print("\nSVR")
        from sklearn import svm

        svr_model = svm.SVR(kernel="poly", coef0=-3500, gamma='scale')
        # coef0 only works with poly and sigmoid kernels
        # it just puts that value instead of the column of 1's

        # without it, this model breaks for some reason

        k_z, k_mse, k_rmse, b_z, b_mse, b_rmse = evaluateModel(svr_model, X_pca, Y.values.ravel(), k=5, B=5)
        if k_rmse < best_svr_rmse:
            best_svr_rmse = k_rmse
            best_svr_rmse_f = f

        # epsilon, degree
        svr_model.fit(X_train_pca, y_train.values.ravel())
        svr_score = svr_model.score(X_test_pca, y_test.values.ravel())
        print(svr_score)
        if svr_score > best_svr_score:
            best_svr_score = svr_score
            best_svr_score_f = f

        svr_predicted = svr_model.predict(X_test_pca)
        svr_pred = y_test.copy()
        svr_pred["predicted"] = svr_predicted
        svr_pred.head()

    return ((best_ada_score, best_ada_score_f), (best_ada_rmse, best_ada_rmse_f), (best_xg_score, best_xg_score_f), (best_xg_rmse, best_xg_rmse_f), (best_svr_score, best_svr_score_f), (best_svr_rmse, best_svr_rmse_f))

print(rest([50, 55, 60, 65, 70], X, Y, X_train, y_train, X_test, y_test))
