
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
import timeit
import math


# In[3]:

data = pd.read_csv("train.csv", header=0)
data.shape


# In[4]:

# this just sums up how many nulls per feature and divides to find percentage of nulls per feature
# if over 50% null then print the feature
data_keys = data.keys()
for i, b in enumerate((data.isnull().sum() / data.shape[0]) > 0.5):
    if b:
        print(data_keys[i])


# In[5]:

data = data.drop(['Alley', 'MiscFeature', 'Fence', 'PoolQC'], axis=1)


# In[6]:

data.head()


# In[7]:

# Replaces categorical value in Quality columns with numerical scale
qualityCols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
              'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']

data[qualityCols].head()

for col in qualityCols:
    # NA is never used since all NA's got converted to NaN objects when pandas read in the csv
    data[col] = data[col].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po':1, 'NA': 0})

data[qualityCols].head()


# In[8]:

# categorical columns
catCols = set(list(data))-set(list(data._get_numeric_data()))
print(catCols)

# #TRY dropping all cat cols
# data = data.drop(columns=catCols)


# In[9]:

#Perform one hot encoding on all categorical columns
frames = []
salePrice = data['SalePrice']
for col in catCols:
    oneHot_encoded = pd.get_dummies(data[col])
    oneHot_encoded = oneHot_encoded.add_prefix(col + '_is_')
    frames.append(oneHot_encoded)
frames.append(salePrice)

data = data.drop(catCols, axis=1)

data = pd.concat(frames, axis=1)


# In[10]:

data.keys()


# In[11]:

data.isnull().values.any()


# In[12]:

# Drop all Na's for now
data = data.dropna()
# Split into training and target sets
num_variables = len(data.columns)
training = data.iloc[:, 0:num_variables-1]
target = data.iloc[:,num_variables-1:]

# 80:20 train test ratio
test_size = 0.2
# This function splits the training and target sets into random train and test subsets.
# X_train and X_test are subsets of the training data
# y_train and y_test are subsets the the target data
# do we do this ourselves or should we be using scikit learn
X_train, X_test, y_train, y_test = train_test_split(training, target, test_size=test_size)


# In[13]:

def k_fold(k, model, X, y):
    n, d = X.shape
    z = np.zeros((k, 1))
    for i in range(k):
        T = list(range(int((i * n) / k), int((n * (i + 1) / k))))
        S = [j for j in range(n) if j not in T]
        model.fit(X[S], y[S])
        # y[T] will be len(T) by 1
        # X[T] will be len(T) by d
        z[i] = (1. / len(T)) * np.sum((y[T] - model.predict(X[T])) ** 2)
    return z


# In[14]:

def evaluateModel(model, splits=5):
    start_time = timeit.default_timer()
    
#     mae = cross_val_score(model, X_test, y_test.values.ravel(), cv=splits, scoring='neg_mean_absolute_error')
#     mae = np.mean(mae)
#     print('Mean Absolute Error: ', -mae)
    
    mse = cross_val_score(model, X_test, y_test.values.ravel(), cv=splits, scoring='neg_mean_squared_error')
    print('Mean Squared Error: ', np.mean(mse * -1))
    
    rmse = math.sqrt(np.mean(mse*-1))
    print('Root Mean Squared Error: ', rmse)

    elapsed = timeit.default_timer() - start_time


# # AdaBoost

# In[15]:

from sklearn.ensemble import AdaBoostRegressor
adaBoost = AdaBoostRegressor()
adaBoost.fit(X_train, y_train.values.ravel())


# In[16]:

evaluateModel(adaBoost)


# In[17]:

#View Predicted values
predicted = adaBoost.predict(X_test)
ada_pred = y_test.copy()
ada_pred['predicted'] = predicted
ada_pred.head()


# In[18]:

ada_z = k_fold(5, adaBoost, training.values, target.values.ravel())
np.mean(ada_z)


# # XGBoost Regressor

# In[19]:

#!pip3 install xgboost


# In[20]:

from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
evaluateModel(xgb)


# In[21]:

predicted = xgb.predict(X_test)
xgb_pred = y_test.copy()
xgb_pred['predicted'] = predicted
xgb_pred.head()


# In[22]:

xgb_z = k_fold(5, xgb, training.values, target.values.ravel())
np.mean(xgb_z)


# # SVM (SVC just to test)

# In[23]:

from sklearn import svm

svc_model = svm.SVC(kernel="rbf", C=1.0)

# change C (error) in hypertuning
svc_model.fit(X_train, y_train.values.ravel())


# In[24]:

# we get a warning because svm is splitting the data into "classes" and because saleprice is numeric there are many prices where there are only 1 of that "class"...
evaluateModel(svc_model, splits=5)
svc_predicted = svc_model.predict(X_test)
svc_pred = y_test.copy()
svc_pred["predicted"] = svc_predicted
svc_pred.head()


# In[25]:

svc_z = k_fold(5, svc_model, training.values, target.values.ravel())
np.mean(svc_z)


# # SVM (SVR)

# In[26]:

from sklearn import svm

svr_model = svm.SVR(kernel="poly", shrinking=False, coef0=-2000)
# coef0 only works with poly and sigmoid kernels
# it just puts that value instead of the column of 1's

# without it, this model breaks for some reason

# epsilon, degree
svr_model.fit(X_train, y_train.values.ravel())


# In[27]:

evaluateModel(svr_model, splits=5)
svr_predicted = svr_model.predict(X_test)
svr_pred = y_test.copy()
svr_pred["predicted"] = svr_predicted
svr_pred.head()


# In[28]:

svr_z = k_fold(5, svr_model, training.values, target.values.ravel())
np.mean(svr_z)


# In[29]:

# looks like order of least to greatest error with the currently tuned models goes:
# XgBoost, SVR, AdaBoost, SVC
print(np.mean(ada_z))
print(np.mean(xgb_z))
print(np.mean(svc_z))
print(np.mean(svr_z))

