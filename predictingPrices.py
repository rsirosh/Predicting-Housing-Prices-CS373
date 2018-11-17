# Requirements:
# pip3 install -U scikit-learn
# pip3 install xgboost
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
import timeit
import math


def k_fold(k, model, X, y):
    n, d = X.shape
    z = np.zeros((k, 1))
    for i in range(k):
        T = list(range(int((i * n) / k), int((n * (i + 1) / k))))
        S = [j for j in range(n) if j not in T]
        model.fit(X[S], y[S])
        z[i] = (1. / len(T)) * np.sum((y[T] - model.predict(X[T])) ** 2)
    return z

def modelPredictions(X_train, X_test, y_train, y_test, training, target):

    adaBoost = AdaBoostRegressor()
    adaBoost.fit(X_train, y_train.values.ravel())
    evaluateModel(adaBoost, X_test, y_test, "AdaBoost")

    xgb = XGBRegressor()
    xgb.fit(X_train, y_train)
    evaluateModel(xgb, X_test, y_test, "XGBoost")

    svc_model = svm.SVC(kernel="rbf", C=1.0)
    svc_model.fit(X_train, y_train.values.ravel())
    evaluateModel(svc_model, X_test, y_test, "SVC")

    svr_model = svm.SVR(kernel="poly", shrinking=False, coef0=-2000)
    svr_model.fit(X_train, y_train.values.ravel())
    evaluateModel(svr_model, X_test, y_test, "SVR")

    evaluateWithKFold(adaBoost, training, target, "AdaBoost")
    evaluateWithKFold(xgb, training, target, "XGBoost")
    evaluateWithKFold(svc_model, training, target, "SVC")
    evaluateWithKFold(svr_model, training, target, "SVR")



def evaluateModel(model, X_test, y_test, name, splits=5):
    start_time = timeit.default_timer()

    mse = cross_val_score(model, X_test, y_test.values.ravel(), cv=splits, scoring='neg_mean_squared_error')
    print(name + ' Mean Squared Error: ', np.mean(mse * -1))

    rmse = math.sqrt(np.mean(mse*-1))
    print(name + ' Root Mean Squared Error: ', rmse)

    elapsed = timeit.default_timer() - start_time

def evaluateWithKFold(model, training, target, name):
    print("K-fold CV: " + name)
    model_z = k_fold(5, model, training.values, target.values.ravel())
    print("Mean Squared Error: " + str(np.mean(model_z)))
    print("Root Mean Squared Error: " + str(math.sqrt(np.mean(model_z))))


def dataPreprocessing(data):
    # Replaces categorical value in Quality columns with numerical scale
    qualityCols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                  'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']

    for col in qualityCols:
        data[col] = data[col].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po':1, 'NA': 0})

    # Gets all Categorical columns
    catCols = set(list(data))-set(list(data._get_numeric_data()))
    #Fill Categorical Column Null values with 0
    for col in catCols:
        data[col].fillna(0, inplace=True)

    #Fill numerical column null values with mean of column
    data = data.fillna(data.mean())
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

    # Split into training and target sets
    num_variables = len(data.columns)
    training = data.iloc[:, 0:num_variables-1]
    target = data.iloc[:,num_variables-1:]

    # 80:20 train test ratio
    test_size = 0.2
    # This function splits the training and target sets into random train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(training, target, test_size=test_size)

    return X_train, X_test, y_train, y_test, training, target


if __name__ == "__main__":
    data = pd.read_csv("train.csv", header=0)

    X_train, X_test, y_train, y_test, training, target = dataPreprocessing(data)

    modelPredictions(X_train, X_test, y_train, y_test, training, target)
