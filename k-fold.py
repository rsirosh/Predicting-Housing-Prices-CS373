import numpy as np
import numpy.linalg as la
import linreg as lr

def run(k, X, y):

    (n,d) = X.shape
    z = np.zeros((k, 1))
    for i in range(k):
        S = []
        T = []
        for j in range(n):
            if j >= np.floor((n * i) / k) and j <= np.floor(((n * (i + 1)) / k) - 1):
                T.append(j)
            else
                S.append(j)

        subset = np.array(S)
        theta = linreg(X[subset,:], y[subset])
        val = 0

        for t in T:
            val += (y[t] - np.dot(theta.transpose(), X[t])) ** 2

        if(len(T) > 0):
            val = val / len(T)

        z[i] = val

    return z

def linreg(X,y):
    return np.dot(la.pinv(X), y)