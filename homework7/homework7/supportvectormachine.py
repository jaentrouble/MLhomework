import LinearRegression as lr
import PythonApplication10 as pa
import numpy as np
import cvxopt as co


def chkvalid(y: list) -> bool :
    """returns false if all y's 1 or -1"""
    return len(y)!=abs(sum(y))

def svm(x: list, y: list) -> np.list :
    """returns SVM g"""
    for n in range(len(x)):
        x[n].pop(0)
    X = np.array(x)
    Y = np.array(y)
    N = len(x)
    P = np.zeros((N,N))
    for k in range(N):
        for l in range(N):
            P[k][l] = Y[k]*Y[l]*(X[k]@X[l])
    q = co.matrix((-1)*np.ones(N))