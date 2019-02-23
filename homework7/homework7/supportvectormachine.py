import LinearRegression as lr
import PythonApplication10 as pa
import numpy as np
import cvxopt as co


def chkvalid(y: list) -> bool :
    """returns false if all y's 1 or -1"""
    return len(y)!=abs(sum(y))

def svm(x: list, y: list) -> list :
    """returns [SVM g, # of support vectors]"""
    for n in range(len(x)):
        x[n].pop(0)
    X = np.array(x)             #x0s are popped out!
    Y = np.array(y)             #1-D
    N = len(x)
    P = np.zeros((N,N))
    for k in range(N):
        for l in range(N):
            P[k][l] = Y[k]*Y[l]*(X[k]@X[l])
    P = co.matrix(P)
    q = co.matrix((-1)*np.ones(N))
    A = co.matrix(Y)
    A = co.matrix(A, (1,len(Y)), 'd')
    b = co.matrix(np.array([[0.0]]))
    sol = co.solvers.qp(P,q,A=A,b=b) # error
    alpha = np.array(sol['x'])       #2-D
    w = np.zeros(len(X[0]))
    for m in range(len(X)):
        w += alpha[m][0]*Y[m]*X[m]
    thres = 0
    sup = 0
    for n in range(N):
        if alpha[n] > 0:
            thres = (1/Y[n])-w@X[n]
            sup += 1
    return [np.insert(w,0,thres),sup]

def question8():
    N=10
    sbp = 0  #svm beats pla
    for n in range(1000):
        while True:
            f = pa.target_function()
            X = lr.x_generation(N,2)
            Y = lr.find_y(X,f)
            if chkvalid(Y) :
                break
        tmp = svm(X,Y)
        gs = tmp[0]
        gp = pa.pla_X(X,Y,100)
        TstX = lr.x_generation(1000,2)
        TstY = lr.find_y(TstX,f)
        Esvm = lr.evaluate(TstX,TstY,gs)
        Epla = lr.evaluate(TstX,TstY,gp)
        if Esvm > Epla:
            sbp +=1
    return sbp/1000

question8()