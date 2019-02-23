import LinearRegression as lr
import PythonApplication10 as pa
import numpy as np
import cvxopt as co
import copy
co.solvers.options['abstol'] = 1e-20
co.solvers.options['reltol'] = 1e-20
co.solvers.options['feastol'] = 1e-20

def chkvalid(y: list) -> bool :
    """returns false if all y's 1 or -1"""
    return len(y)!=abs(sum(y))

def svm(sx: list, sy: list) -> list :
    """returns [SVM g, # of support vectors]"""
    x=copy.deepcopy(sx)
    y=copy.deepcopy(sy)
    for n in range(len(x)):
        x[n].pop(0)
    X = np.array(x)             #x0s are popped out!
    Y = np.array(y)             #1-D
    N = len(x)
    P = np.zeros((N,N))
    for k in range(N):
        for l in range(N):
            P[k][l] = round(Y[k]*Y[l]*(X[k]@X[l]),9)
    print(np.linalg.matrix_rank(P))
    P = co.matrix(P)
    q = co.matrix((-1)*np.ones(N))
    A = co.matrix(np.reshape(Y,(1,len(Y))))
    A = co.matrix(A, (1,len(Y)), 'd')
    b = co.matrix(np.array([[0.0]]))
    sol = co.solvers.qp(P,q,A=A,b=b)
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
    N=4
    sbp = 0  #svm beats pla
    for n in range(100):
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
        print(Esvm,Epla)
        if Esvm > Epla:
            sbp +=1
    return sbp/100

print(question8())