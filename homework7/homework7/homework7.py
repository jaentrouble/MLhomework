import homework6 as h6
import numpy as np
import LinearRegression as lr

def dimension_constraint(k: int, x: list) -> list:
    """same as non_lin_transform from h6, returns only from 0th ~ kth variable"""
    transx = []
    for n in range(len(x)):
        x1=x[n][0]
        x2=x[n][1]
        tmp=[1,x1,x2,x1**2,x2**2,x1*x2,abs(x1-x2),abs(x1+x2)]
        for m in range(len(tmp)-k-1):
            tmp.pop()
        transx.append(tmp)
    return transx

def model_selection(g: list, zval: list, yval: list) -> int :
    """g is a list of testing functions, returns the best performing function's index in validation set
    zval is list of lists of Zval's for each g"""
    idx=0
    err=0
    for n in range(len(g)):
        tmperr=lr.evaluate(zval[n],yval,g[n])
        if err<tmperr:
            idx=n
            err=tmperr
    return idx
        

def question1() -> list :
    Dx= h6.mprtx('in')
    Dy= h6.mprty('in')
    Xtrain = Dx[0:25]
    Ytrain = Dy[0:25]
    Xvalid = Dx[25:]
    Yvalid = Dy[25:]
    g=[]
    Zvalid=[]
    for n in range(5):
        Ztrain = dimension_constraint(n+3, Xtrain)
        g.append(lr.linear_regression(Ztrain,Ytrain))
        Zvalid.append(dimension_constraint(n+3,Xvalid))
    idx=model_selection(g,Zvalid,Yvalid)
    print("k : ",idx+3)
    print("model : ", g[idx])
    return [g[idx], idx]

def question2():
    Dx= h6.mprtx('in')
    Dy= h6.mprty('in')
    Xtrain = Dx[0:25]
    Ytrain = Dy[0:25]
    Xvalid = Dx[25:]
    Yvalid = Dy[25:]
    Xout = h6.mprtx('out')
    Yout = h6.mprty('out')
    g=[]
    Zout=[]
    for n in range(5):
        Ztrain = dimension_constraint(n+3, Xtrain)
        g.append(lr.linear_regression(Ztrain,Ytrain))
        Zout.append(dimension_constraint(n+3,Xout))
    idx=model_selection(g,Zout,Yout)
    print("k : ",idx+3)
    print("model : ", g[idx])

def question3() -> list :
    Dx= h6.mprtx('in')
    Dy= h6.mprty('in')
    Xvalid = Dx[0:25]
    Yvalid = Dy[0:25]
    Xtrain = Dx[25:]
    Ytrain = Dy[25:]
    g=[]
    Zvalid=[]
    for n in range(5):
        Ztrain = dimension_constraint(n+3, Xtrain)
        g.append(lr.linear_regression(Ztrain,Ytrain))
        Zvalid.append(dimension_constraint(n+3,Xvalid))
    idx=model_selection(g,Zvalid,Yvalid)
    print("k : ",idx+3)
    print("model : ", g[idx])
    return [g[idx],idx]

def question4():
    Dx= h6.mprtx('in')
    Dy= h6.mprty('in')
    Xvalid = Dx[0:25]
    Yvalid = Dy[0:25]
    Xtrain = Dx[25:]
    Ytrain = Dy[25:]
    Xout = h6.mprtx('out')
    Yout = h6.mprty('out')
    g=[]
    Zout=[]
    for n in range(5):
        Ztrain = dimension_constraint(n+3, Xtrain)
        g.append(lr.linear_regression(Ztrain,Ytrain))
        Zout.append(dimension_constraint(n+3,Xout))
    idx=model_selection(g,Zout,Yout)
    print("k : ",idx+3)
    print("model : ", g[idx])

def question5() :
    q1 = question1()
    q2 = question3()
    Xout = h6.mprtx('out')
    Yout = h6.mprty('out')
    e1 = lr.evaluate(dimension_constraint(q1[1]+3,Xout),Yout,q1[0])
    e2 = lr.evaluate(dimension_constraint(q2[1]+3,Xout),Yout,q2[0])
    print(1-e1,1-e2)

