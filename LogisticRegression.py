#x,y -> list, all functions -> np.array

import LinearRegression as lr
import random
import numpy as np
import PythonApplication10 as pa
import math

def logisticregression_sgd(x: list, y: list, rate:float, init: np.array, threshold: float, cnt=[]) -> np.array :
    """returns g by logistic regression with stochastic gradiet descent
    starts from init, count return how many times it iterated to get below threshold"""
    g=init
    l=len(y)
    epoch=0
    while True:
        rnd=random.randrange(0,l)
        lst=list(range(l))
        random.shuffle(lst)
        gbefore = g
        for n in range(l):
            g = g - rate*cre_gradient([x[lst[n]]],[y[lst[n]]],g)
        epoch+=1
        if np.linalg.norm(gbefore-g) < threshold:
            break
    cnt.append(epoch)
    return g
    

def crossentropy(x: list, y: list, w: np.array) -> float :
    """returns cross-entropy error
    1/N(sigma(ln(1+e^(-ywx))))"""
    sum=0
    l=len(y)
    for n in range(l) :
        sum+=math.log(1+math.exp(-(y[n]*(x[n]@w))))
    return sum/l

def cre_gradient(x: list, y: list, w: np.array) -> np.array :
    """returns gradient of crossentropy"""
    X=np.array(x)
    l=len(y)
    sum=np.zeros(len(x[0]))
    for n in range(l) :
        sum += (y[n]*X[n])/(1+math.exp(y[n]*(X[n]@w)))
    return -(1/l)*sum

def theta(s: float) -> float :
    return math.exp(s)/(1+math.exp(s))

def question8():
    Tfunc=pa.target_function()
    x=lr.x_generation(100,2)    #Training points
    y=lr.find_y(x,Tfunc)
    n=[]
    g=logisticregression_sgd(x,y,0.01,np.array([0,0,0]),0.01,n)
    x_out=lr.x_generation(1000,2)  #out-of-sample samples to test Eout
    y_out=lr.find_y(x_out,Tfunc)
    Eout=lr.evaluate(x_out,y_out,g)
    print(Tfunc,(1/g[0])*g,n,Eout)
    return Eout

e=0
for n in range(100):
    e += question8()
print(e/100)