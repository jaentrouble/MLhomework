#x,y -> list, all functions -> np.array

import random
import numpy as np
import openpyxl as op
import hoeffding as hf
import PythonApplication10 as pa

def linear_regression(x: list, y: list ) -> np.array :
    """Tests 'N' points with Tfunc and returns a linear regression function, x and y have to be the same length"""
    X=np.array(x)
    Y=np.array(y)
    return np.linalg.inv(np.transpose(X)@X)@np.transpose(X)@Y

def find_y(x: list, Tfunc: np.array) -> list :
    """+1 if x@Tfunc>0, -1 if x@Tfunc<0"""
    y=[]
    x=np.array(x)
    for n in range(len(x)) :
        y.append(pa.sign(x[n]@Tfunc))
    return y

def evaluate(x: list, y: list, g: np.array ) -> float :
    """returns E value, x and y have to be the same length, **returns 'correct' ratio"""
    correct=0
    for n in range(len(y)) :
        if pa.sign(x[n]@g) == y[n]:
            correct +=1
    return correct/len(y)

def x_generation(n: int, d: int) -> list :
    """creates a list of (x0,...,xd) * n rows, x0=1 (artificial)"""
    x=[]
    for a in range(n) :
        t=[1]
        for b in range(d) :
            t.append(round(pa.numgen(),2))
        x.append(t)
    return x

def lr_pla(x: list, y: list, cnt=[] ) -> np.array :
    """gets g by pla, initial function by linear regression, count returns number of iterations if needed """
    g=np.array(linear_regression(x,y))
    X=np.array(x)
    c=0
    while True:
        E=[]
        for n in range(len(y)) :
            if pa.sign(X[n]@g) != y[n] :
                E.append(n)
        c+=1
        if sum(E) == 0 or c > 100000 :
            break
        s = random.sample(E,1)[0]
        g = g + y[s]*X[s]
    cnt.append(c)
    return g

def LinearRegressionExperiment() -> None :
    N = int(input('experiment :'))
    O = int(input('Out :'))
    e = int(input('In : '))
    s = []
    t = []
    for n in range(N) :
        target = pa.target_function()
        x=x_generation(e,2)
        y=find_y(x,target)
        nx = x_generation(O,2)
        ny = find_y(nx,target)
        g = linear_regression(x, y)
        s.append(evaluate(x,y,g))
        t.append(evaluate(nx,ny,g))
    print('Ein average = ',sum(s)/N)
    print('Eout average = ', sum(t)/O)

def Linear_PlaExperiment(N: int) -> int :
    c = []
    Tfunc=pa.target_function()
    x = x_generation(N,2)
    y = find_y(x,Tfunc)
    g = lr_pla(x,y,c)
    return c[0]

if __name__ == "__main__" :
    n = int(input('number of iteration :'))
    N = int(input('N :'))
    cnt = []
    for a in range(n):
        cnt.append(Linear_PlaExperiment(N))
    print(cnt)
    print(sum(cnt)/n)