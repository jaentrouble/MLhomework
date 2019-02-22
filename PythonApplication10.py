# Perceptron Learning Algorithm
# 범위 [-1,1] x [-1,1]
# target function f 임의로 선정 (점 두개 랜덤으로 픽)


import random
import numpy as np
import openpyxl as op

#excel_doc = op.Workbook()
#sheet = excel_doc.active

def numgen() :
    """-1~1 사이 랜덤 실수 제공 """
    return random.random()*random.choice((-1,1))

def target_function() -> np.array :
    """ target function f 를 생성, w1x1 + w2x2 + 1 = 0에서 array([1,w1,w2]) 반환"""
   
    while True :
        x11 = numgen()
        x12 = numgen()
        x21 = numgen()
        x22 = numgen()
        if x11 != x12 and x21 != x22:
            break
    
    a = np.array([[x11,x12],[x21,x22]])
    b = np.array([[-1],[-1]])

    return np.insert(np.linalg.solve(a,b),0,1)

def sign(a) -> int :
    """ 부호가 양수면 1, 음수면 -1 반환"""
    if a>= 0:
        return 1
    else:
        return -1

def pla(N:int, Tfunc: np.array) -> np.array :
    """ returns learned 'g : w0+w1x1+w2x2 = 0' in array([w0,w1,w2]) form and saves in 'worksheet' """
    s = np.size(Tfunc)
    g = np.zeros(1*s)   # g=array([0,0,....,0])
    num=0
    while num<int(N):
        t = np.array([])
        for k in range(s):           #generate a new 'test' point
            t = np.append(t,numgen())
        if sign(t@g) != sign(t@Tfunc):
            g = g + sign(t@Tfunc)*t
            num += 1                                   #count only when updated by misclassified points
    return g

