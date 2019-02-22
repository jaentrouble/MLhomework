# flip 1000 coins 10 times each
# run 100,000 times
# tail = 0, head = 1
# c1 = first coin, crand = random coin from 1000 coins, cmin = coin which had minimum frequency of head (1)
# v1 = 'fraction of heads' from c1 -> same to vrand, vmin

import random
import numpy as np
import openpyxl as op

#wb = op.Workbook()
#ws = wb.active

def flip_coins(num:int, count:int) -> list :
    """ flips 'num' of coins 'count' times and returns a list of [v1, vrand, vmin] """
    lst = []
    for k in range(num) :
        tmp = 0
        for j in range(count) :
            tmp += random.randint(0,1)
        lst.append(tmp)
    return [lst[0]/count,lst[random.randrange(0,num)]/count,min(lst)/count]
    
def write_xl (n:int, val:list, ws: 'worksheet') :
    """ Writes values of 'list' in 'n+2' row at 'ws' """
    for c in range(len(val)) :
        ws.cell(row=n+2, column=c+1).value = val[c]
