import numpy as np
import math

def normL1(v):
    result = 0
    for i in range(len(v)):
        result += abs(v[i])
    return result

def normL2(v):
    result = 0
    for i in range(len(v)):
        result += pow(v[i],2)
    return math.sqrt(result)

def normINF(v):
    for el in v:
        el = abs(el)
    return max(v)

a = np.array([22,8,14])
print("L1",normL1(a))
print("L2 {:.2f}".format(normL2(a)))
print(normINF(a))
