import math
import numpy as np

def mean_loop(v):
    sum = 0
    for el in v:
        sum += el
    return sum/len(v)

def variance_loop(v):
    sum = 0
    for el in v:
        sum += (el - mean_loop(v))**2
    return sum/len(v)

def standard_deviation(v):
    return math.sqrt(variance_loop(v))
    

a=np.array([22,8,14])

print("Mean loop {:.2f}".format(mean_loop(a)))
print("Mean Numpy {:.2f}".format(np.mean(a)))

print("Variance loop {:.2f}".format(variance_loop(a)))
print("Variance numpy {:.2f}".format(np.var(a)))

print("Standard deviation loop {:.2f}".format(standard_deviation(a)))
print("Standard deviation numpy {:.2f}".format(np.std(a)))