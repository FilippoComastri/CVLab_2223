import numpy as np
import math

b = np.array([[10],[11],[12]])
a = np.array([[1,2,3],[4,5,6]])

def matrix_mul(v1,v2):
    c1 = v1.shape[1]
    r1 = v1.shape[0]
    c2 = v2.shape[1]
    r2 = v2.shape[0]
    print(r1,c1,r2,c2)
    result = np.zeros((r1,c2),dtype=int)
    if(c1==r2):
        for i in range(r1): # riga 0
            for j in range(c2): # colonna
               for k in range(r2):
                    result[i,j]+=v1[i,k]*v2[k,j]
    return result

print("Dot with loop",matrix_mul(a,b))
print("Dot with NumPy",np.dot(a,b))