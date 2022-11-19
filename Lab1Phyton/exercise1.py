import numpy as np
va = np.array([92,12,29])
vb = np.array([14,9,91])

def dot_with_loops(a,b):
    result = -1
    if len(a) == len(b):
        result = 0
        for i in range(len(a)):
            result += a[i]*b[i]
    return result



print("Dot product with loops: ", dot_with_loops(va,vb))
print("Dot product with NumPy: ",va.dot(vb))