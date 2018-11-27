import numpy as np

a = np.array([1,2])
b = np.array([10,20,30])
z = np.matmul(a[None, :].T, b[None, :])

print(z)
