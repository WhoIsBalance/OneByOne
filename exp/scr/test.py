import numpy as np

a = np.array([[1,2,3],[5,2,3]])
idx = np.argsort(a, axis=1)[::,::-1]
b = a[idx]
print(b)