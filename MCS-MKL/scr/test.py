import numpy as np

a = np.array([3,5,4])
indices = np.argsort(a)[::-1]
sorted = np.sort(a)[::-1]
print(indices)
print(sorted)