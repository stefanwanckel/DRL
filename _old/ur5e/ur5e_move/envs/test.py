import numpy as np

a = 1000 * np.ones(6)
b = 500 * np.ones(6)

c = np.concatenate((a,b))
print(c)