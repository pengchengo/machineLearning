import numpy as np

a = np.array([[1, 2],
             [3,4]])
b = np.array([[5, 6],
              [7, 8]])

c = a@b
d = a*b
print(c)
print(d)