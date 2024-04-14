import numpy as np

a = np.array([[1, 2, 3],
             [3,4,6],
              [7,8,9]])
b = np.array([[1, 2, 3],
             [3,4,6],
              [7,8,9]])

c = a@b
d = a*b
e = a[1:2,1:2]
f = b[1:2,1:2]
print(c)
print(d)
print(e)
print(e*f)