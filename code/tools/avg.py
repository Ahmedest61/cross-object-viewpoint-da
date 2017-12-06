import numpy as np
a = np.loadtxt('adda.txt')
print a
a = a.reshape(4, 2 * 6)
print a
for i in range(4):
    print a[i, 0::2].mean(), a[i, 1::2].mean()
