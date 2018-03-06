
import matplotlib.pyplot as plt
import numpy as np

file_name = 'moon_dataset_pairs3000_r10_w6_d1.npy'

ds = np.load(file_name)

A = ds[ds[:,2] == 0]
B = ds[ds[:,2] == 1]


x0 = A[:,0]
y0 = A[:,1]
x1 = B[:,0]
y1 = B[:,1]
plt.scatter(x0, y0, marker='x')
plt.scatter(x1, y1, marker='x')
plt.show()
