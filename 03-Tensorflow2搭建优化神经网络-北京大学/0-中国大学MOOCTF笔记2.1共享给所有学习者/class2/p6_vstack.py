import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.vstack((a, b))   # vertical 方向 将两个矩阵 堆叠在一起
print("c:\n", c)

