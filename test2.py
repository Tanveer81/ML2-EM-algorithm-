import numpy as np

mean1 = [0, 0]
cov1 = [[1, 0], [0, 10]]  # diagonal covariance
mean2 = (20, 20)
cov2 = [[10, 0], [0, 1]]
mean3 = (20, 0)
cov3 = [[10, 0], [0, 1]]
x1, y1 = np.random.multivariate_normal(mean1, cov1, 1).T
print(x1)
print(y1)