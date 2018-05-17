import matplotlib.pyplot as plt
import numpy as np
import time
from math import pi
from scipy.stats import multivariate_normal
import sklearn.datasets as sk

mean1 = [0, 0]
cov1 = [[1, 0], [0, 10]]  # diagonal covariance
mean2 = (20, 20)
cov2 = [[10, 0], [0, 1]]
mean3 = (20, 0)
cov3 = [[10, 0], [0, 1]]
x1, y1 = np.random.multivariate_normal(mean1, cov1, 500).T
x2, y2 = np.random.multivariate_normal(mean2, cov2, 500).T
x3, y3 = np.random.multivariate_normal(mean3, cov3, 500).T
x = []
y = []

miu = np.random.rand(2, 3) * 25
miu2 = np.transpose(miu)
k = 1/3
w = np.array([k, k, k])
sigma = []
totalPoints = 1500
pij = np.empty(shape=[3, totalPoints])


def drawEllipse(u, v, a, b):
    # u = x of center, v = y of center, a = radius on x, b = radius on y.
    t = np.linspace(0, 2 * pi, 100)
    plt.plot(u + a * np.cos(t), v + b * np.sin(t))


def init():
    for j in range(0, 500):
        x.append(x1[j])
        x.append(x2[j])
        x.append(x3[j])
        y.append(y1[j])
        y.append(y2[j])
        y.append(y3[j])

    # for i in range(0, 3):
    #     for j in range(0, 2):
    #         for k in range(0, 2):
    #             if j != k:
    #                 sigma[i][j][k] = 0
    for i in range(0,3):
        sigma.append(sk.make_spd_matrix(2))
    print(sigma)


def show():
    plt.plot(x, y, 'x')
    plt.plot(miu[0], miu[1], 'o')
    for i in range(0, 3):
        drawEllipse(miu[0][i], miu[1][i], sigma[i][0][0], sigma[i][1][1])
    plt.axis('equal')
    plt.show()
    # plt.clf()
    time.sleep(.5)


def E():
    for j in range(0, totalPoints):
        down = 0
        for i in range(0, 3):
            down += w[i] * multivariate_normal.pdf([x[j], y[j]], miu2[i], sigma[i])
        for i in range(0, 3):
            pij[i][j] = (w[i] * multivariate_normal.pdf([x[j], y[j]], miu2[i], sigma[i])) / down


def M():
    # estimating miu
    for i in range(0, 3):
        a = np.array([0, 0])
        b = 0
        for j in range(0, totalPoints):
            a[0] += x[j] * pij[i][j]
            a[1] += y[j] * pij[i][j]
            b += pij[i][j]
        for k in range(0, 2):
            miu2[i][k] = a[k] / b
        w[i] = b / totalPoints
        print("printing miu")
        print(miu)

    for i in range(0, 3):
        e = 0
        d = None
        for j in range(0, totalPoints):
            a = np.subtract(np.array([x[j], y[j]]), miu2[i])
            b = np.transpose(a)
            c = np.outer(a, b)
            if j == 0:
                d = np.multiply(c, pij[i][j])
            else:
                d = np.add(d, np.multiply(c, pij[i][j]))
            e += pij[i][j]
        print(e)
        sigma[i] = np.divide(d, e)

        print("Printing Sigma")
        print(sigma)


def main():
    init()
    for i in range(0,100):
        E()
        M()
        show()
    print(w)


if __name__ == "__main__":
    main()
