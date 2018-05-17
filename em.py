import random
import matplotlib.pyplot as plt
import numpy as np
import time
from math import pi
from scipy.stats import multivariate_normal
import sklearn.datasets as sk

random.seed(4000)
mean1 = [0, 0]
cov1 = [[1, 0], [0, 10]]  # diagonal covariance
mean2 = (20, 20)
cov2 = [[10, 0], [0, 1]]
mean3 = (20, 0)
cov3 = [[10, 0], [0, 1]]

totalPoints = 1500
"""for calculation"""
x = []
"""for show"""
a = []
b = []
for i in range(0, totalPoints):
    r = random.random()
    bb = 1 / 3
    c = 2 / 3
    if 0 <= r < bb:
        x1, y1 = np.random.multivariate_normal(mean1, cov1, 1).T
        a.append(x1[0])
        b.append(y1[0])
        x.append([x1[0], y1[0]])
    if bb <= r < c:
        x1, y1 = np.random.multivariate_normal(mean2, cov2, 1).T
        a.append(x1[0])
        b.append(y1[0])
        x.append([x1[0], y1[0]])
    if c <= r <= 1:
        x1, y1 = np.random.multivariate_normal(mean3, cov3, 1).T
        a.append(x1[0])
        b.append(y1[0])
        x.append([x1[0], y1[0]])

miu = np.random.rand(2, 3) * 25
miu2 = np.transpose(miu)
k = 1 / 3
w = np.array([k, k, k])
sigma = []
pij = np.empty(shape=[3, totalPoints])
# print(pij.shape)
for i in range(0, 3):
    sigma.append(sk.make_spd_matrix(2)*40)
# print(sigma)


def drawEllipse(u, v, a, b):
    # u = x of center, v = y of center, a = radius on x, b = radius on y.
    t = np.linspace(0, 2 * pi, 100)
    plt.plot(u + a * np.cos(t), v + b * np.sin(t))


def show():
    plt.plot(a, b, 'x')
    plt.plot(miu[0], miu[1], 'o')
    for i in range(0, 3):
        drawEllipse(miu[0][i], miu[1][i], sigma[i][0][0]**(.5), sigma[i][1][1]**(.5))
    plt.axis('equal')
    plt.show()
    # plt.clf()
    time.sleep(1)


def ll():
    sum1 = 0
    for j in range(0, totalPoints):
        sum2 = 0
        for i in range(0, 3):
            sum2 += w[i] * multivariate_normal.pdf([x[j][0], x[j][1]], miu2[i], sigma[i])
        sum1 += np.log(sum2)
        return sum1


def E():
    for j in range(0, totalPoints):
        down = 0
        for i in range(0, 3):
            down += w[i] * multivariate_normal.pdf([x[j][0], x[j][1]], miu2[i], sigma[i])
        for i in range(0, 3):
            pij[i][j] = (w[i] * multivariate_normal.pdf([x[j][0], x[j][1]], miu2[i], sigma[i])) / down


def M():
    # estimating miu
    for i in range(0, 3):
        a = np.array([0, 0])
        b = 0
        for j in range(0, totalPoints):
            a[0] += x[j][0] * pij[i][j]
            a[1] += x[j][1] * pij[i][j]
            b += pij[i][j]
        for k in range(0, 2):
            miu2[i][k] = a[k] / b
        w[i] = b / totalPoints
        # print("printing miu")
        # print(miu)

    for i in range(0, 3):
        e = 0
        d = None
        for j in range(0, totalPoints):
            a = np.subtract(np.array([x[j][0], x[j][1]]), miu2[i])
            b = np.transpose(a)
            c = np.outer(a, b)
            if j == 0:
                d = np.multiply(c, pij[i][j])
            else:
                d = np.add(d, np.multiply(c, pij[i][j]))
            e += pij[i][j]
        # print(e)
        sigma[i] = np.divide(d, e)

        # print("Printing Sigma")
        # print(sigma)


def main():
    # for i in range(0, 100):
    #     E()
    #     M()
    #     show()

    l1 = ll()
    l2 = 0
    # m = miu2.copy()
    i = 0
    while True:
        i += 1
        E()
        M()
        show()
        l2 = ll()
        if abs(l1 - l2) < .000001:
            break
        l1 = l2
        # m = miu2.copy()
        E()
        M()
        show()


    print(i)


if __name__ == "__main__":
    main()
