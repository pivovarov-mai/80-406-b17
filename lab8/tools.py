import numpy as np


@np.vectorize
def u0yt(y, t, a, mu1, mu2):
    return np.cos(mu2 * y) * np.exp(-(mu1 * mu1 + mu2 * mu2) * a * t)


@np.vectorize
def u1yt(y, t, a, mu1, mu2):
    return 0.0


@np.vectorize
def u0xt(x, t, a, mu1, mu2):
    return np.cos(mu1 * x) * np.exp(-(mu1 * mu1 + mu2 * mu2) * a * t)


@np.vectorize
def u1xt(x, t, a, mu1, mu2):
    return 0.0


@np.vectorize
def u0xy(x, y, a, mu1, mu2):
    return np.cos(mu1 * x) * np.cos(mu2 * y)


def analitic_solution(a, x, y, t, mu1, mu2):
    p1 = np.cos(mu1 * x)
    p2 = np.cos(mu2 * y)
    p3 = np.exp(-(mu1 * mu1 + mu2 * mu2) * a * t)
    return p1 * p2 * p3


def analitic_grid(a, x, y, t, f):
    grid = np.zeros((len(t), len(y), len(x)))

    for i in range(len(t)):
        for j in range(len(y)):
            for k in range(len(x)):
                grid[i, j, k] = f(a, x[k], y[j], t[i])

    return grid


def error(approx, analytic):
    return np.max(np.abs(approx - analytic))
