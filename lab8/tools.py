import numpy as np


def ux0y(y):
    return np.zeros(len(y))


def u1y(y):
    return 1.0 - y * y


def uyx0(x):
    return np.zeros(len(x))


def ux1(x):
    return x * x - 1.0


def analitic_solution(x, y):
    return x * x - y * y


def analitic_grid(x, y):
    grid = np.zeros((len(y), len(x)))

    for i in range(len(y)):
        for j in range(len(x)):
            grid[i, j] = analitic_solution(x[j], y[i])
    return grid


def error(approx, analytic):
    return np.max(np.abs(approx - analytic))
