import numpy as np


def phi(x, t, a):
    return -psi0(x, t, a)


def psi0(x, t, a):
    return np.sin(x) + np.cos(x)


def psi1(x, t, a):
    return -a * (np.sin(x) + np.cos(x))


def analitic_solution(x, t, a):
    return np.sin(x - a * t) + np.cos(x + a * t)


def analitic_grid(x, t, a):
    grid = np.zeros((len(t), len(x)))

    for i in range(len(t)):
        for j in range(len(x)):
            grid[i, j] = analitic_solution(x[j], t[i], a)
    return grid


def second_derivative(lst, step):
    der = np.zeros(lst.shape)
    for i in range(1, len(lst) - 1):
        der[i] = (lst[i - 1] - 2.0 * lst[i] + lst[i + 1]) / (step * step)
    return der


def tridiagonal_matrix_algorithm(matrix, target):
    n = matrix.shape[1]

    p = np.zeros(n)
    q = np.zeros(n)

    p[0] = -matrix[0, 1] / matrix[0, 0]
    q[0] = target[0] / matrix[0, 0]

    for i in range(1, n - 1):
        den = matrix[i, i] + matrix[i, i - 1] * p[i - 1]
        p[i] = -matrix[i, i + 1] / den

        den = matrix[i, i] + matrix[i, i - 1] * p[i - 1]
        q[i] = (target[i] - matrix[i, i - 1] * q[i - 1]) / den

    p[-1] = 0

    res = np.zeros(n + 1)

    for i in range(n - 1, -1, -1):
        res[i] = p[i] * res[i + 1] + q[i]

    return res[:-1]


def error(approx, analytic):
    errors = []

    for v1, v2 in zip(approx, analytic):
        errors.append(np.mean(np.abs(v1 - v2)))

    return np.array(errors)
