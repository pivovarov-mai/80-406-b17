import numpy as np


def phi(x, t, a):
    if x == 0.0 or x == 1.0:
        return x
    else:
        raise ValueError('incorrect border value')


def psi(x, t=0.0, a=0.0):
    return x + np.sin(np.pi * x)


def analitic_solution(x, t, a):
    return x + np.exp(-np.pi * np.pi * a * t) * np.sin(np.pi * x)


def tridiagonal_matrix_algorithm(a, b, c, d):
    n = len(a)

    p = np.zeros(n)
    q = np.zeros(n)

    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]

    for i in range(1, n):
        p[i] = -c[i] / (b[i] + a[i] * p[i - 1])
        q[i] = (d[i] - a[i] * q[i - 1]) / (b[i] + a[i] * p[i - 1])

    x = np.zeros(n)
    x[-1] = q[-1]

    for i in range(n - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]

    return x
