import numpy as np
from copy import deepcopy

from tools import ux0y, u1y, uyx0, ux1, analitic_grid, error
from plot import plot


def relaxation(params, x, y):
    h = params['h']
    eps = params['eps']
    epochs = params['epochs']
    diverge = params['diverge']
    w = params['w']

    u = np.zeros((len(y), len(x)))
    prev_u = np.zeros((len(y), len(x)))

    u[:, -1] = u1y(y)
    u[-1, :] = ux1(x)

    for i in range(len(x) - 2, -1, -1):
        for j in range(len(y) - 2, -1, -1):
            u[j, i] = u[j + 1, i] * x[i] / (x[i] + y[j] + eps)
            u[j, i] += u[j, i + 1] * y[j] / (x[i] + y[j] + eps)

    diff_u = np.max(np.abs(u - prev_u))

    for _ in range(epochs):
        if diff_u <= eps:
            break

        prev_diff_u = diff_u
        prev_u = deepcopy(u)

        val1 = -2.0 * h * uyx0(x[:-1])
        val2 = 4.0 * u[1, :-1]
        val3 = u[2, :-1]

        u[0, :-1] += w * ((val1 + val2 - val3) / 3.0 - u[0, :-1])

        val1 = -2.0 * h * ux0y(y[:-1])
        val2 = 4.0 * u[:-1, 1]
        val3 = u[:-1, 2]

        u[:-1, 0] += w * ((val1 + val2 - val3) / 3.0 - u[:-1, 0])

        for i in range(1, len(y) - 1):
            for j in range(1, len(x) - 1):
                sumv = u[i - 1, j] + prev_u[i + 1, j]
                sumv += u[i, j - 1] + prev_u[i, j + 1]

                u[i, j] += w * (sumv / 4.0 - prev_u[i, j])

        diff_u = np.max(np.abs(u - prev_u))

        if diff_u > diverge * prev_diff_u:
            break

    return u


def main(params):
    h = params['h']
    eps = params['eps']
    epochs = params['epochs']
    diverge = params['diverge']
    w = params['w']

    xlhs, xrhs = params['xlhs'], params['xrhs']
    ylhs, yrhs = params['ylhs'], params['yrhs']

    x = np.arange(xlhs, xrhs + h / 2.0, step=h)
    y = np.arange(ylhs, yrhs + h / 2.0, step=h)

    analytic = analitic_grid(x, y)
    approx = relaxation(params, x, y)

    err = error(approx, analytic)
    print(f'error={err:.3f}')

    filename = f'images/relaxation_h={h}_xlhs={xlhs}_xrhs={xrhs}'
    filename += f'_ylhs={ylhs}_yrhs={yrhs}_eps={eps}_epochs={epochs}'
    filename += f'_diverge={diverge}_w={w}.jpg'

    plot(approx, analytic, x, y, filename)


if __name__ == '__main__':
    params = {
        "h": 0.1,
        "xlhs": 0.0,
        "xrhs": 1.0,
        "ylhs": 0.0,
        "yrhs": 1.0,
        "eps": 0.0001,
        "epochs": 100,
        "diverge": 1.5,
        "w": 1.5
    }

    main(params)
