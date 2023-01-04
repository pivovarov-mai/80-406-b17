import numpy as np
from tools import psi, phi
from plot import plot_stat


def explicit(a, n, cnt, step, x_min, x_max, approx):
    """
    Parameters:
        a: коэффициент температуропроводности
        n: количество точек в пространстве
        cnt: количество временных точек
        step: временной шаг
        x_min: левая граница
        x_max: правая граница
        approx: тип апроксимации
            approx = 21: двухточечная аппроксимация с первым порядком
            approx = 32: трехточечная аппроксимация со вторым порядком
            approx = 22: двухточечная аппроксимация со вторым порядком

    Return:
        u: сеточная функция
    """
    h = (x_max - x_min) / n
    sigma = (a * a * step) / (h * h)

    if sigma > 0.5:
        raise ValueError(f'Явная схема не устойчива sigma = {sigma}')

    u = np.zeros((cnt, n))

    for i in range(1, n - 1):
        u[0][i] = psi(x_min + i * h)

    for i in range(1, cnt):
        for j in range(1, n - 1):
            add = u[i - 1][j + 1] + u[i - 1][j - 1]
            sub = 1 - 2 * sigma
            u[i][j] = sigma * add + sub * u[i - 1][j]

        if approx == 21:
            u[i][0] = u[i][1] - h * phi(0.0, i * step, a)
            u[i][-1] = u[i][-2] + h * phi(1.0, i * step, a)
        elif approx == 32:
            phi1 = phi(0.0, i * step, a)
            phi2 = phi(1.0, i * step, a)

            val1 = phi1 + u[i][2] / (2 * h) - 2 * u[i][1] / h
            val2 = phi2 - u[i][-3] / (2 * h) + 2 * u[i][-2] / h

            u[i][0] = val1 * 2 * h / -3
            u[i][-1] = val2 * 2 * h / 3
        elif approx == 22:
            add1 = h * h / (2 * step) * u[i - 1][0]
            add2 = h * h / (2 * step) * u[i - 1][-1]

            val1 = u[i][1] - h * phi(0.0, i * step, a) + add1
            val2 = u[i][-2] + h * phi(1.0, i * step, a) + add2

            u[i][0] = val1 / (1 + h * h / (2 * step))
            u[i][-1] = val2 / (1 + h * h / (2 * step))
        else:
            str = 'Такого типа апроксимации граничных условий не существует'
            raise ValueError(str)
    return u


def main(params):
    a = params['a']
    n = params['n']
    cnt = params['cnt']
    step = params['step']
    x_min = params['x_min']
    x_max = params['x_max']
    approx = params['approx']

    filename_u = f'images/explicit_function_a={a}_n={n}_cnt={cnt}_step={step}'
    filename_u += f'_xmin={x_min}_xmax={x_max}_approx={approx}.jpg'

    filename_e = f'images/explicit_error_a={a}_n={n}_cnt={cnt}_step={step}'
    filename_e += f'_xmin={x_min}_xmax={x_max}_approx={approx}.jpg'

    u = explicit(a, n, cnt, step, x_min, x_max, approx)
    plot_stat(filename_u, filename_e, a, n, cnt, step, x_min, x_max, u)


if __name__ == '__main__':
    run1 = {'a': 0.001,
            'n': 100,
            'cnt': 2000,
            'step': 0.001,
            'x_min': 0.0,
            'x_max': 1.0,
            'approx': 21}

    run2 = {'a': 0.001,
            'n': 100,
            'cnt': 2000,
            'step': 0.001,
            'x_min': 0.0,
            'x_max': 1.0,
            'approx': 32}

    run3 = {'a': 0.001,
            'n': 100,
            'cnt': 2000,
            'step': 0.001,
            'x_min': 0.0,
            'x_max': 1.0,
            'approx': 22}

    main(run1)
    main(run2)
    main(run3)
