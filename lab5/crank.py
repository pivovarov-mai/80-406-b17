import numpy as np
from tools import psi, phi, tridiagonal_matrix_algorithm
from plot import plot_stat


def crank(a, n, cnt, step, x_min, x_max, approx):
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
    eta = 0.5

    u = np.zeros((cnt, n))
    h = (x_max - x_min) / n
    sigma = (a * a * step) / (h * h)

    for i in range(1, n - 1):
        u[0][i] = psi(x_min + i * h)

    for i in range(1, cnt):
        alst, blst, clst, dlst = np.zeros(
            n), np.zeros(n), np.zeros(n), np.zeros(n)

        for j in range(1, n - 1):
            alst[j] = sigma
            blst[j] = -(1 + 2 * sigma)
            clst[j] = sigma
            dlst[j] = -u[i - 1][j]

        if approx == 21:
            blst[0] = -1 / h
            clst[0] = 1 / h
            dlst[0] = phi(0.0, (i + 1) * step, a)
            alst[-1] = -1 / h
            alst[-1] = 1 / h
            dlst[-1] = phi(1.0, (i + 1) * step, a)
        elif approx == 32:
            k0 = 1 / (2 * h) / clst[1]
            blst[0] = (-3 / (2 * h) + alst[1] * k0)
            clst[0] = 2 / h + blst[1] * k0
            dlst[0] = phi(0.0, (i + 1) * step, a) + dlst[1] * k0
            k1 = -(1 / (h * 2)) / alst[-2]
            alst[-1] = (-2 / h) + blst[-2] * k1
            blst[-1] = (3 / (h * 2)) + clst[-2] * k1
            dlst[-1] = phi(1.0, (i + 1) * step, a) + dlst[-2] * k1
        elif approx == 22:
            blst[0] = 2 * a * a / h + h / step
            clst[0] = -2 * a * a / h
            dsub = phi(0.0, (i + 1) * step, a) * 2 * a * a
            dlst[0] = (h / step) * u[i - 1][0] - dsub
            alst[-1] = -2 * a * a / h
            blst[-1] = 2 * a * a / h + h / step
            dadd = phi(1.0, (i + 1) * step, a) * 2 * a * a
            dlst[-1] = (h / step) * u[i - 1][-1] + dadd
        else:
            str = 'Такого типа апроксимации граничных условий не существует'
            raise ValueError(str)

        u[i] = eta * tridiagonal_matrix_algorithm(alst, blst, clst, dlst)
        explicit_part = np.zeros(n)

        for j in range(1, n - 1):
            val1 = sigma * (u[i - 1][j + 1] + u[i - 1][j - 1])
            val2 = (1 - 2 * sigma) * u[i - 1][j]
            explicit_part[j] = val1 + val2

        if approx == 21:
            explicit_part[0] = explicit_part[1] - h * phi(0.0, i * step, a)
            explicit_part[-1] = explicit_part[-2] + h * phi(1.0, i * step, a)
        elif approx == 32:
            sub1 = 2 * explicit_part[1] / h
            val1 = phi(0.0, i * step, a) + explicit_part[2] / (2 * h) - sub1
            explicit_part[0] = val1 * 2 * h / -3

            add2 = 2 * explicit_part[-2] / h
            val2 = phi(1.0, i * step, a) - explicit_part[-3] / (2 * h) + add2
            explicit_part[-1] = val2 * 2 * h / 3
        elif approx == 22:
            add1 = h * h / (2 * step) * u[i - 1][0]
            val1 = explicit_part[1] - h * phi(0.0, i * step, a) + add1
            explicit_part[0] = val1 / (1 + h * h / (2 * step))

            add2 = h * h / (2 * step) * u[i - 1][-1]
            val2 = explicit_part[-2] + h * phi(1.0, i * step, a) + add2
            explicit_part[-1] = val2 / (1 + h * h / (2 * step))
        else:
            str = 'Такого типа апроксимации граничных условий не существует'
            raise ValueError(str)

        u[i] += (1 - eta) * explicit_part

    return u


def main(params):
    a = params['a']
    n = params['n']
    cnt = params['cnt']
    step = params['step']
    x_min = params['x_min']
    x_max = params['x_max']
    approx = params['approx']

    filename_u = f'images/crank_function_a={a}_n={n}_cnt={cnt}_step={step}'
    filename_u += f'_xmin={x_min}_xmax={x_max}_approx={approx}.jpg'

    filename_e = f'images/crank_error_a={a}_n={n}_cnt={cnt}_step={step}'
    filename_e += f'_xmin={x_min}_xmax={x_max}_approx={approx}.jpg'

    u = crank(a, n, cnt, step, x_min, x_max, approx)
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
