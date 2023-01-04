import numpy as np

from tools import phi, psi0, psi1
from tools import tridiagonal_matrix_algorithm, error, analitic_grid

from plot import plot


def implicit(params):
    psi0 = params['psi0']
    psi1 = params['psi1']
    phi = params['phi']
    a = params['a']
    h = params['h']
    tau = params['tau']
    lhs = params['lhs']
    rhs = params['rhs']
    t_last = params['t_last']
    approx = params['approx']

    sigma = (a * tau * tau) / (h * h)
    x = np.arange(lhs, rhs + h / 2.0, step=h)
    t = np.arange(0, t_last + tau / 2.0, step=tau)
    u = np.zeros((len(t), len(x)))

    u[0] = psi0(x, 0, 0)
    u[1] = u[0] + tau * psi1(x, 0, a) + tau * tau * phi(x, 0, 0) / 2.0

    for i in range(1, len(t) - 1):
        if approx == 21:
            lst0 = ((1.0 + h), -1.0, 0.0)
            lstl = (-1.0, (1.0 - h), 0.0)
        elif approx == 32:
            lst0 = (-(2.0 + 2.0 * h),
                    -(1.0 / sigma - 2.0),
                    (-2.0 * u[i][1] + u[i - 1][1]) / sigma)

            lstl = (-(1.0 / sigma - 2.0),
                    -(2.0 - 2.0 * h),
                    (-2.0 * u[i][-2] + u[i - 1][-2]) / sigma)
        elif approx == 22:
            lst0 = (-(2.0 + 2.0 * h + 1.0 / sigma),
                    2.0,
                    (-2.0 * u[i][0] + u[i - 1][0]) / sigma)

            lstl = (2.0,
                    -(2.0 - 2.0 * h + 1.0 / sigma),
                    (-2.0 * u[i][-1] + u[i - 1][-1]) / sigma)

        matrix = np.zeros((len(x), len(x)))

        matrix[0] += np.array([lst0[0], lst0[1]] + [0.0] * (len(matrix) - 2))
        target = [lst0[2]]

        for j in range(1, len(matrix) - 1):
            add1 = [0.0] * (j - 1)
            add2 = [1.0, -(2.0 + 1.0 / sigma), 1.0]
            add3 = [0.0] * (len(matrix) - j - 2)

            matrix[j] += np.array(add1 + add2 + add3)
            target += [(-2.0 * u[i][j] + u[i - 1][j]) / sigma]

        add1 = [0.0] * (len(matrix) - 2)
        add2 = [lstl[0], lstl[1]]

        matrix[-1] += np.array(add1 + add2)

        target += [lstl[2]]
        target = np.array(target)

        u[i + 1] = tridiagonal_matrix_algorithm(matrix, target)

    return u


def main(params):
    a = params['a']
    h = params['h']
    tau = params['tau']
    lhs = params['lhs']
    rhs = params['rhs']
    t_last = params['t_last']
    approx_type = params['approx']

    x = np.arange(0, rhs + h / 2.0, step=h)
    t = np.arange(0, t_last + tau / 2.0, step=tau)

    analytic = analitic_grid(x, t, a)
    approx = implicit(params)
    err = error(approx, analytic)

    filename_u = f'images/implicit_function_a={a}_h={h}_tau={tau}_lhs={lhs}'
    filename_u += f'_rhs={rhs}_t_last={t_last}_approx={approx_type}.jpg'

    filename_e = f'images/implicit_error_a={a}_h={h}_tau={tau}_lhs={lhs}'
    filename_e += f'_rhs={rhs}_t_last={t_last}_approx={approx_type}.jpg'

    filenames = (filename_u, filename_e)

    plot(approx, analytic, x, t, err, filenames)


if __name__ == '__main__':
    run1 = {
        "psi0": psi0,
        "psi1": psi1,
        "phi": phi,
        "a": 0.9,
        "h": 0.1,
        "tau": 0.01,
        "lhs": 0.0,
        "rhs": 3.14,
        "t_last": 3.14,
        "approx": 21,
    }

    run2 = {
        "psi0": psi0,
        "psi1": psi1,
        "phi": phi,
        "a": 0.9,
        "h": 0.1,
        "tau": 0.01,
        "lhs": 0.0,
        "rhs": 3.14,
        "t_last": 3.14,
        "approx": 32,
    }

    run3 = {
        "psi0": psi0,
        "psi1": psi1,
        "phi": phi,
        "a": 0.9,
        "h": 0.1,
        "tau": 0.01,
        "lhs": 0.0,
        "rhs": 3.14,
        "t_last": 3.14,
        "approx": 22,
    }

    main(run1)
    main(run2)
    main(run3)
