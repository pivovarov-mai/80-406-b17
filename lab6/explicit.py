import numpy as np

from tools import phi, psi0, psi1, second_derivative, error, analitic_grid
from plot import plot


def explicit(params):
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

    if sigma > 1.0:
        raise ValueError(f'Явная схема не устойчива sigma = {sigma}')

    x = np.arange(lhs, rhs + h / 2.0, step=h)
    t = np.arange(0, t_last + tau / 2.0, step=tau)
    u = np.zeros((len(t), len(x)))

    u[0] = psi0(x, 0, 0)
    u[1] = u[0] + tau * psi1(x, 0, a) + tau * tau * phi(x, 0, 0) / 2.0

    for i in range(1, len(t) - 1):
        val1 = 2.0 * u[i]
        val2 = u[i - 1]
        val3 = tau**2 * a * second_derivative(u[i], step=h)
        u[i + 1] = val1 - val2 + val3

        if approx == 21:
            u[i + 1, 0] = u[i + 1, 1] / (1.0 + h)
            u[i + 1, -1] = u[i + 1, -2] / (1.0 - h)
        elif approx == 32:
            u[i + 1, 0] = (4.0 * u[i + 1][1] - u[i + 1][2]) / (3.0 + 2.0 * h)
            den = (3.0 - 2.0 * h)

            u[i + 1, -1] = (4.0 * u[i + 1][-2] - u[i + 1][-3]) / den
        elif approx == 22:
            add1 = sigma * (2.0 * u[i][1] - (2.0 + 2.0 * h) * u[i][0])
            add2 = 2.0 * u[i][0] - u[i - 1][0]
            u[i + 1, 0] = add1 + add2

            add1 = sigma * (2.0 * u[i][-2] + (2.0 * h - 2.0) * u[i][-1])
            add2 = 2.0 * u[i][-1] - u[i - 1][-1]
            u[i + 1, -1] = add1 + add2

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
    approx = explicit(params)
    err = error(approx, analytic)

    filename_u = f'images/explicit_function_a={a}_h={h}_tau={tau}_lhs={lhs}'
    filename_u += f'_rhs={rhs}_t_last={t_last}_approx={approx_type}.jpg'

    filename_e = f'images/explicit_error_a={a}_h={h}_tau={tau}_lhs={lhs}'
    filename_e += f'_rhs={rhs}_t_last={t_last}_approx={approx_type}.jpg'

    filenames = (filename_u, filename_e)

    plot(approx, analytic, x, t, err, filenames)


if __name__ == '__main__':
    run1 = {
        "psi0": psi0,
        "psi1": psi1,
        "phi": phi,
        "a": 0.6,
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
        "a": 0.6,
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
        "a": 0.6,
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
