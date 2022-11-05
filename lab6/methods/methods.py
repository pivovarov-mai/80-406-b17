import numpy as np
from typing import List, Callable

from logger import base_logger
from sweep import sweep_solve
from derivatives import second_derivative
from approximator import Approx


def explicit_method(u_initial: Callable, u_t_initial: Callable, u_x2_initial: Callable,
                    a: float, h: float, tau: float,
                    l: float, r: float, t_bound: float,
                    approx: Approx) -> np.ndarray:
    sigma: float = a * tau**2 / h**2
    if sigma > 1.0:
        base_logger.warning("WARNING : explicit method is not stable")

    x: np.ndarray = np.arange(l, r + h / 2.0, step=h)
    t: np.ndarray = np.arange(0, t_bound + tau / 2.0, step=tau)
    u: np.ndarray = np.zeros(shape=(len(t), len(x)))

    u[0] = u_initial(x)
    u[1] = u[0] + tau * u_t_initial(a, x) + tau**2 * u_x2_initial(x) / 2.0

    for k in range(1, len(t) - 1):
        u[k+1] = 2.0 * u[k] - u[k-1] + tau**2 * a * second_derivative(u[k], step=h)
        u[k+1, 0] = approx.explicit_0(h, sigma, u, k, tau)
        u[k+1, -1] = approx.explicit_l(h, sigma, u, k, tau)

    return u


def implicit_method(u_initial: Callable, u_t_initial: Callable, u_x2_initial: Callable,
                    a: float, h: float, tau: float,
                    l: float, r: float, t_bound: float,
                    approx: Approx) -> np.ndarray:
    sigma: float = a * tau ** 2 / h ** 2
    x: np.ndarray = np.arange(l, r + h/2.0, step=h)
    t: np.ndarray = np.arange(0, t_bound + tau/2.0, step=tau)
    u: np.ndarray = np.zeros(shape=(len(t), len(x)))

    u[0] = u_initial(x)
    u[1] = u[0] + tau * u_t_initial(a, x) + tau**2 * u_x2_initial(x) / 2.0

    for k in range(1, len(t) - 1):
        start = approx.implicit_0(h, sigma, u, k)
        end = approx.implicit_l(h, sigma, u, k)
        matrix: np.ndarray = np.zeros(shape=(len(x), len(x)))
        matrix[0] += np.array(
            [start[0], start[1]] + [0.0] * (len(matrix) - 2)
        )
        target: List[float] = [start[2]]

        for i in range(1, len(matrix) - 1):
            matrix[i] += np.array(
                [0.0] * (i - 1)
                + [
                    1.0,
                    -(2.0 + 1.0 / sigma),
                    1.0
                ]
                + [0.0] * (len(matrix) - i - 2)
            )
            target += [(-2.0 * u[k][i] + u[k-1][i]) / sigma]

        matrix[-1] += np.array(
            [0.0] * (len(matrix) - 2) + [end[0], end[1]]
        )
        target += [end[2]]

        u[k+1] = sweep_solve(matrix, np.array(target)).tolist()

    return u
