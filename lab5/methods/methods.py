import numpy as np
from typing import List, Callable

from logger import base_logger
from sweep import sweep_solve
from derivatives import second_derivative


def explicit_method(u_initial: Callable, u_left_border: Callable, u_right_border: Callable,
                    a: float, h: float, tau: float,
                    l: float, r: float, t_bound: float) -> np.ndarray:
    if a * tau / h**2 > 0.5:
        base_logger.warning("WARNING : explicit method is not stable")

    x: np.ndarray = np.arange(l, r + h/2.0, step=h)
    t: np.ndarray = np.arange(0, t_bound + tau/2.0, step=tau)
    u: np.ndarray = np.zeros(shape=(len(t), len(x)))

    u[0] = u_initial(x)
    u[0, 0] = u_left_border()
    u[0, -1] = u_right_border()

    for k in range(len(t) - 1):
        u[k+1] = u[k] + tau * a * second_derivative(u[k], step=h)

    return u


def hybrid_method(u_initial: Callable, u_left_border: Callable, u_right_border: Callable,
                  a: float, h: float, tau: float,
                  l: float, r: float, t_bound: float, theta: float) -> np.ndarray:
    x: np.ndarray = np.arange(l, r + h/2.0, step=h)
    t: np.ndarray = np.arange(0, t_bound + tau/2.0, step=tau)
    u: np.ndarray = np.zeros(shape=(len(t), len(x)))

    u[0] = u_initial(x)
    u[0, 0] = u_left_border()
    u[0, -1] = u_right_border()

    for k in range(len(t) - 1):
        matrix: np.ndarray = np.zeros(shape=(len(x) - 2, len(x) - 2))
        matrix[0] += np.array(
            [
                -(1.0 + (2.0 * theta * a * tau) / h**2),
                (theta * a * tau) / h**2
            ]
            + [0.0] * (len(matrix) - 2)
        )
        target: List[float] = [(theta - 1.0) * a * tau * u[k][0] / h**2 +
                               (2.0 * (1.0 - theta) * a * tau / h**2 - 1.0) * u[k][1] +
                               (theta - 1.0) * a * tau * u[k][2] / h**2 -
                               theta * a * tau * u_left_border() / h**2]

        for i in range(1, len(matrix) - 1):
            matrix[i] += np.array(
                [0.0] * (i - 1)
                + [
                    theta * a * tau / h**2,
                    -(1.0 + (2.0 * theta * a * tau) / h**2),
                    (theta * a * tau) / h**2
                ]
                + [0.0] * (len(matrix) - i - 2)
            )
            target += [(theta - 1.0) * a * tau * u[k][i] / h**2 +
                       (2.0 * (1.0 - theta) * a * tau / h**2 - 1.0) * u[k][i+1] +
                       (theta - 1.0) * a * tau * u[k][i+2] / h**2]

        matrix[-1] += np.array(
            [0.0] * (len(matrix) - 2)
            + [
                theta * a * tau / h ** 2,
                -(1.0 + (2.0 * theta * a * tau) / h ** 2)
            ]
        )
        target += [(theta - 1.0) * a * tau * u[k][-3] / h**2 +
                   (2.0 * (1.0 - theta) * a * tau / h**2 - 1.0) * u[k][-2] +
                   (theta - 1.0) * a * tau * u[k][-1] / h**2 -
                   theta * a * tau * u_right_border() / h**2]

        u[k+1] += np.array([u_left_border()]
                           + sweep_solve(matrix, np.array(target)).tolist()
                           + [u_right_border()])

    return u


def implicit_method(**kwargs) -> np.ndarray:
    return hybrid_method(**kwargs, theta=1.0)


def crank_nicolson_method(**kwargs) -> np.ndarray:
    return hybrid_method(**kwargs, theta=0.5)
