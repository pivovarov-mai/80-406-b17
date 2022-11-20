import numpy as np
from typing import List, Callable

from sweep import sweep_solve


def common_algo(u_yt_initial_0: Callable, u_yt_initial_1: Callable,
                u_xt_initial_0: Callable, u_xt_initial_1: Callable,
                u_xy_initial_0,
                a: float, hx: float, hy: float, tau: float,
                lx: float, rx: float,
                ly: float, ry: float,
                t_bound: float,
                coef: float) -> np.ndarray:
    x: np.ndarray = np.arange(lx, rx + hx / 2.0, step=hx)
    y: np.ndarray = np.arange(ly, ry + hy / 2.0, step=hy)
    t: np.ndarray = np.arange(0, t_bound + tau / 4.0, step=tau / 2.0)
    u: np.ndarray = np.zeros(shape=(len(t), len(y), len(x)))

    for i, yi in enumerate(y):
        for j, xj in enumerate(x):
            u[0, i, j] = u_xy_initial_0(xj, yi)
    for i, ti in enumerate(t):
        for j, yj in enumerate(y):
            u[i, j, 0] = u_yt_initial_0(yj, ti)
            u[i, j, -1] = u_yt_initial_1(yj, ti)
    for i, ti in enumerate(t):
        for j, xj in enumerate(x):
            u[i, 0, j] = u_xt_initial_0(xj, ti)
            u[i, -1, j] = u_xt_initial_1(xj, ti)

    for k in range(0, len(t) - 2, 2):
        for i in range(1, len(y) - 1):
            matrix: np.ndarray = np.zeros(shape=(len(x) - 2, len(x) - 2))
            matrix[0] += np.array(
                [
                    -(2.0 * a * tau * hy**2 + (1.0 + coef) * hx**2 * hy**2),
                    a * tau * hy**2
                ]
                + [0.0] * (len(matrix) - 2)
            )
            target: List[float] = [-a * tau * hx**2 * coef * u[k, i-1, 1] -
                                   ((1.0 + coef) * hx**2 * hy**2 - 2.0 * a * tau * hx**2 * coef) * u[k, i, 1] -
                                   a * tau * hx**2 * coef * u[k, i+1, 1] -
                                   a * tau * hy**2 * u[k+1, i, 0]]

            for j in range(1, len(matrix) - 1):
                matrix[j] += np.array(
                    [0.0] * (j - 1)
                    + [
                        a * tau * hy**2,
                        -(2.0 * a * tau * hy**2 + (1.0 + coef) * hx**2 * hy**2),
                        a * tau * hy**2
                    ]
                    + [0.0] * (len(matrix) - j - 2)
                )
                target += [-a * tau * hx**2 * coef * u[k, i-1, j+1] -
                           ((1.0 + coef) * hx**2 * hy**2 - 2.0 * a * tau * hx**2 * coef) * u[k, i, j+1] -
                           a * tau * hx**2 * coef * u[k, i+1, j+1]]

            matrix[-1] += np.array(
                [0.0] * (len(matrix) - 2)
                + [
                    a * tau * hy ** 2,
                    -(2.0 * a * tau * hy**2 + (1.0 + coef) * hx**2 * hy**2)
                ]
            )
            target += [-a * tau * hx**2 * coef * u[k, i-1, -2] -
                       ((1.0 + coef) * hx**2 * hy**2 - 2.0 * a * tau * hx**2 * coef) * u[k, i, -2] -
                       a * tau * hx**2 * coef * u[k, i+1, -2] -
                       a * tau * hy**2 * u[k+1, i, -1]]

            u[k+1, i] += np.array([0.0] + sweep_solve(matrix, np.array(target)).tolist() + [0.0])

        for j in range(1, len(x) - 1):
            matrix: np.ndarray = np.zeros(shape=(len(y) - 2, len(y) - 2))
            matrix[0] += np.array(
                [
                    -(2.0 * a * tau * hx ** 2 + (1.0 + coef) * hx ** 2 * hy ** 2),
                    a * tau * hx ** 2
                ]
                + [0.0] * (len(matrix) - 2)
            )
            target: List[float] = [-a * tau * hy ** 2 * coef * u[k+1, 1, j-1] -
                                   ((1.0 + coef) * hx ** 2 * hy ** 2 - 2.0 * a * tau * hy ** 2 * coef) * u[k+1, 1, j] -
                                   a * tau * hy ** 2 * coef * u[k+1, 1, j+1] -
                                   a * tau * hx ** 2 * u[k+2, 0, j]]

            for i in range(1, len(matrix) - 1):
                matrix[i] += np.array(
                    [0.0] * (i - 1)
                    + [
                        a * tau * hx ** 2,
                        -(2.0 * a * tau * hx**2 + (1.0 + coef) * hx**2 * hy**2),
                        a * tau * hx ** 2
                    ]
                    + [0.0] * (len(matrix) - i - 2)
                )
                target += [-a * tau * hy ** 2 * coef * u[k+1, i+1, j-1] -
                           ((1.0 + coef) * hx**2 * hy**2 - 2.0 * a * tau * hy**2 * coef) * u[k+1, i+1, j] -
                           a * tau * hy**2 * coef * u[k+1, i+1, j+1]]

            matrix[-1] += np.array(
                [0.0] * (len(matrix) - 2)
                + [
                    a * tau * hx ** 2,
                    -(2.0 * a * tau * hx**2 + (1.0 + coef) * hx ** 2 * hy ** 2)
                ]
            )
            target += [-a * tau * hy ** 2 * coef * u[k+1, -2, j-1] -
                       ((1.0 + coef) * hx ** 2 * hy ** 2 - 2.0 * a * tau * hy ** 2 * coef) * u[k+1, -2, j] -
                       a * tau * hy ** 2 * coef * u[k+1, -2, j+1] -
                       a * tau * hx ** 2 * u[k+2, -1, j]]

            u[k+2, :, j] += np.array([0.0] + sweep_solve(matrix, np.array(target)).tolist() + [0.0])

    return u[::2]


def fractional_steps_method(u_yt_initial_0: Callable, u_yt_initial_1: Callable,
                            u_xt_initial_0: Callable, u_xt_initial_1: Callable,
                            u_xy_initial_0,
                            a: float, hx: float, hy: float, tau: float,
                            lx: float, rx: float,
                            ly: float, ry: float,
                            t_bound: float) -> np.ndarray:
    return common_algo(coef=0.0, **locals())


def alternating_directions_methods(u_yt_initial_0: Callable, u_yt_initial_1: Callable,
                                   u_xt_initial_0: Callable, u_xt_initial_1: Callable,
                                   u_xy_initial_0,
                                   a: float, hx: float, hy: float, tau: float,
                                   lx: float, rx: float,
                                   ly: float, ry: float,
                                   t_bound: float) -> np.ndarray:
    return common_algo(coef=1.0, **locals())
