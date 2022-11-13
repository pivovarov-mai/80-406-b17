import copy
import numpy as np
from typing import List, Callable
from functools import partial

from logger import base_logger


def init_u(x: np.ndarray, y: np.ndarray,
           u_x_initial_1: Callable, u_y_initial_1: Callable,
           l: float, r: float) -> np.ndarray:
    u: np.ndarray = np.zeros(shape=(len(y), len(x)))
    u[:, -1] = u_y_initial_1(y)
    u[-1, :] = u_x_initial_1(x)
    for j in range(len(x) - 2, -1, -1):
        for i in range(len(y) - 2, -1, -1):
            u[i, j] = u[i + 1, j] * x[j] / (x[j] + y[i] + 0.0001)
            u[i, j] += u[i, j + 1] * y[i] / (x[j] + y[i] + 0.0001)
    return u


def simple_iter_method(u_y_initial_0_dx: Callable, u_y_initial_1: Callable,
                       u_x_initial_0_dy: Callable, u_x_initial_1: Callable,
                       h: float, l: float, r: float) -> np.ndarray:
    x: np.ndarray = np.arange(0, 1.0 + h/2.0, step=h)
    y: np.ndarray = np.arange(0, 1.0 + h/2.0, step=h)
    u: np.ndarray = init_u(x, y, u_x_initial_1, u_y_initial_1, l, r)

    eps: float = 1e-4
    prev: np.ndarray = np.zeros(shape=(len(y), len(x)))
    curr_iter: int = 0
    max_iter: int = 100
    diff: float = np.abs(u - prev).max()
    diverge_coef: float = 1.5
    while diff > eps and curr_iter <= max_iter:
        prev_diff: float = diff
        prev = copy.deepcopy(u)

        u[0, :-1] = (-2.0 * h * u_x_initial_0_dy(x[:-1]) + 4.0 * prev[1, :-1] - prev[2, :-1]) / 3.0
        u[:-1, 0] = (-2.0 * h * u_y_initial_0_dx(y[:-1]) + 4.0 * prev[:-1, 1] - prev[:-1, 2]) / 3.0
        for i in range(1, len(y) - 1):
            for j in range(1, len(x) - 1):
                u[i, j] = (prev[i-1, j] + prev[i+1, j] + prev[i, j-1] + prev[i, j+1]) / 4.0

        diff = np.abs(u - prev).max()
        curr_iter += 1
        if diff > diverge_coef * prev_diff:
            base_logger.warning("WARNING : Max_iter starts diverge on iter = %s", curr_iter)
            break

    if curr_iter >= max_iter:
        base_logger.warning("WARNING : Max_iter was reached")

    base_logger.info("INFO : iters count = %s", curr_iter)

    return u


def relaxation_iter_method(u_y_initial_0_dx: Callable, u_y_initial_1: Callable,
                           u_x_initial_0_dy: Callable, u_x_initial_1: Callable,
                           h: float, l: float, r: float,
                           w: float) -> np.ndarray:
    x: np.ndarray = np.arange(0, 1.0 + h/2.0, step=h)
    y: np.ndarray = np.arange(0, 1.0 + h/2.0, step=h)
    u: np.ndarray = init_u(x, y, u_x_initial_1, u_y_initial_1, l, r)

    eps: float = 1e-4
    prev: np.ndarray = np.zeros(shape=(len(y), len(x)))
    curr_iter: int = 0
    max_iter: int = 100
    diff: float = np.abs(u - prev).max()
    diverge_coef: float = 1.5
    while diff > eps and curr_iter <= max_iter:
        prev_diff: float = diff
        prev = copy.deepcopy(u)

        u[0, :-1] += w * ((-2.0 * h * u_x_initial_0_dy(x[:-1]) + 4.0 * u[1, :-1] - u[2, :-1]) / 3.0 - u[0, :-1])
        u[:-1, 0] += w * ((-2.0 * h * u_y_initial_0_dx(y[:-1]) + 4.0 * u[:-1, 1] - u[:-1, 2]) / 3.0 - u[:-1, 0])
        for i in range(1, len(y) - 1):
            for j in range(1, len(x) - 1):
                u[i, j] += w * ((u[i-1, j] + prev[i+1, j] + u[i, j-1] + prev[i, j+1]) / 4.0 - prev[i, j])

        diff = np.abs(u - prev).max()
        curr_iter += 1
        if diff > diverge_coef * prev_diff:
            base_logger.warning("WARNING : Max_iter starts diverge on iter = %s", curr_iter)
            break

    if curr_iter >= max_iter:
        base_logger.warning("WARNING : Max_iter was reached")

    base_logger.info("INFO : iters count = %s", curr_iter)

    return u


seidel_method = partial(relaxation_iter_method, w=1.0)
