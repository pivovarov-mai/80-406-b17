import numpy as np
from typing import List, Callable

from logger import base_logger
from sweep import sweep_solve


def fractional_steps_method(u_yt_initial_0: Callable, u_yt_initial_1: Callable,
                            u_xt_initial_0: Callable, u_xt_initial_1: Callable,
                            u_xy_initial_0,
                            a: float, hx: float, hy: float, tau: float,
                            l: float, r: float, t_bound: float) -> np.ndarray:
    pass


def alternating_directions_methods(u_yt_initial_0: Callable, u_yt_initial_1: Callable,
                                   u_xt_initial_0: Callable, u_xt_initial_1: Callable,
                                   u_xy_initial_0,
                                   a: float, hx: float, hy: float, tau: float,
                                   l: float, r: float, t_bound: float) -> np.ndarray:
    pass
