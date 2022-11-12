import numpy as np
from typing import List, Callable

from logger import base_logger


def simple_iter_method(u_y_initial_0_dx: Callable, u_y_initial_1: Callable,
                       u_x_initial_0_dy: Callable, u_x_initial_1: Callable,
                       h: float, l: float, r: float) -> np.ndarray:
    pass
