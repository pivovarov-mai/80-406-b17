import numpy as np


def second_derivative(values: np.ndarray, step: float):
    derivatives = np.zeros(shape=values.shape)
    for i in range(1, len(values) - 1):
        derivatives[i] = (values[i - 1] - 2.0 * values[i] + values[i + 1]) / step**2
    return derivatives
