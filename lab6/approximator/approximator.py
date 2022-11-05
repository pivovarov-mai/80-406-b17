import numpy as np


class Approx:
    def __init__(self):
        pass

    def explicit_0(self, h: float, sigma: float,
                   u: np.ndarray,
                   k: float, tau: float):
        pass

    def explicit_l(self, h: float, sigma: float,
                   u: np.ndarray,
                   k: float, tau: float):
        pass

    def implicit_0(self, h: float, sigma: float,
                   u: np.ndarray, k: np.ndarray):
        pass

    def implicit_l(self, h: float, sigma: float,
                   u: np.ndarray, k: np.ndarray):
        pass


class Approx2p1a(Approx):
    def explicit_0(self, h: float, sigma: float,
                   u: np.ndarray,
                   k: float, tau: float):
        return u[k+1, 1] / (1.0 + h)

    def explicit_l(self, h: float, sigma: float,
                   u: np.ndarray,
                   k: float, tau: float):
        return u[k+1, -2] / (1.0 - h)

    def implicit_0(self, h: float, sigma: float,
                   u: np.ndarray, k: np.ndarray):
        return (1.0 + h), -1.0, 0.0

    def implicit_l(self, h: float, sigma: float,
                   u: np.ndarray, k: np.ndarray):
        return -1.0, (1.0 - h), 0.0


class Approx3p2a(Approx):
    def explicit_0(self, h: float, sigma: float,
                   u: np.ndarray,
                   k: float, tau: float):
        return (4.0 * u[k+1][1] - u[k+1][2]) / (3.0 + 2.0 * h)

    def explicit_l(self, h: float, sigma: float,
                   u: np.ndarray,
                   k: float, tau: float):
        return (4.0 * u[k+1][-2] - u[k+1][-3]) / (3.0 - 2.0 * h)

    def implicit_0(self, h: float, sigma: float,
                   u: np.ndarray, k: np.ndarray):
        return (-(2.0 + 2.0 * h),
                -(1.0 / sigma - 2.0),
                (-2.0 * u[k][1] + u[k-1][1]) / sigma)

    def implicit_l(self, h: float, sigma: float,
                   u: np.ndarray, k: np.ndarray):
        return (-(1.0 / sigma - 2.0),
                -(2.0 - 2.0 * h),
                (-2.0 * u[k][-2] + u[k-1][-2]) / sigma)


class Approx2p2a(Approx):
    def explicit_0(self, h: float, sigma: float,
                   u: np.ndarray,
                   k: float, tau: float):
        return sigma * (2.0 * u[k][1] - (2.0 + 2.0 * h) * u[k][0]) + \
               2.0 * u[k][0] - u[k-1][0]

    def explicit_l(self, h: float, sigma: float,
                   u: np.ndarray,
                   k: float, tau: float):
        return sigma * (2.0 * u[k][-2] + (2.0 * h - 2.0) * u[k][-1]) + \
               2.0 * u[k][-1] - u[k-1][-1]

    def implicit_0(self, h: float, sigma: float,
                   u: np.ndarray, k: np.ndarray):
        return (-(2.0 + 2.0 * h + 1.0 / sigma),
                2.0,
                (-2.0 * u[k][0] + u[k-1][0]) / sigma)

    def implicit_l(self, h: float, sigma: float,
                   u: np.ndarray, k: np.ndarray):
        return (2.0,
                -(2.0 - 2.0 * h + 1.0 / sigma),
                (-2.0 * u[k][-1] + u[k-1][-1]) / sigma)
