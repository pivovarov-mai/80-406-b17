import numpy as np
import sys

from sweep import sweep_solve
from derivatives import second_derivative

sys.path.append(".")


def analytical_solution(a: float, x: float, t: float) -> float:
    assert(a > 0.0)
    return x + np.exp(-np.pi**2 * a * t) * np.sin(np.pi * x)


def analytical_grid(a: float, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    grid: np.ndarray = np.zeros(shape=(len(t), len(x)))
    for i in range(len(t)):
        for j in range(len(x)):
            grid[i, j] = analytical_solution(a, x[j], t[i])
    return grid


def u_initial(x: np.ndarray) -> np.ndarray:
    return x + np.sin(np.pi * x)


def u_left_border():
    return 0.0


def u_right_border():
    return 1.0


def explicit_method(a: float, h: float, tau: float,
                    l: float, r: float, t_bound: float) -> np.ndarray:
    x: np.ndarray = np.arange(l, r + h/2.0, step=h)
    t: np.ndarray = np.arange(0, t_bound + tau/2.0, step=tau)
    u: np.ndarray = np.zeros(shape=(len(t), len(x)))

    u[0] = u_initial(x)
    u[0, 0] = u_left_border()
    u[0, -1] = u_right_border()

    for k in range(0, len(u) - 1):
        u[k+1] = u[k] + tau * a * second_derivative(u[k], step=h)

    return u


def hybrid_method() -> np.ndarray:



if __name__ == "__main__":
    a = float(input("Enter parameter 'a': "))
    h = float(input("Enter step 'h': "))
    tau = float(input("Enter step 'tau': "))
    t_bound = float(input("Enter time border: "))
    x: np.ndarray = np.arange(0, 1.0 + h/2.0, step=h)
    t: np.ndarray = np.arange(0, t_bound + tau/2.0, step=tau)
    print(np.round(explicit_method(a, h, tau, 0.0, 1.0, t_bound), 3))
    print("==================")
    print(np.round(analytical_grid(a, x, t), 3))

