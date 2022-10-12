import numpy as np
import sys

from methods import implicit_method, explicit_method, crank_nicolson_method

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


if __name__ == "__main__":
    a = float(input("Enter parameter 'a': "))
    h = float(input("Enter step 'h': "))
    tau = float(input("Enter step 'tau': "))
    t_bound = float(input("Enter time border: "))
    x: np.ndarray = np.arange(0, 1.0 + h/2.0, step=h)
    t: np.ndarray = np.arange(0, t_bound + tau/2.0, step=tau)
    print(np.round(explicit_method(u_initial=u_initial,
                                   u_left_border=u_left_border,
                                   u_right_border=u_right_border,
                                   a=a, h=h, tau=tau, l=0.0, r=1.0, t_bound=t_bound), 3))
    print("==================")
    print(np.round(implicit_method(u_initial=u_initial,
                                   u_left_border=u_left_border,
                                   u_right_border=u_right_border,
                                   a=a, h=h, tau=tau, l=0.0, r=1.0, t_bound=t_bound), 3))
    print("==================")
    print(np.round(crank_nicolson_method(u_initial=u_initial,
                                         u_left_border=u_left_border,
                                         u_right_border=u_right_border,
                                         a=a, h=h, tau=tau, l=0.0, r=1.0, t_bound=t_bound), 3))
    print("==================")
    print(np.round(analytical_grid(a, x, t), 3))

