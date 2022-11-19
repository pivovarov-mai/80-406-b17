import numpy as np
import sys
import matplotlib.pyplot as plt

from matplotlib import cm
from typing import Callable, List
from functools import partial

from methods import fractional_steps_method, alternating_directions_methods

sys.path.append(".")


def analytical_solution(a: float,
                        x: float, y: float,
                        t: float,
                        mu1: float, mu2: float) -> float:
    return np.cos(mu1, x) * np.cos(mu1, y) * np.exp(-(mu1**2 + mu2**2) * a * t)


def analytical_grid(a: float, x: np.ndarray, y: np.ndarray, t: np.ndarray,
                    afunc: Callable) -> np.ndarray:
    grid: np.ndarray = np.zeros(shape=(len(t), len(y), len(x)))
    for i in range(len(t)):
        for j in range(len(y)):
            for k in range(len(x)):
                grid[i, j, k] = afunc(a, x[k], y[j], t[i])
    return grid


def u_yt_initial_0(y: float, t: float,
                   mu1: float, mu2: float) -> float:
    return np.cos(mu2, y) * np.exp(-(mu1**2 + mu2**2) * a * t)


def u_yt_initial_1(y: float, t: float,
                   mu1: float, mu2: float) -> float:
    return 0.0


def u_xt_initial_0(x: float, t: float,
                   mu1: float, mu2: float) -> float:
    return np.cos(mu1, x) * np.exp(-(mu1**2 + mu2**2) * a * t)


def u_xt_initial_1(x: float, t: float,
                   mu1: float, mu2: float) -> float:
    return 0.0


def u_xy_initial_0(x: float, y: float,
                   mu1: float, mu2: float) -> float:
    return np.cos(mu1, x) * np.cos(mu2, y)


def error(numeric: np.ndarray, analytical: np.ndarray) -> np.ndarray:
    return np.max(np.abs(numeric - analytical))


def draw(numerical1: np.ndarray, label1: str,
         numerical2: np.ndarray, label2: str,
         analytical: np.ndarray,
         x: np.ndarray, y: np.ndarray, t: List[float]):
    fig = plt.figure(figsize=plt.figaspect(0.7))
    xx, yy = np.meshgrid(x, y)

    ax = fig.add_subplot(len(t), 3, 1, projection='3d')
    for ti in t:
        plt.title(label1 + f', t = {ti}')
        ax.set_xlabel('x', fontsize=20)
        ax.set_ylabel('y', fontsize=20)
        ax.set_zlabel(f'u[t={ti}]', fontsize=20)
        ax.plot_surface(xx, yy, numerical1[ti], cmap=cm.coolwarm, linewidth=0, antialiased=True)

        ax.set_xlabel('x', fontsize=20)
        ax.set_ylabel('y', fontsize=20)
        ax.set_zlabel(f'u[t={ti}]', fontsize=20)
        plt.title(label2 + f', t = {ti}')
        ax.plot_surface(xx, yy, numerical2[ti], cmap=cm.coolwarm, linewidth=0, antialiased=True)

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.set_xlabel('x', fontsize=20)
        ax.set_ylabel('y', fontsize=20)
        ax.set_zlabel(f'u[t={ti}]', fontsize=20)
        plt.title(f'analytic, t = {ti}')
        ax.plot_surface(xx, yy, analytical[ti], cmap=cm.coolwarm, linewidth=0, antialiased=True)

    plt.show()


if __name__ == "__main__":
    a = float(input("Enter parameter 'a': "))
    hx = float(input("Enter step 'hx': "))
    hy = float(input("Enter step 'hy': "))
    tau = float(input("Enter step 'tau': "))
    t_bound = float(input("Enter time border: "))
    x: np.ndarray = np.arange(0, np.pi / 2.0 + hx / 2.0, step=hx)
    y: np.ndarray = np.arange(0, np.pi / 2.0 + hy / 2.0, step=hy)
    t: np.ndarray = np.arange(0, t_bound + tau / 2.0, step=tau)

    mu = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)]

    for mu1, mu2 in mu:
        kwargs = {
            "u_yt_initial_0": partial(u_yt_initial_0, mu1=mu1, mu2=mu2),
            "u_yt_initial_1": partial(u_yt_initial_1, mu1=mu1, mu2=mu2),
            "u_xt_initial_0": partial(u_xt_initial_0, mu1=mu1, mu2=mu2),
            "u_xt_initial_1": partial(u_xt_initial_1, mu1=mu1, mu2=mu2),
            "u_xy_initial_0": partial(u_xy_initial_0, mu1=mu1, mu2=mu2),
            "a": a,
            "hx": hx,
            "hy": hy,
            "tau": tau,
            "l": 0.0,
            "r": np.pi / 2.0,
            "t_bound": t_bound
        }

        analytical = analytical_grid(a, x, y, t, partial(analytical_solution,
                                                         mu1=mu1, mu2=mu2))

        print("---------------- FSM ----------------")
        sol1 = fractional_steps_method(**kwargs)
        print(np.round(sol1, 2))
        print("\nError: ", error(sol1, analytical))
        print("-------------------------------------\n")
        print("---------------- ADM ----------------")
        sol2 = alternating_directions_methods(**kwargs)
        print(np.round(sol2, 2))
        print("\nError: ", error(sol2, analytical))
        print("-------------------------------------\n")
        print("------------- ANALYTICAL ------------")
        print(np.round(analytical, 2))

        t_points = [0.0, np.pi / 4.0, np.pi / 2.0]
        draw(sol1, "FSM", sol2, "ADM", analytical, x, y, t_points)

        print("=====================================\n\n")
