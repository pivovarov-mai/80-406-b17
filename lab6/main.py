import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import cm

from methods import explicit_method
from approximator import Approx3p2a, Approx2p2a, Approx2p1a

sys.path.append(".")


def analytical_solution(a: float, x: float, t: float) -> float:
    return np.sin(x - a * t) + np.cos(x + a * t)


def analytical_grid(a: float, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    grid: np.ndarray = np.zeros(shape=(len(t), len(x)))
    for i in range(len(t)):
        for j in range(len(x)):
            grid[i, j] = analytical_solution(a, x[j], t[i])
    return grid


def u_initial(x: np.ndarray) -> np.ndarray:
    return np.sin(x) + np.cos(x)


def u_t_initial(a: float, x: np.ndarray) -> np.ndarray:
    return -a * (np.sin(x) + np.cos(x))


def u_x2_initial(x: np.ndarray) -> np.ndarray:
    return -u_initial(x)


def error(numeric: np.ndarray, analytical: np.ndarray) -> np.ndarray:
    return np.abs(numeric - analytical)


def draw(numerical: np.ndarray, analytical: np.ndarray,
         x: np.ndarray, t: np.ndarray):
    fig = plt.figure(figsize=plt.figaspect(0.7))
    xx, tt = np.meshgrid(x, t)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    plt.title('numerical')
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('t', fontsize=20)
    ax.set_zlabel('u', fontsize=20)
    ax.plot_surface(xx, tt, numerical, cmap=cm.coolwarm, linewidth=0, antialiased=True)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('t', fontsize=20)
    ax.set_zlabel('u', fontsize=20)
    plt.title('analytic')
    ax.plot_surface(xx, tt, analytical, cmap=cm.coolwarm, linewidth=0, antialiased=True)

    plt.show()


if __name__ == "__main__":
    a = float(input("Enter parameter 'a': "))
    h = float(input("Enter step 'h': "))
    tau = float(input("Enter step 'tau': "))
    t_bound = float(input("Enter time border: "))
    x: np.ndarray = np.arange(0, 3.14 + h/2.0, step=h)
    t: np.ndarray = np.arange(0, t_bound + tau/2.0, step=tau)

    kwargs = {
        "u_initial": u_initial,
        "u_t_initial": u_t_initial,
        "u_x2_initial": u_x2_initial,
        "a": a,
        "h": h,
        "tau": tau,
        "l": 0.0,
        "r": 3.14,
        "t_bound": t_bound
    }

    analytical = analytical_grid(a, x, t)

    print("---------------- EXPLICIT (2p1a) ----------------")
    approx = Approx2p1a()
    sol = explicit_method(**kwargs, approx=approx)
    print(np.round(sol, 3))
    print("\nError: ", error(sol[-1], analytical[-1]))
    print("-------------------------------------------------\n")
    print("---------------- EXPLICIT (3p2a) ----------------")
    approx = Approx3p2a()
    sol = explicit_method(**kwargs, approx=approx)
    print(np.round(sol, 3))
    print("\nError: ", error(sol[-1], analytical[-1]))
    print("-------------------------------------------------\n")
    print("---------------- EXPLICIT (2p2a) ----------------")
    approx = Approx2p2a()
    sol = explicit_method(**kwargs, approx=approx)
    print(np.round(sol, 3))
    print("\nError: ", error(sol[-1], analytical[-1]))
    print("-------------------------------------------------\n")
    print("--------------- ANALYTICAL ---------------")
    print(np.round(analytical, 3))

    draw(sol, analytical, x, t)
