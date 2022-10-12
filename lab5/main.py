import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import cm

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
    x: np.ndarray = np.arange(0, 1.0 + h/2.0, step=h)
    t: np.ndarray = np.arange(0, t_bound + tau/2.0, step=tau)

    kwargs = {
        "u_initial": u_initial,
        "u_left_border": u_left_border,
        "u_right_border": u_right_border,
        "a": a,
        "h": h,
        "tau": tau,
        "l": 0.0,
        "r": 1.0,
        "t_bound": t_bound
    }

    analytical = analytical_grid(a, x, t)

    print("---------------- EXPLICIT ----------------")
    sol = explicit_method(**kwargs)
    print(np.round(sol, 3))
    print("\nError: ", error(sol[-1], analytical[-1]))
    print("------------------------------------------\n")
    print("---------------- IMPLICIT ----------------")
    sol = implicit_method(**kwargs)
    print(np.round(sol, 3))
    print("\nError: ", error(sol[-1], analytical[-1]))
    print("------------------------------------------\n")
    print("------------- CRANK-NICOLSON -------------")
    sol = crank_nicolson_method(**kwargs)
    print(np.round(sol, 3))
    print("\nError: ", error(sol[-1], analytical[-1]))
    print("------------------------------------------\n")
    print("--------------- ANALYTICAL ---------------")
    print(np.round(analytical, 3))

    draw(sol, analytical, x, t)
