import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import cm

from methods import simple_iter_method

sys.path.append(".")


def analytical_solution(x: float, y: float) -> float:
    return x**2 - y**2


def analytical_grid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    grid: np.ndarray = np.zeros(shape=(len(y), len(x)))
    for i in range(len(y)):
        for j in range(len(x)):
            grid[i, j] = analytical_solution(x[j], y[i])
    return grid


def u_y_initial_0_dx(y: np.ndarray) -> np.ndarray:
    return np.zeros(len(y))


def u_y_initial_1(y: np.ndarray) -> np.ndarray:
    return 1.0 - y**2


def u_x_initial_0_dy(x: np.ndarray) -> np.ndarray:
    return np.zeros(len(x))


def u_x_initial_1(x: np.ndarray) -> np.ndarray:
    return x**2 - 1.0


def error(numeric: np.ndarray, analytical: np.ndarray) -> np.ndarray:
    return np.abs(numeric - analytical).max()


def draw(numerical: np.ndarray, analytical: np.ndarray,
         x: np.ndarray, y: np.ndarray,
         title_lhs: str, title_rhs: str):
    fig = plt.figure(figsize=plt.figaspect(0.7))
    xx, yy = np.meshgrid(x, y)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    plt.title(title_lhs)
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.set_zlabel('u', fontsize=20)
    ax.plot_surface(xx, yy, numerical, cmap=cm.coolwarm, linewidth=0, antialiased=True)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.set_zlabel('u', fontsize=20)
    plt.title(title_rhs)
    ax.plot_surface(xx, yy, analytical, cmap=cm.coolwarm, linewidth=0, antialiased=True)

    plt.show()


if __name__ == "__main__":
    h = float(input("Enter step 'h': "))
    x: np.ndarray = np.arange(0, 1.0 + h/2.0, step=h)
    y: np.ndarray = np.arange(0, 1.0 + h/2.0, step=h)

    kwargs = {
        "u_y_initial_0_dx": u_y_initial_0_dx,
        "u_y_initial_1": u_y_initial_1,
        "u_x_initial_0_dy": u_x_initial_0_dy,
        "u_x_initial_1": u_x_initial_1,
        "h": h,
        "l": 0.0,
        "r": 1.0
    }

    analytical = analytical_grid(x, y)

    print("---------------- SIMPLE ITER ----------------")
    sol = simple_iter_method(**kwargs)
    print(np.round(sol, 2))
    print("\nError: ", error(sol, analytical))
    print("---------------------------------------------\n")
    print("--------------- ANALYTICAL ---------------")
    print(np.round(analytical, 2))

    draw(sol, analytical, x, y, 'simple iter', 'analytic')
