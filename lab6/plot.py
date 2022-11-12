import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot(approx, analytic, x, t, err, filenames):
    fig = plt.figure(figsize=plt.figaspect(0.7))
    x_mesh, t_mesh = np.meshgrid(x, t)

    ax = fig.add_subplot(1, 2, 1, projection='3d')

    plt.title('approx')
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('t', fontsize=14)
    ax.set_zlabel('u', fontsize=14)

    ax.plot_surface(x_mesh,
                    t_mesh,
                    approx,
                    cmap=cm.BrBG)

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    plt.title('analytic')
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('t', fontsize=14)
    ax.set_zlabel('u', fontsize=14)

    ax.plot_surface(x_mesh,
                    t_mesh,
                    analytic,
                    cmap=cm.BrBG)

    plt.savefig(filenames[0])

    fig = plt.figure(figsize=plt.figaspect(0.7))

    plt.title('error')
    plt.xlabel('t')
    plt.ylabel('error')
    plt.plot(t, err, color='green')

    plt.savefig(filenames[1])
