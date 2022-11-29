import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot(approx,
         analytic,
         x,
         y,
         tp,
         t,
         filename,
         label):
    fig = plt.figure(figsize=plt.figaspect(0.7))
    plt.subplots_adjust(left=0.1,
                        right=0.9,
                        bottom=0.1,
                        top=0.9,
                        hspace=0.5,
                        wspace=0.5)

    x_mesh, y_mesh = np.meshgrid(x, y)

    for i, ti in enumerate(tp):
        ax = fig.add_subplot(len(tp),
                             2,
                             (len(tp) - 1) * i + 1,
                             projection='3d')

        plt.title(f'{label} t={t[ti]}')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.plot_surface(x_mesh,
                        y_mesh,
                        approx[ti],
                        cmap=cm.BrBG)

        ax = fig.add_subplot(len(tp),
                             2,
                             (len(tp) - 1) * i + 2,
                             projection='3d')

        plt.title(f'analytic t={t[ti]}')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.plot_surface(x_mesh,
                        y_mesh,
                        analytic[ti],
                        cmap=cm.BrBG)

    plt.savefig(filename)
