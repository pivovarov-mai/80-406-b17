import numpy as np
import matplotlib.pyplot as plt
from tools import analitic_solution


def plot_stat(name_u, name_e, a, n, cnt, step, x_min, x_max, u):
    """
    Parameters:
        name_u: название файла с графиком сеточных функций
        name_e: название файла с графиком ошибки
        a: коэффициент температуропроводности
        n: количество точек в пространстве
        cnt: количество временных точек
        step: временной шаг
        x_min: левая граница
        x_max: правая граница
        u: сеточная функция
    """
    t = np.zeros(cnt)

    for i in range(cnt):
        t[i] = i * step

    space = np.zeros(n)
    h = (x_max - x_min) / n

    for i in range(n):
        space[i] = x_min + i * h

    t_idx = np.linspace(0, t.shape[0] - 1, 6, dtype=np.int32)

    sx, sy = 3, 2
    fig, ax = plt.subplots(sx, sy)
    fig.tight_layout()
    fig.suptitle('Сеточная функция')
    fig.set_figwidth(12)
    fig.set_figheight(7)

    k = 0
    for i in range(sx):
        for j in range(sy):
            idx = t_idx[k]
            sol = [analitic_solution(x, t[idx], a) for x in space]

            ax[i][j].plot(space, u[idx], label='Численный метод')
            ax[i][j].plot(space, sol, label='Аналитическое решение')

            ax[i][j].grid(True)
            ax[i][j].set_xlabel('x')
            ax[i][j].set_ylabel('u')
            ax[i][j].set_title(f'Решения при t = {t[idx]}')

            k += 1

    plt.legend()
    plt.savefig(name_u, dpi=300)

    plt.clf()
    plt.cla()

    error = np.zeros(cnt)
    for i in range(cnt):
        sol = [analitic_solution(x, t[i], a) for x in space]
        error[i] = np.max(np.abs(u[i] - np.array(sol)))

    plt.figure(figsize=(12, 7))
    plt.plot(t[1:], error[1:], 'violet', label='Ошибка')
    plt.title('График изменения ошибки во времени')
    plt.xlabel('t')
    plt.ylabel('error')
    plt.grid(True)
    plt.legend()

    plt.savefig(name_e, dpi=300)

    plt.clf()
    plt.cla()
