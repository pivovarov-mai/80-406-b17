import numpy as np
import matplotlib.pyplot as plt

class Task():
    def __init__(self, ulx, urx, uly, ury, u0, a, b, c, d, f, x0, x1, y0, y1):
        self.ulx = ulx
        self.urx = urx
        self.uly = uly
        self.ury = ury
        self.u0 = u0
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.f = f

        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1

# Аналитическое решение
def analytic(x_0, x_1, y_0, y_1, end_time, func, x_res, y_res, t_res):
    x = (x_1 - x_0) / (x_res-1)
    y = (y_1 - y_0) / (y_res-1)
    t = (end_time) / (t_res-1)
    u = np.zeros(shape=(t_res, y_res, x_res))

    for k in range(0, t_res):
        _t = t * k
        for j in range(0, y_res):
            _y = y_0 + j * y
            for i in range(0, x_res):
                _x = x_0 + i * x
                u[k][j][i] = func(_x, _y, _t)
    return u

def tridiagonal_matrix_algorithm(a, b, c, d):
    n = len(a)

    p = np.zeros(n)
    q = np.zeros(n)

    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]

    for i in range(1, n):
        p[i] = -c[i] / (b[i] + a[i] * p[i - 1])
        q[i] = (d[i] - a[i] * q[i - 1]) / (b[i] + a[i] * p[i - 1])

    x = np.zeros(n)
    x[-1] = q[-1]

    for i in range(n - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]

    return x

# Метод переменных направлений
def pdir(task: Task, x_res, y_res, t_res, end_time):
    x = (task.x1 - task.x0) / (x_res-1)
    y = (task.y1 - task.y0) / (y_res-1)
    tau = (end_time) / (t_res-1)

    u = np.zeros(shape=(t_res, y_res, x_res))
    # Заполнение первого слоя
    for j in range(0, y_res):
        _y = task.y0 + j * y
        for i in range(0, x_res):
            _x = task.x0 + i * x
            u[0][j][i] = task.u0(_x,_y)
    
    for k in range(1, t_res):
        # y
        _t = k * tau
        u1 = np.zeros(shape=(y_res, x_res))
        _t2 = _t - tau / 2
        for j in range(y_res - 1):
            _y = task.y0 + j * y

            a = np.zeros(x_res)
            b = np.zeros(x_res)
            c = np.zeros(x_res)
            d = np.zeros(x_res)
            
            a[0] = 0
            b[0] = x
            c[0] = 0
            d[0] = task.ulx(_y, _t2) * x

            a[-1] = 0
            b[-1] = x
            c[-1] = 0
            d[-1] = task.urx(_y, _t2) * x

            for i in range(x_res - 1):
                _x = task.x0 + i * x

                a[i] = task.a - x * task.c / 2
                b[i] = x ** 2 - 2 * (x ** 2) / tau - 2 * task.a
                c[i] = task.a + x * task.c / 2
                d[i] = -2 * (x ** 2) * u[k - 1][j][i] / tau
                - task.b * (x ** 2) * (u[k - 1][j + 1][i] - 2 * u[k - 1][j][i] + u[k-1][j-1][i]) / (y ** 2)
                - task.d * (x ** 2) * (u[k-1][j+1][i] - u[k-1][j-1][i]) / (2 * y ** 2)
                - (x ** 2) * task.f(_x, _y, _t)

            xx = tridiagonal_matrix_algorithm(a, b, c, d)
            for i in range(x_res):
                _x = task.x0 + i * x
                u1[j][i] = xx[i]
                u1[0][i] = task.uly(_x, _t2)
                u1[-1][i] = task.ury(_x, _t2)
        
        # x bounds
        for j in range(y_res):
            _y = task.y0 + j * y
            u1[j][0] = task.ulx(_y, _t2)
            u1[j][-1] = task.urx(_y, _t2)
        # x
        u2 = np.zeros((y_res, x_res))
        for i in range(x_res - 1):
            _x = task.x0 + i * x
            a = np.zeros(y_res)
            b = np.zeros(y_res)
            c = np.zeros(y_res)
            d = np.zeros(y_res)

            a[0] = 0
            b[0] = y
            c[0] = 0
            d[0] = task.uly(_x, _t) * y
            
            a[-1] = 0
            b[-1] = y
            c[-1] = 0
            d[-1] = task.ury(_x, _t) * y

            for j in range(y_res - 1):
                _y = task.y0 + j * y
                a[j] = task.b - y * task.d / 2
                b[j] = y ** 2 - 2 * (y ** 2) / tau - 2 * task.b
                c[j] = task.b + y * task.d / 2
                d[j] = -2 * (y ** 2) * u1[j][i] / tau
                - task.a * (y ** 2) * (u1[j][i + 1] - 2 * u1[j][i] + u1[j][i - 1]) / (x ** 2)
                - task.c * (y ** 2) * (u1[j][i + 1] - u1[j][i - 1]) / (2 * x ** 2)
                - (y ** 2) * task.f(_x, _y, _t)

            xx = tridiagonal_matrix_algorithm(a, b, c, d)
            for j in range(y_res):
                _y = task.y0 + j * y
                u2[j][i] = xx[j]
                u2[j][0] = task.ulx(_y, _t)
                u2[j][-1] = task.urx(_y, _t)
        
        # x bounds
        for i in range(x_res):
            _x = task.x0 + i * x
            u2[0][i] = task.uly(_x, _t)
            u2[-1][i] = task.ury(_x, _t)

        # copy
        for i in range(x_res):
            for j in range(y_res):
                u[k][j][i] = u2[j][i]
    
    return u

# Метод дробных шагов
def frac_steps(task: Task, x_res, y_res, t_res, end_time):
    x = (task.x1 - task.x0) / (x_res-1)
    y = (task.y1 - task.y0) / (y_res-1)
    tau = (end_time) / (t_res-1)

    u = np.zeros(shape=(t_res, y_res, x_res))
    # Заполнение первого слоя
    for j in range(0, y_res):
        _y = task.y0 + j * y
        for i in range(0, x_res):
            _x = task.x0 + i * x
            u[0][j][i] = task.u0(_x,_y)

    for k in range(t_res):
        _t = k * tau
        u1 = np.zeros((y_res, x_res))
        _t2 = _t - tau / 2
        for j in range(y_res - 1):
            _y = task.y0 + j * y

            a = np.zeros(x_res)
            b = np.zeros(x_res)
            c = np.zeros(x_res)
            d = np.zeros(x_res)
            
            a[0] = 0
            b[0] = x
            c[0] = 0
            d[0] = task.ulx(_y, _t2) * x

            a[-1] = 0
            b[-1] = x
            c[-1] = 0
            d[-1] = task.urx(_y, _t2) * x

            for i in range(x_res - 1):
                _x = task.x0 + i * x
                a[i] = task.a
                b[i] = -(x ** 2) / tau - 2 * task.a
                c[i] = task.a
                d[i] = -(x ** 2) * u[k - 1][j][i] / tau - (x ** 2) * task.f(_x, _y, _t2) / 2
            
            xx = tridiagonal_matrix_algorithm(a, b, c, d)
            for i in range(x_res):
                _x = task.x0 + i * x
                u1[j][i] = xx[i]
                u1[0][i] = task.uly(_x, _t2)
                u1[-1][i] = task.ury(_x, _t2)

        for j in range(y_res):
            _y = task.y0 + j * y
            u1[j][0] = task.ulx(_x, _t2)
            u1[j][-1] = task.urx(_x, _t2)

        u2 = np.zeros((y_res, x_res))
        for i in range(x_res - 1):
            _x = task.x0 + i * x
            a = np.zeros(y_res)
            b = np.zeros(y_res)
            c = np.zeros(y_res)
            d = np.zeros(y_res)

            a[0] = 0
            b[0] = y
            c[0] = 0
            d[0] = task.uly(_x, _t) * y
            
            a[-1] = 0
            b[-1] = y
            c[-1] = 0
            d[-1] = task.ury(_x, _t) * y

            for j in range(y_res - 1):
                _y = task.y0 + j * y
                a[j] = task.b
                b[j] = -(y ** 2) / tau - 2 * task.b
                c[j] = task.b
                d[j] = -(y ** 2) * u1[j][i] / tau - (y ** 2) * task.f(_x, _y, _t) / 2
            
            xx = tridiagonal_matrix_algorithm(a, b, c, d)
            for j in range(y_res):
                _y = task.y0 + j * y
                u2[j][i] = xx[j]
                u2[j][0] = task.ulx(_y, _t)
                u2[j][-1] = task.urx(_y, _t)
        
        for i in range(x_res):
            _x = task.x0 + i * x
            u2[0][i] = task.uly(_x, _t)
            u2[-1][i] = task.ury(_x, _t)

        for i in range(x_res):
            for j in range(y_res):
                u[k][j][i] = u2[j][i]
    return u