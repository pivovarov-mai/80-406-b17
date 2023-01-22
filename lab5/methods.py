import numpy as np
import matplotlib.pyplot as plt

# Описание задачи
class Task():
    def __init__(self, u0, ul, l, ur, r, f, a, b, c):
        self.u0 = u0
        self.ul = ul
        self.l = l
        self.ur = ur
        self.r = r
        self.f = f
        self.a = a
        self.b = b
        self.c = c

# Аналитическое решение
def analytic(l_bound, r_bound, func, end_time, t_res, h_res):
    h = (r_bound - l_bound) / h_res
    tau = end_time / t_res
    u = np.zeros(shape=(t_res, h_res))

    for t_itr in range(0, t_res):
        for x in range(0, h_res):
            u[t_itr][x] = func(l_bound + x * h, t_itr * tau)
    
    return u

# Явная схема
def explicit(task: Task, end_time, t_res, h_res, approx='1p1'):
        h = (task.r - task.l) / h_res
        tau = end_time / t_res
        sigma = (task.a * tau)/(h**2)

        if sigma > 0.5:
            raise ValueError(f"Sigma: {sigma}")
        
        u = np.zeros((t_res, h_res))
        u[0] = task.u0(np.arange(task.l, task.r, h))

        for k in range(1, t_res):
            for j in range(1, h_res - 1):
                u[k][j] = sigma * u[k - 1][j + 1] + (1 - 2 * sigma) * u[k - 1][j] + sigma * u[k - 1][j - 1] 
                u[k][j] += tau * task.f(task.l + j * h, k * tau)
           
            if approx == '1p2':
                u[k][0] = u[k][1] - h * task.ul(k * tau)
                u[k][-1] = u[k][-2] + h * task.ur(k * tau)
                    
            elif approx == '2p2':
                u[k][0] = (u[k][1] - h * task.ul(k * tau) + (h ** 2 / (2 * tau) * u[k - 1][0])) / (1 + h ** 2 / (2 * tau))
                u[k][-1] = (u[k][-2] + h * task.ur(k * tau) + (h ** 2 / (2 * tau) * u[k - 1][-1])) / (1 + h ** 2 / (2 * tau))
        

            elif approx == '2p3':
                u[k][0] = (task.ul(k * tau) + u[k][2] / (2 * h) - 2 * u[k][1] / h) * 2 * h / -3
                u[k][-1] = (task.ur(k * tau) - u[k][-3] / (2 * h) + 2 * u[k][-2] / h) * 2 * h / 3

        return u

# Метод прогонки
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

# Метод прогонки
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

# Неявная схема
def implicit(task: Task, end_time, t_res, h_res, approx='1p2'):
    h = (task.r - task.l) / h_res
    tau = end_time / t_res
    sigma = (task.a * tau)/(h**2)

    a = np.zeros(h_res)
    b = np.zeros(h_res)
    c = np.zeros(h_res)
    d = np.zeros(h_res)
    u = np.zeros((t_res, h_res))
    u[0] = task.u0(np.arange(task.l, task.r, h))

    for k in range(1, t_res):
        for j in range(1, h_res - 1):
            a[j] = sigma
            b[j] = -(1 + 2 * sigma)
            c[j] = sigma
            d[j] = -u[k - 1][j] - tau * task.f(task.l + j * h, k * tau)

        if approx == '1p2':
            a[0] = 0
            b[0] = -1 / h
            c[0] = 1 / h
            d[0] = task.ul(k * tau)
            a[-1] = -1 / h
            b[-1] = 1 / h
            c[-1] = 0
            d[-1] = task.ur(k * tau)

        elif approx == '2p2':  
            b[0] = 2 * task.a ** 2 / h + h / tau
            c[0] = - 2 * task.a ** 2 / h
            d[0] = (h / tau) * u[k - 1][0] - task.ul(k * tau) * 2 * task.a ** 2
            a[-1] = -2 * task.a ** 2 / h
            b[-1] = 2 * task.a ** 2 / h + h / tau
            d[-1] = (h / tau) * u[k - 1][-1] + task.ur(k * tau) * 2 * task.a ** 2
        elif approx == '2p3':
            k0 = 1 / (2 * h) / c[1]
            b[0] = (-3 / (2 * h) + a[1] * k0)
            c[0] = 2 / h + b[1] * k0
            d[0] = task.ul(k* tau) + d[1] * k0
            k1 = -(1 / (h * 2)) / a[-2]
            a[-1] = (-2 / h) + b[-2] * k1
            b[-1] = (3 / (h * 2)) + c[-2] * k1
            d[-1] = task.ur(k* tau) + d[-2] * k1

        u[k] = tridiagonal_matrix_algorithm(a, b, c, d)

    return u

# Кобинированный метод
def combined_method(task: Task, end_time, t_res, h_res, theta=0.5, approx='1p2'):
    h = (task.r - task.l) / h_res
    tau = end_time / t_res
    sigma = (task.a * tau)/(h**2)
    
    a = np.zeros(h_res)
    b = np.zeros(h_res)
    c = np.zeros(h_res)
    d = np.zeros(h_res)
    tmp_imp = np.zeros(h_res)
    u = np.zeros((t_res, h_res))

    u[0] = task.u0(np.arange(task.l, task.r, h))

    for k in range(1, t_res):
        for j in range(1, h_res - 1):
            a[j] = sigma * theta
            b[j] = -(1 + 2 * sigma * theta)
            c[j] = sigma * theta
            d[j] = -u[k - 1][j] - tau * task.f(task.l + j * h, k * tau)
            d[j] -= (1-theta)*sigma*(u[k - 1][j + 1] - 2*u[k - 1][j] + u[k - 1][j - 1])

        if approx == '1p2': 
            a[0] = 0
            b[0] = -1 / h
            c[0] = 1 / h
            d[0] = task.ul(k * tau)
            a[-1] = -1 / h
            b[-1] = 1 / h
            c[-1] = 0
            d[-1] = task.ur(k * tau)

        elif approx == '2p2':  
            b[0] = 2 * task.a ** 2 / h + h / tau
            c[0] = - 2 * task.a ** 2 / h
            d[0] = (h / tau) * u[k - 1][0] - task.ul(k * tau) * 2 * task.a ** 2
            a[-1] = -2 * task.a ** 2 / h
            b[-1] = 2 * task.a ** 2 / h + h / tau
            d[-1] = (h / tau) * u[k - 1][-1] + task.ur(k * tau) * 2 * task.a ** 2
        elif approx == '2p3':
            k0 = 1 / (2 * h) / c[1]
            b[0] = (-3 / (2 * h) + a[1] * k0)
            c[0] = 2 / h + b[1] * k0
            d[0] = task.ul(k* tau) + d[1] * k0
            k1 = -(1 / (h * 2)) / a[-2]
            a[-1] = (-2 / h) + b[-2] * k1
            b[-1] = (3 / (h * 2)) + c[-2] * k1
            d[-1] = task.ur(k* tau) + d[-2] * k1

        u[k]= tridiagonal_matrix_algorithm(a, b, c, d)
    return u