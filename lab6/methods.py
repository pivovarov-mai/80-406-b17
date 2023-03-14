import numpy as np
import matplotlib.pyplot as plt

class Task():
    def __init__(self, u0, u0t, ul, l, ur, r, f, a, b, c, d):
        self.u0 = u0
        self.u0t = u0t
        self.ul = ul
        self.l = l
        self.ur = ur
        self.r = r
        self.f = f
        self.a = a
        self.b = b
        self.c = c
        self.d = d

# Аналитическое решение
def analytic(l_bound, r_bound, func, end_time, t_res, h_res):
    h = (r_bound - l_bound) / h_res
    tau = end_time / t_res
    u = np.zeros(shape=(t_res, h_res))

    for t_itr in range(0, t_res):
        for x in range(0, h_res):
            u[t_itr][x] = func(l_bound + x * h, t_itr * tau)
    
    return u

# Численное вычисление первой производной
def num_der_1(f,x,h):
    return (f(x + h) - f(x - h))/(2*h) 

# Численное вычисление второй производной
def num_der_2(f,x,h):
    return (f(x - h) - 2*f(x) + f(x + h))/(h**2) 

# Инициализация первых двух слоев сетки
def init_grid(task: Task, end_time, t_res, h_res, approx = '1p2'):
    h = (task.r - task.l) / h_res
    tau = end_time / t_res
    u = np.zeros((t_res, h_res))

    for j in range(0, h_res - 1):
        x = j * h + task.l
        u[0][j] = task.u0(x)

        if approx == '1p2': # двухточечная первого поряда
            u[1][j] = u[0][j] + task.u0t(x)*tau
        elif approx == '2p2': # двухточечная второго порядка
            u[1][j] = u[0][j] + task.u0t(x)*tau + task.a*num_der_1(task.u0t, x, h) * (tau ** 2) / 2
    return u

# Явная схема   
def explicit(task: Task, end_time, t_res, h_res, init_approx='1p2', approx='c'):
    h = (task.r - task.l) / h_res
    tau = end_time / t_res
    sigma = (task.a * tau**2)/(h**2)

    if sigma > 1:
        raise ValueError(f"Sigma: {sigma}")

    u = init_grid(task, end_time, t_res, h_res, approx=init_approx)
    for k in range(t_res):
        u[k,0] = task.ul(k*tau)
        u[k,-1] = task.ur(k*tau)

    for k in range(2, t_res):
        for j in range(1, h_res - 1):
            x = task.l + j * h
            u[k][j] = task.a*(tau**2)/(h**2)*(u[k-1][j-1] - 2*u[k-1][j] + u[k-1][j+1])
            u[k][j] += task.b*(tau**2)/(2*h)*(u[k-1][j+1] - u[k-1][j-1])
            u[k][j] += task.c*(tau**2)*(u[k-1][j])
            u[k][j] += (tau**2)*task.f(x, k * tau)
            
            if approx == 'c': # центральная аппроксимация первой производной по времени
                u[k][j] += -(1/2)*(-4*u[k-1][j] + 2*u[k-2][j] - task.d*tau*u[k-2][j]) * 2/(2 + task.d*tau)
            elif approx == 't': #хвостовая аппроксимация первой производной по времени
                u[k][j] += 2*u[k-1][j] - u[k-2][j] - task.d*tau*(u[k-1][j] - u[k-2][j])

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

# Неявная схема
def implicit(task: Task, end_time, t_res, h_res, init_approx='1p2'):
    h = (task.r - task.l) / h_res
    tau = end_time / t_res
    sigma = (task.a * tau)/(h**2)

    a = np.zeros(h_res)
    b = np.zeros(h_res)
    c = np.zeros(h_res)
    d = np.zeros(h_res)
    u = init_grid(task, end_time, t_res, h_res, init_approx)
    for k in range(t_res):
        u[k,0] = task.ul(k*tau)
        u[k,-1] = task.ur(k*tau)

    for k in range(2, t_res):
        for j in range(1, h_res-1):
            a[j] = -((task.a / (h**2)) - (task.b / (2*h)))
            b[j] = (1/(tau**2) + task.d/tau + ((2*task.a) / (h**2)) - task.c)
            c[j] = -((task.a / (h**2)) + (task.b / (2*h)))
            d[j] = (2*u[k-1][j] - u[k-2][j])/(tau**2) + task.d * u[k-1][j]/tau + task.f(task.l + j * h, k * tau)

        a[0] = 0
        b[0] = 1
        c[0] = 0
        d[0] = u[k][0]
        a[-1] = 0
        b[-1] = 1
        c[-1] = 0
        d[-1] = u[k][-1]

        u[k] = tridiagonal_matrix_algorithm(a, b, c, d)

    return u