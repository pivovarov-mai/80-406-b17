import numpy as np
import matplotlib.pyplot as plt

class Task():
    def __init__(self, ux0, ux1, x0, x1 ,uy0, uy1, y0, y1, a):
        self.ux0 = ux0
        self.ux1 = ux1
        self.x0 = x0
        self.x1 = x1
        self.uy0 = uy0
        self.uy1 = uy1
        self.y0 = y0
        self.y1 = y1
        self.a = a

# Аналитическое решение
def analytic(x_0, x_1, y_0, y_1, func, x_res, y_res):
    x = (x_1 - x_0) / (x_res-1)
    y = (y_1 - y_0) / (y_res-1)
    u = np.zeros(shape=(y_res, x_res))

    for j in range(0, y_res):
        _y = y_0 + j * y
        for i in range(0, x_res):
            _x = x_0 + i * x
            u[j][i] = func(_x, _y)
    
    return u

# Метод простых итераций
def libman(task: Task, x_res, y_res, max_itr = 1000, eps = 0.0001):
    x = (task.x1 - task.x0) / (x_res-1)
    y = (task.y1 - task.y0) / (y_res-1)
    u = np.zeros(shape=(y_res, x_res))

    for j in range(y_res):
        _y = task.y0 + j * y
        u[j][0] = task.ux0(_y)
        u[j][-1] = task.ux1(_y)
    for i in range(x_res):
        _x = task.x0 + i * x
        u[0][i] = task.uy0(_x)
        u[-1][i] = task.uy1(_x)

    for itr in range(max_itr):
        new_u = np.copy(u)
        for i in range(1,y_res-1):
            for j in range(1,x_res-1):
                new_u[i][j] = (u[i][j-1] + u[i][j+1] + u[i+1][j] + u[i-1][j])/(4-task.a*x*y)
        
        if np.max(np.abs(new_u - u)) < eps:
            return new_u, itr
        u = new_u

    return u, max_itr 

# Метод Зейделя
def seidel(task: Task, x_res, y_res, max_itr = 1000, eps = 0.0001):
    x = (task.x1 - task.x0) / (x_res-1)
    y = (task.y1 - task.y0) / (y_res-1)
    u = np.zeros(shape=(y_res, x_res))

    for j in range(y_res):
        _y = task.y0 + j * y
        u[j][0] = task.ux0(_y)
        u[j][-1] = task.ux1(_y)
    for i in range(x_res):
        _x = task.x0 + i * x
        u[0][i] = task.uy0(_x)
        u[-1][i] = task.uy1(_x)

    for itr in range(max_itr):
            new_u = np.copy(u)
            
            for i in range(1,y_res-1):
                for j in range(1,x_res-1):
                    new_u[i,j] = (new_u[i,j-1] + new_u[i,j+1] + new_u[i+1,j] + new_u[i-1,j])/(4-task.a*x*y)

            if np.max(np.abs(new_u - u)) < eps:
                return new_u, itr
            u = new_u
            
    return u, max_itr

# Метод простых итераций с верхней релаксацией
def libman_UR(task: Task, x_res, y_res, max_itr = 1000, eps = 0.0001, w = 1.8):
    x = (task.x1 - task.x0) / (x_res-1)
    y = (task.y1 - task.y0) / (y_res-1)
    u = np.zeros(shape=(y_res, x_res))

    for j in range(y_res):
        _y = task.y0 + j * y
        u[j][0] = task.ux0(_y)
        u[j][-1] = task.ux1(_y)
    for i in range(x_res):
        _x = task.x0 + i * x
        u[0][i] = task.uy0(_x)
        u[-1][i] = task.uy1(_x)

    for itr in range(max_itr):
        new_u = np.copy(u)
        for i in range(1,y_res-1):
            for j in range(1,x_res-1):
                new_u[i][j] = (new_u[i][j-1] + new_u[i][j+1] + new_u[i+1][j] + new_u[i-1][j])/(4-task.a*x*y)
                new_u[i,j] = (1 - w)*u[i][j] + w*new_u[i][j]
        
        if np.max(np.abs(new_u - u)) < eps:
            return new_u, itr
        u = new_u        
    
    return u, max_itr 