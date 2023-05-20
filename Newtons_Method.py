import numpy as np
import torch
from torch.autograd.functional import hessian
import matplotlib.pyplot as plt


call_count_f = 0
call_count_grad = 0


def f(x):
    global call_count_f
    call_count_f += 1
    return 4 * pow(x[0], 4) - 6 * x[0] * x[1] - 34 * x[0] + 5 * pow(x[1], 4) + 42 * x[1] + 7


def grad_f(x):
    global call_count_grad
    call_count_grad += 1
    return np.array([16 * pow(x[0], 3) - 6 * x[1] - 34, - 6 * x[0] + 20 * pow(x[1], 3) + 42])


def newton_method():
    iter_count = 0
    # Шаг 1
    k = 0
    x = x0
    x_traj = [x.copy()]
    grad = grad_f(x)
    hess = hessian(f, torch.tensor(x))
    while np.linalg.norm(grad) >= eps1 and k < M:
        iter_count += 1
        # Шаг 3
        grad = grad_f(x)
        # Шаг 4
        if np.linalg.norm(grad) < eps1:
            return x, iter_count, x_traj
        # Шаг 5
        if k >= M:
            return x, iter_count, x_traj
        # Шаг 6
        hess1 = hessian(f, torch.tensor(x))
        # Шаг 7
        hess_inv = np.linalg.inv(hess1)
        # Шаг 8
        if hess_inv.all() > 0:
            delta_x = -hess_inv.sum() * grad_f(x)
            x_traj.append(delta_x.copy())
        else:
            delta_x = -grad_f(x)
            x_traj.append(delta_x.copy())
        # Шаг 9
        alpha = 1.0
        while f(x + alpha * delta_x) >= f(x):
            alpha /= 2.0
        # Шаг 10
        x_new = x + alpha * delta_x
        x_traj.append(x_new.copy())
        # x = x_new
        # Шаг 11
        if (np.linalg.norm(x_new - x) <= eps2) and (abs(f(x_new) - f(x)) <= eps2):    #пункт 9
            return x_new, iter_count, x_traj
        k += 1
    # Шаг 4.1, 5.1
    return x, iter_count, x_traj


eps1 = 0.0005
eps2 = 0.0005
M = 1000
x0 = np.array([1.0, 2.0])
min, iters, x_traj = newton_method()
print("Число итераций: ", iters)
print("Количество вычислений функции: ", call_count_f)
print("Количество вычислений градиента функции: ", call_count_grad)
print("Найденное решение (min): ", min)
print("Значение функции: ", f(min))

print("Траектория движения к экстремуму", x_traj)
# x_traj = np.array(x_traj)
# plt.figure(figsize=(6, 6))
# plt.plot(x_traj[:, 0], x_traj[:, 1], '-o')
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.title('Newton Method')
# plt.show()
# Визуализация
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=np.logspace(-10, 10, 100), cmap='jet')
plt.plot(*zip(*x_traj), '-o', color='black')
plt.title('Newton Method', fontsize=14)
plt.xlabel('x1', fontsize=12)
plt.ylabel('x2', fontsize=12)
plt.show()
