import numpy as np
from numpy import argmin
import matplotlib.pyplot as plt


call_count_f = 0
call_count_grad = 0


def f(x):
    global call_count_f
    call_count_f += 1
    return pow(x[0], 3) + pow(x[1], 2) - 3*x[0] - 2*x[1] + 2


def grad_f(x):
    global call_count_grad
    call_count_grad += 1
    return np.array([3 * pow(x[0], 2) - 3, 2 * x[1] - 2])


def line_search(f, grad_f, x, delta_x):
    alpha = 1.0
    rho = 0.5
    c = 0.1
    phi_0 = f(x)
    dphi_0 = np.dot(grad_f(x), delta_x)
    while True:
        phi_alpha = f(x + alpha * delta_x)
        if phi_alpha > phi_0 + c * alpha * dphi_0:
            alpha *= rho
        else:
            dphi_alpha = np.dot(grad_f(x + alpha * delta_x), delta_x)
            if dphi_alpha < c * dphi_0:
                alpha /= rho
            else:
                break
    return alpha


def ravine():
    iter_count = 0
    x_prev = x0.copy()
    x_curr = x1.copy()
    x_traj = [x_curr.copy()]
    k = 1
    while True:
        iter_count += 1
        # шаг 2
        grad1 = grad_f(x_curr)
        # gamma_star = argmin(f(x_curr - gamma * grad1))
        gamma_star = line_search(f, grad_f, x_curr, grad1)
        x_next = x_curr - gamma_star * grad1
        x_traj.append(x_next.copy())
        # шаг 4
        sgn = np.sign(f(x_next) - f(x_curr))
        x_ovr = x_curr - (x_curr - x_prev) / np.linalg.norm(x_curr - x_prev) * h * sgn
        x_traj.append(x_ovr.copy())
        # шаг 5
        # gamma_star1 = argmin(f(x_ovr - gamma * grad_f(x_ovr)))
        gamma_star1 = line_search(f, grad_f, x_ovr, grad_f(x_ovr))
        x_new = x_ovr - gamma_star1 * grad_f(x_ovr)
        x_traj.append(x_new.copy())
        # шаг 6
        if np.linalg.norm(x_new - x_curr) < eps:
            x_star = x_new
            x_traj.append(x_star.copy())
            break
        else:
            k += 1
            x_prev = x_curr
            x_curr = x_new
            x_traj.append(x_curr.copy())
    return x_star, iter_count, x_traj


x0 = np.array([1.25, 1.0])
x1 = np.array([1.0, 1.0])
h = 0.1
eps = 0.0005
gamma = 0.1

min, iters, x_traj = ravine()
print("Число итераций: ", iters)
print("Количество вычислений функции: ", call_count_f)
print("Количество вычислений градиента функции: ", call_count_grad)
print("Найденное решение (min): ", min)
print("Значение функции: ", f(min))
print("Траектория движения к экстремуму", x_traj)
# Визуализация
x = np.linspace(-2, 4, 100)
y = np.linspace(-3, 4, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=np.logspace(-10, 10, 100), cmap='jet')
plt.plot(*zip(*x_traj), '-o', color='black')
plt.title('Ravine Method', fontsize=14)
plt.xlabel('x1', fontsize=12)
plt.ylabel('x2', fontsize=12)
plt.show()