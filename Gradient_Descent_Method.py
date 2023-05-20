import numpy as np
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


def gradient_descent():
    iter_count = 0
    k = 0
    x = x0.copy()   #начальная точка
    x_traj = [x.copy()]
    while True:
        iter_count += 1
        grad = grad_f(x)   #пункт №3
        if np.linalg.norm(grad) < eps1:   #пункт 4
            return x, iter_count, x_traj
        if k >= M:     #пункт 5
            return x, iter_count, x_traj
        x_new = x - gamma * grad   #пункт 6 и 7
        x_traj.append(x_new.copy())
        while f(x_new) - f(x) >= 0:    #пункт 8
            gamma_new = gamma / 2
            x_new = x_traj - gamma_new * grad
            x_traj.append(x_new.copy())
        if (np.linalg.norm(x_new - x_traj) <= eps2) and (abs(f(x_new) - f(x)) <= eps2):    #пункт 9
            return x_new, iter_count, x_traj
        x = x_new
        x_traj.append(x.copy())
        k += 1


x0 = np.array([3, 3.25])   #начальная точка
eps1 = 0.0005
eps2 = 0.0005
gamma = 0.1
M = 1000
min, iters, x_traj = gradient_descent()
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
# plt.title('Gradient Descent')
# plt.show()
x = np.linspace(-1, 4, 100)
y = np.linspace(-2.5, 4, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=np.logspace(-10, 10, 100), cmap='jet')
plt.plot(*zip(*x_traj), '-o', color='black')
plt.title('Gradient Descent', fontsize=14)
plt.xlabel('x1', fontsize=12)
plt.ylabel('x2', fontsize=12)
plt.show()
