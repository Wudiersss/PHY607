import numpy as np
from scipy.integrate import solve_ivp


# Right side of the two-temperature ODE
def rhs(t, y, c1, c2, k):
    t1 = y[0]
    t2 = y[1]
    q = k * (t1 - t2)
    return np.array([-q / c1, q / c2])


# Euler
def el(t, y0, c1, c2, k):
    y = np.zeros((len(t), 2))
    y[0] = y0

    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        y[i + 1] = y[i] + h * rhs(t[i], y[i], c1, c2, k)

    return y


# RK4
def rk4(t, y0, c1, c2, k):
    y = np.zeros((len(t), 2))
    y[0] = y0

    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        a = rhs(t[i], y[i], c1, c2, k)
        b = rhs(t[i] + h / 2, y[i] + h * a / 2, c1, c2, k)
        c = rhs(t[i] + h / 2, y[i] + h * b / 2, c1, c2, k)
        d = rhs(t[i] + h, y[i] + h * c, c1, c2, k)
        y[i + 1] = y[i] + h * (a + 2 * b + 2 * c + d) / 6

    return y


# SciPy
def sp(t, y0, c1, c2, k):
    sol = solve_ivp(rhs, [t[0], t[-1]], y0, t_eval=t,
                    args=(c1, c2, k))
    return sol.y.T


# Exact answer
def exact(t, y0, c1, c2, k):
    y = np.zeros((len(t), 2))
    y[:, 0] = (c1 * y0[0] + c2 * y0[1]) / (c1 + c2) + c2 * (y0[0] - y0[1]) * np.exp(-k * (1 / c1 + 1 / c2) * (t - t[0])) / (c1 + c2)
    y[:, 1] = (c1 * y0[0] + c2 * y0[1]) / (c1 + c2) - c1 * (y0[0] - y0[1]) * np.exp(-k * (1 / c1 + 1 / c2) * (t - t[0])) / (c1 + c2)
    return y
