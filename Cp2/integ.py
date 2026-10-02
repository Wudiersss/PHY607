import numpy as np
from math import erf, sqrt, pi


# Gaussian
def gauss(x, a, mu):
    return np.exp(-a * (x - mu) * (x - mu))


# Exact Gaussian integral
def gauss_exact(lo, hi, a, mu):
    return sqrt(pi) * (erf(sqrt(a) * (hi - mu)) - erf(sqrt(a) * (lo - mu))) / (2 * sqrt(a))


# Left Riemann sum
def left(f, lo, hi, n, a, mu):
    h = (hi - lo) / n
    total = 0.0

    for i in range(n):
        x = lo + i * h
        total = total + f(x, a, mu)

    return total * h



# Trapezoid method
def trap(f, lo, hi, n, a, mu):
    h = (hi - lo) / n
    total = (f(lo, a, mu) + f(hi, a, mu)) / 2

    for i in range(1, n):
        x = lo + i * h
        total = total + f(x, a, mu)

    return total * h




# Simpson method; n must be even
def simp(f, lo, hi, n, a, mu):
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson method")

    h = (hi - lo) / n
    total = f(lo, a, mu) + f(hi, a, mu)

    for i in range(1, n):
        x = lo + i * h
        if i % 2 == 0:
            total = total + 2 * f(x, a, mu)
        else:
            total = total + 4 * f(x, a, mu)

    return total * h / 3
