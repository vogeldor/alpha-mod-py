import numpy as np


# evaluate B-Spline of order m at point(s) x
def Bspline(m, x):
    if m == 1:
        b = 1 * (x >= 0) * (x < 1)
    else:
        b = (x * Bspline(m - 1, x) + (m - x) * Bspline(m - 1, x - 1)) / (m - 1)
    return b


def Bspline_deriv2(m, n, x):
    if n >= m:
        return 0
    elif m == 1:
        return 1 * (x >= 0) * (x <= 1)
    else:
        return Bspline_deriv2(m-1, n-1, x) - Bspline_deriv2(m-1, n-1, x-1)


# evaluates the n-th derivative of the B-Spline of order m at point(s) x
# dir is only relevant for the derivate of m = 2. 0 means t -> x from above. 1 means t -> x from below.
def Bspline_deriv(m, n, x, dir):
    tol = 1e-12
    if n >= m:
        b = 0
    elif n == 0:
        if m == 1:
            if -tol < x < 1 - tol and dir == 0:
                return 1
            elif tol < x < 1 + tol and dir == 1:
                return 1
            else:
                return 0
        else:
            b = Bspline(m, x)
    elif n == m-1 == 1:
        #if (abs(x-1) < tol) and dir == 0:
        #    return -1
        #elif (abs(x-1) < tol) and dir == 1:
        #    return 1
        #elif -tol <= x < 0 and dir == 0:
        #    return 1
        #elif 2+tol >= x > 2 and dir == 1:
        #    return -1
        #else:
        #    return 1 * (x >= 0) * (x < 1) - 1 * (x > 1) * (x <= 2)
        return Bspline_deriv(m-1, n-1, x, dir) - Bspline_deriv(m-1, n-1, x-1, dir)
    else:
        b = Bspline_deriv(m-1, n-1, x, dir) - Bspline_deriv(m-1, n-1, x-1, dir)
    return b


# returns omega, beta and x for a given index (j,k)
def get_ombx(j, k, alpha, epsilon):
    omega = np.sign(j) * ((1 + (1 - alpha) * abs(epsilon * j)) ** (1 / (1 - alpha)) - 1)
    beta = (1 + abs(omega)) ** (-alpha)
    x = epsilon * beta * k
    return omega, beta, x


# evaluate frame element \psi_{j,k} at point(s) t
def eval_frame_element(t, j, k, m, alpha, epsilon):
    omega, beta, x = get_ombx(j, k, alpha, epsilon)
    return beta ** (-1 / 2) * np.exp(2 * np.pi * 1j * omega * (t - x)) * Bspline(m, (t - x) / beta)


# evaluate a linear combination of frame elements
def evaluate_linear_combination(m, c, indexSet, t, alpha, epsilon):
    y = np.zeros(len(t), dtype=complex)
    for n in range(len(indexSet)):
        if c[n] != 0:
            y += c[n] * eval_frame_element(t, indexSet[n][0], indexSet[n][1], m, alpha, epsilon)
    return y


# return the support of a frame element
def support(j, k, m, alpha, epsilon):
    omega, beta, x = get_ombx(j, k, alpha, epsilon)
    return x, beta * m + x


# calculating the singular support of a frame element
def singsupp(j, k, m, alpha, epsilon):
    omega, beta, x = get_ombx(j, k, alpha, epsilon)
    return beta * np.arange(m+1) + x


# create a matrix that represents the finite index set \Lambda \subset \mathbb{Z} x \mathbb{Z}.
# each row represents one index of the form [j, k, number]. number is for convenience.
def setup_indexSet(I, m, jmin, jmax, alpha, epsilon):
    a, b = I[0], I[1]
    indexSet = []
    number = 1
    for j in range(jmin, jmax+1):
        omega = np.sign(j) * ((1 + (1 - alpha) * abs(epsilon * j)) ** (1 / (1 - alpha)) - 1)
        beta = (1 + abs(omega)) ** (-alpha)
        kmin = int(np.floor(1 / epsilon * (1 / beta * a - m)) + 1)
        kmax = int(np.ceil(1 / (epsilon * beta) * b) - 1)
        for k in range(kmin, kmax+1):
            indexSet.append([j, k, number])
            number += 1
    return indexSet
