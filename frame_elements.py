import numpy as np


# evaluate B-Spline of order m at point(s) x
def Bspline(m, x):
    if m == 1:
        b = 1 * (x >= 0) * (x < 1)
    else:
        b = (x * Bspline(m - 1, x) + (m - x) * Bspline(m - 1, x - 1)) / (m - 1)
    return b


# evaluate frame element \psi_{j,k} at point(s) t
def eval_frame_element(t, j, k, m, alpha, epsilon):
    omega = np.sign(j) * ((1 + (1 - alpha) * abs(epsilon * j)) ** (1 / (1 - alpha)) - 1)
    beta = (1 + abs(omega)) ** (-alpha)
    x = epsilon * beta * k
    return beta ** (-1 / 2) * np.exp(2 * np.pi * 1j * omega * (t - x)) * Bspline(m, (t - x) / beta)


# evaluate a linear combination of frame elements
def evaluate_linear_combination(m, c, indexSet, t, alpha, epsilon):
    u = np.zeros(len(t))
    for n in range(len(indexSet)):
        j = indexSet[n][0]
        k = indexSet[n][1]
        if c[n] != 0:
            u = u + c[n] * eval_frame_element(t, j, k, m, alpha, epsilon)
    return u


# return the support of a frame element
def support(j, k, m, alpha, epsilon):
    omega = np.sign(j) * ((1 + (1 - alpha) * abs(epsilon * j)) ** (1 / (1 - alpha)) - 1)
    beta = (1 + abs(omega)) ** (-alpha)
    x = epsilon * beta * k
    return x, beta * m + x


# calculating the singular support of a frame element
def singsupp(j, k, m, alpha, epsilon):
    omega = np.sign(j) * ((1 + (1 - alpha) * abs(epsilon * j)) ** (1 / (1-alpha)) - 1)
    beta = (1 + abs(omega)) ** (-alpha)
    x = epsilon * beta * k
    singsupp = beta * np.arange(m+1) + x
    return singsupp


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
            number = number + 1
    return indexSet
