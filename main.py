import numpy as np
import matplotlib.pyplot as plt
from frame_elements import setup_indexSet, evaluate_linear_combination
from frame_operations import analysis_operator, frame_algorithm, analysis_operator_sample, setup_gramian2, setup_gramian3
from scipy import sparse
import time

start = time.time()
I = [-10, 10]
m = 3
alpha = 0.5
epsilon = 0.25
jmin = -10
jmax = 10
indexSet = setup_indexSet(I, m, jmin, jmax, alpha, epsilon)

def f(t):
    return np.nan_to_num(np.exp(-1/t) * (t > 0), nan=0.0)

def g(t):
    return f(t) / (f(t) + f(1-t))

def bump(t, a, b):
    return 1 - g((t*t - a*a) / (b*b - a*a))
def signal(x):
    a, b = 8, 10
    #return bump(t, a, b) * (np.sin(np.pi * x) + 3 * (x > -4) * (x < 3))
    #return bump(t, a, b) * (4 * x ** (-2) * (x > 1) + 4 * (x-2) ** (-2) * (x <= 1) + np.sin( np.pi * x / 2))
    return bump(t, a, b) * (2 * (x - 1) ** 2 * (x >= 0) * (x < 1) + 2 * (-x - 1) ** 2 * (x < 0) * (x > -1) + np.sin(np.pi * (x+0.5)))
    #return bump(t, a, b) * (np.sin(np.pi * x))
    #return -np.sin(3*np.pi * x) + 10 * (2 * x * x * (x < 1 / 2) + 2 * (1 - x) ** 2 * (x >= 1 / 2)) * (x >= 0) * (x <= 1)


h = 0.001
t = np.arange(I[0], I[1] + h, h)
samples = signal(t)

#c0 = analysis_operator(m, indexSet, f, alpha, epsilon)  # needs f in analytic form
c0 = analysis_operator_sample(m, indexSet, samples, t, alpha, epsilon)  # f as samples

load = 0  # 1 load pre-computed gramian, 0 calculate (and save) gramian

if load == 0:
    gramian = setup_gramian3(m, indexSet, alpha, epsilon)
    sparse.save_npz('gramian2.npz', gramian)
else:
    gramian = sparse.load_npz('gramian2.npz')

relax = 0.1
iters = 10 ** 5
coeffs = frame_algorithm(relax, iters, c0, gramian)

y = evaluate_linear_combination(m, coeffs, indexSet, t, alpha, epsilon)
plt.plot(t, y, t, samples)
plt.show()

L2error = np.sqrt(np.trapz(abs(y - samples) ** 2, t))
print('L2-error = ', L2error)

end = time.time()
print(end - start, 's')
