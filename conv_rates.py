import numpy as np
import matplotlib.pyplot as plt
from frame_elements import setup_indexSet, evaluate_linear_combination
from frame_operations import analysis_operator, frame_algorithm, analysis_operator_sample, setup_gramian3
from thresholding import nterm
from scipy import sparse


def f(t):
    return np.nan_to_num(np.exp(-1/t) * (t > 0), nan=0.0)

def g(t):
    return f(t) / (f(t) + f(1-t))

def bump(t, a, b):
    return 1 - g((t*t - a*a) / (b*b - a*a))

def parab(x):
    return 2 * (x - 1) ** 2 * (x >= 0) * (x < 1) + 2 * (-x - 1) ** 2 * (x < 0) * (x > -1)

def signal(x):
    a, b = 8, 10
    #return bump(t, a, b) * (np.sin(np.pi * x) + 3 * (x > -4) * (x < 3))
    #return bump(t, a, b) * (4 * x ** (-2) * (x > 1) + 4 * (x-2) ** (-2) * (x <= 1) + np.sin( np.pi * x / 2))
    return bump(x, a, b) * (parab(x) - parab(x-5) - parab(x+5) + parab(x-8) + parab(x+8) + np.sin(np.pi * (x+0.5)))
    #return bump(t, a, b) * (np.sin(np.pi * x))
    #return -np.sin(3*np.pi * x) + 10 * (2 * x * x * (x < 1 / 2) + 2 * (1 - x) ** 2 * (x >= 1 / 2)) * (x >= 0) * (x <= 1)



I = [-10, 10]  # interval
m = 3  # order of the Bspline
epsilon = 0.25

h = 0.0001
t = np.arange(I[0], I[1] + h, h)
samples = signal(t)



# --------------------- test with alpha = 0.0 ---------------------
alpha1 = 0.0
J = 60
indexSet1 = setup_indexSet(I, m, -J, J, alpha1, epsilon)

# c0 = analysis_operator(m, indexSet, f, alpha, epsilon)  # needs f in analytic form
c01 = analysis_operator_sample(m, indexSet1, samples, t, alpha1, epsilon)  # f as samples

gramian1 = setup_gramian3(m, indexSet1, alpha1, epsilon)
sparse.save_npz('gramian_j60_a00.npz', gramian1)
#gramian1 = sparse.load_npz('gramian_j60_a00.npz')


# --------------------- test with alpha = 0.5 ---------------------
alpha2 = 0.5
J = 60
indexSet2 = setup_indexSet(I, m, -J, J, alpha2, epsilon)

# c0 = analysis_operator(m, indexSet, f, alpha, epsilon)  # needs f in analytic form
c02 = analysis_operator_sample(m, indexSet2, samples, t, alpha2, epsilon)  # f as samples

gramian2 = setup_gramian3(m, indexSet2, alpha2, epsilon)
sparse.save_npz('gramian_j60_a05.npz', gramian2)
#gramian2 = sparse.load_npz('gramian_j60_a05.npz')


# run frame algorithm
relax = 0.1
iters = 5 * 10 ** 4

coeffs1 = frame_algorithm(relax, iters, c01, gramian1)
coeffs2 = frame_algorithm(relax, iters, c02, gramian2)
np.save('coeffs_m3_a00_e25_j60_sinparab5.npy', coeffs1)
np.save('coeffs_m3_a05_e25_j60_sinparab5.npy', coeffs2)

#coeffs1 = np.load('coeffs_m3_a00_e25_j60_sinspikes.npy')
#coeffs2 = np.load('coeffs_m3_a05_e25_j30_sinspikes.npy')


err1, dof1 = nterm(coeffs1, m, indexSet1, alpha1, epsilon, t, samples)
err2, dof2 = nterm(coeffs2, m, indexSet2, alpha2, epsilon, t, samples)


plt.plot(np.log(dof1), np.log(err1), label='alpha = 0')
plt.plot(np.log(dof2), np.log(err2), label='alpha = 0.5')
plt.legend()
plt.show()
