import numpy as np
import matplotlib.pyplot as plt
from frame_elements import setup_indexSet, evaluate_linear_combination
from frame_operations import analysis_operator, setup_gramian, frame_algorithm, analysis_operator_sample, setup_gramian2
from thresholding import nterm
import time

start = time.time()
I = [-10, 10]  # interval
m = 2  # order of the Bspline
alpha = 0.5
epsilon = 0.25
jmin = -1
jmax = 1
indexSet = setup_indexSet(I, m, jmin, jmax, alpha, epsilon)


def f(x):
    #return np.sin(np.pi * x)
    return np.sin(np.pi * x) + 10 * (2 * x * x * (x < 1/2) + 2 * (1-x) ** 2 * (x >= 1/2)) * (x >= 0) * (x <= 1)


h = 0.001
t = np.arange(I[0], I[1] + h, h)
samples = f(t)

c0 = analysis_operator(m, indexSet, f, alpha, epsilon)  # needs f in analytic form
#c0 = analysis_operator_sample(m, indexSet, samples, t, alpha, epsilon)  # f as samples

load = 0  # 1 load pre-computed gramian, 0 calculate (and save) gramian
if load == 0:
    gramian = setup_gramian2(m, indexSet, alpha, epsilon) # setup_gramian or setup_gramian2
    np.save('gramian.npy', gramian)
else:
    gramian = np.load('gramian.npy')

relax = 0.1
iters = 10 ** 5
coeffs = frame_algorithm(relax, iters, c0, gramian)

y = evaluate_linear_combination(m, coeffs, indexSet, t, alpha, epsilon)
plt.plot(t, y, t, f(t))
plt.show()
L2error = np.sqrt(np.trapz(abs(y - f(t)) ** 2, t))
print('L2-error = ', L2error)

#nsteps = 20  # stepsize for the n-term approximation
#err, dof = nterm(coeffs, nsteps, m, indexSet, alpha, epsilon, t, samples)
#plt.plot(np.log(dof), np.log(err))
#plt.show()

end = time.time()
print(end - start, 's')
