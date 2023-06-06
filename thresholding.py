import numpy as np
from frame_elements import evaluate_linear_combination


# hard thresholding
def thresholding(coeffs, mu):
    c_mu = np.array(coeffs)
    c_mu[abs(c_mu) < mu] = 0
    return c_mu


# n-term approximation
def nterm(coeffs, narray, m, indexSet, alpha, epsilon, t, samples):
    coeffs_sorted = sorted(coeffs, key=abs, reverse=True)
    L2error, dof = np.zeros(len(narray)), np.zeros(len(narray))

    k = 0
    for n in narray:
        print(k, '/', len(narray))
        c_mu = thresholding(coeffs, abs(coeffs_sorted[n-1]))
        y = evaluate_linear_combination(m, c_mu, indexSet, t, alpha, epsilon)
        L2error[k] = np.sqrt(np.trapz(abs(y - samples) ** 2, t))
        #L2error[n] = (np.linalg.norm(y - samples) ** 2) / len(t)
        dof[k] = np.count_nonzero(c_mu)
        k += 1
    return L2error, dof
