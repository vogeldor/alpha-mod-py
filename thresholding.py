import numpy as np
from frame_elements import evaluate_linear_combination


# hard thresholding
def thresholding(coeffs, mu):
    c_mu = coeffs
    c_mu[abs(c_mu) < mu] = 0
    return c_mu


# n-term approximation
def nterm(coeffs, nsteps, m, indexSet, alpha, epsilon, t, samples):
    coeffs_sorted = np.sort(abs(coeffs))
    L2error = np.zeros(int(len(coeffs) / nsteps))
    dof = np.zeros(int(len(coeffs) / nsteps))

    for n in range(int(len(coeffs) / nsteps)):
        c_mu = thresholding(coeffs, coeffs_sorted[n * nsteps])
        y = evaluate_linear_combination(m, c_mu, indexSet, t, alpha, epsilon)
        L2error[n] = np.sqrt(np.trapz(abs(y - samples) ** 2, t))
        #L2error[n] = (np.linalg.norm(y - samples) ** 2) / len(t)
        dof[n] = np.count_nonzero(c_mu)
    return L2error, dof
