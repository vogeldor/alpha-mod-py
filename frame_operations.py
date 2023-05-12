import math

import numpy as np
from scipy.integrate import quad
from scipy import integrate
from frame_elements import eval_frame_element, singsupp, Bspline, get_ombx, Bspline_deriv


# returns the coefficient sequence from applying the analysis operator
def analysis_operator(m, indexSet, f, alpha, epsilon):
    result = np.zeros(len(indexSet), dtype=complex)
    for n in range(len(indexSet)):
        j = indexSet[n][0]
        k = indexSet[n][1]
        omega, beta, x = get_ombx(j, k, alpha, epsilon)

        # split inner product in real and imaginary part
        def integrand1(t):
            return np.real(f(t) * eval_frame_element(t, j, k, m, alpha, epsilon))

        def integrand2(t):
            return np.imag(f(t) * eval_frame_element(t, j, k, m, alpha, epsilon))

        # integrate w.r.t. singular support
        for counter in range(m):
            result[n] += quad(integrand1, x + beta * counter, x + beta * (counter + 1))[0] - 1j * \
                         quad(integrand2, x + beta * counter, x + beta * (counter + 1))[0]
    return result


# returns the coefficient sequence from applying the analysis operator whereby f is sampled at points t
def analysis_operator_sample(m, indexSet, f, t, alpha, epsilon):
    result = np.zeros(len(indexSet), dtype=complex)
    for n in range(len(indexSet)):
        j = indexSet[n][0]
        k = indexSet[n][1]

        # split inner product in real and imaginary part
        y_real = np.real(f * eval_frame_element(t, j, k, m, alpha, epsilon))
        y_imag = np.imag(f * eval_frame_element(t, j, k, m, alpha, epsilon))
        result[n] = integrate.trapz(y_real, t) - 1j * integrate.trapz(y_imag, t)
    return result


# returns the gramian w.r.t. a given index set
def setup_gramian(m, indexSet, alpha, epsilon):
    indexsetlen = len(indexSet)
    gramian = np.zeros((indexsetlen, indexsetlen), dtype=complex)

    for r in range(indexsetlen):
        j1 = indexSet[r][0]
        k1 = indexSet[r][1]
        print(indexSet[r][2], '/', indexSet[-1][2])
        ssupp1 = singsupp(j1, k1, m, alpha, epsilon)
        for n in range(r, indexsetlen):
            j2 = indexSet[n][0]
            k2 = indexSet[n][1]
            ssupp2 = singsupp(j1, k1, m, alpha, epsilon)

            # find the boundary of the intersection
            boundleft = max(ssupp1[0], ssupp2[0])
            boundright = min(ssupp1[-1], ssupp2[-1])
            if boundleft < boundright:

                # find the singular supports contained in the intersection of psi_{j1, k1} and psi_{j2, k2}
                ssuppint = np.hstack((ssupp1, ssupp2))
                ssuppint = ssuppint[ssuppint >= boundleft]
                ssuppint = ssuppint[ssuppint <= boundright]
                ssuppint = np.sort(np.unique(ssuppint))

                # split the integrand
                def integrand1(t):
                    return np.real(eval_frame_element(t, j1, k1, m, alpha, epsilon)) * np.real(
                        eval_frame_element(t, j2, k2, m, alpha, epsilon))

                def integrand2(t):
                    return np.real(eval_frame_element(t, j1, k1, m, alpha, epsilon)) * np.imag(
                        eval_frame_element(t, j2, k2, m, alpha, epsilon))

                def integrand3(t):
                    return np.imag(eval_frame_element(t, j1, k1, m, alpha, epsilon)) * np.real(
                        eval_frame_element(t, j2, k2, m, alpha, epsilon))

                def integrand4(t):
                    return np.imag(eval_frame_element(t, j1, k1, m, alpha, epsilon)) * np.imag(
                        eval_frame_element(t, j2, k2, m, alpha, epsilon))

                # integrate and sum over all intervals
                for counter in range(len(ssuppint) - 1):
                    val1 = quad(integrand1, ssuppint[counter], ssuppint[counter + 1])[0]
                    val2 = quad(integrand2, ssuppint[counter], ssuppint[counter + 1])[0]
                    val3 = quad(integrand3, ssuppint[counter], ssuppint[counter + 1])[0]
                    val4 = quad(integrand4, ssuppint[counter], ssuppint[counter + 1])[0]
                    gramian[n][r] = gramian[n][r] + val1 - 1j * val2 + 1j * val3 + val4
                gramian[r][n] = np.conj(gramian[n][r])  # use that the gramian is hermitian
    return gramian


# returns the gramian w.r.t. a given index set - uses the structure of the inner product for faster quadrature
def setup_gramian2(m, indexSet, alpha, epsilon):
    gramian = np.zeros((len(indexSet), len(indexSet)), dtype=complex)
    for r in range(len(indexSet)):
        j1 = indexSet[r][0]
        k1 = indexSet[r][1]
        print(indexSet[r][2], '/', indexSet[-1][2])
        omega1, beta1, x1 = get_ombx(j1, k1, alpha, epsilon)
        ssupp1 = singsupp(j1, k1, m, alpha, epsilon)

        for n in range(r, len(indexSet)):
            j2 = indexSet[n][0]
            k2 = indexSet[n][1]
            omega2, beta2, x2 = get_ombx(j2, k2, alpha, epsilon)
            ssupp2 = singsupp(j1, k1, m, alpha, epsilon)

            # find the boundary of the intersection
            boundleft = max(ssupp1[0], ssupp2[0])
            boundright = min(ssupp1[-1], ssupp2[-1])
            if boundleft < boundright:

                # find the singular supports contained in the intersection of psi_{j1, k1} and psi_{j2, k2}
                ssupint = np.hstack((ssupp1, ssupp2))
                ssupint = ssupint[ssupint >= boundleft]
                ssupint = ssupint[ssupint <= boundright]
                ssupint = np.sort(np.unique(ssupint))

                # split the integrand
                def integrand1(t):
                    return np.real(
                        np.exp(2 * np.pi * 1j * t * (omega1 - omega2)) * Bspline(m, (t - x1) / beta1) * Bspline(m, (
                                    t - x2) / beta2))

                def integrand2(t):
                    return np.imag(
                        np.exp(2 * np.pi * 1j * t * (omega1 - omega2)) * Bspline(m, (t - x1) / beta1) * Bspline(m, (
                                    t - x2) / beta2))

                # integrate and sum over all intervals
                for counter in range(len(ssupint) - 1):
                    val = quad(integrand1, ssupint[counter], ssupint[counter + 1])[0] + 1j * \
                          quad(integrand2, ssupint[counter], ssupint[counter + 1])[0]
                    gramian[n][r] += (beta1 * beta2) ** (-1 / 2) * np.exp(
                        2 * np.pi * 1j * (omega2 * x2 - omega1 * x1)) * val
                gramian[r][n] = np.conj(gramian[n][r])  # use that the gramian is hermitian
    return gramian


# returns the gramian w.r.t. a given index set - exact (analytic) integration
def setup_gramian3(m, indexSet, alpha, epsilon):
    gramian = np.zeros((len(indexSet), len(indexSet)), dtype=complex)
    for r in range(len(indexSet)):
        j1 = indexSet[r][0]
        k1 = indexSet[r][1]
        print(indexSet[r][2], '/', indexSet[-1][2])
        omega1, beta1, x1 = get_ombx(j1, k1, alpha, epsilon)
        ssupp1 = singsupp(j1, k1, m, alpha, epsilon)

        for n in range(r, len(indexSet)):
            j2 = indexSet[n][0]
            k2 = indexSet[n][1]
            omega2, beta2, x2 = get_ombx(j2, k2, alpha, epsilon)
            ssupp2 = singsupp(j1, k1, m, alpha, epsilon)

            # find the boundary of the intersection
            boundleft = max(ssupp1[0], ssupp2[0])
            boundright = min(ssupp1[-1], ssupp2[-1])
            if boundleft < boundright:

                # find the singular supports contained in the intersection of psi_{j1, k1} and psi_{j2, k2}
                ssupint = np.hstack((ssupp1, ssupp2))
                ssupint = ssupint[ssupint >= boundleft]
                ssupint = ssupint[ssupint <= boundright]
                ssupint = np.sort(np.unique(ssupint))

                for counter in range(len(ssupint) - 1):  # sum over all intervals
                    left = ssupint[counter]
                    right = ssupint[counter + 1]
                    val = 0
                    if omega1 == omega2:
                        def integrand1(t):
                            return Bspline(m, (t - x1) / beta1) * Bspline(m, (t - x2) / beta2)

                        val = quad(integrand1, left, right)[0]
                    else:
                        for a in range(2 * m + 1):  # integration by parts
                            factor = (-1j / (2 * np.pi * (omega1 - omega2))) ** (a + 1)
                            valleft, valright = 0, 0
                            for k in range(a + 1):  # leibniz rule
                                eps = np.finfo(float).eps  # this solution is not ideal!
                                #eps = 0
                                valleft += math.comb(a, k) * Bspline_deriv(m, a - k, (left + eps - x1) / beta1) * Bspline_deriv(m, k, (left + eps - x2) / beta2)
                                valright += math.comb(a, k) * Bspline_deriv(m, a - k, (right - eps - x1) / beta1) * Bspline_deriv(m, k, (right - eps - x2) / beta2)
                            val += factor * (np.exp(2 * np.pi * 1j * (omega1 - omega2) * right) * valright - np.exp(2 * np.pi * 1j * (omega1 - omega2) * left) * valleft)
                    gramian[n][r] += (beta1 * beta2) ** (-1 / 2) * np.exp(2 * np.pi * 1j * (omega2 * x2 - omega1 * x1)) * val
                gramian[r][n] = np.conj(gramian[n][r])  # use that the gramian is hermitian
    return gramian


def frame_algorithm(relax, iters, c0, gramian):
    c = relax * c0
    for n in range(iters - 1):
        c = c + relax * (c0 - gramian.dot(c))
    return c
