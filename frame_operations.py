import math
import numpy as np
from scipy.integrate import quad
from scipy import integrate
from frame_elements import eval_frame_element, singsupp, Bspline, get_ombx, Bspline_deriv
from scipy import sparse


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
        if np.mod(n, 1000) == 0:
            print(n, '/', indexSet[-1][2])

        j = indexSet[n][0]
        k = indexSet[n][1]

        # split inner product in real and imaginary part
        y_real = np.real(f * eval_frame_element(t, j, k, m, alpha, epsilon))
        y_imag = np.imag(f * eval_frame_element(t, j, k, m, alpha, epsilon))
        result[n] = integrate.trapz(y_real, t) - 1j * integrate.trapz(y_imag, t)
    return result


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
            ssupp2 = singsupp(j2, k2, m, alpha, epsilon)

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
    gramian = sparse.lil_matrix((len(indexSet), len(indexSet)), dtype=complex)
    tol = 1e-10
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
            ssupp2 = singsupp(j2, k2, m, alpha, epsilon)

            # find the boundary of the intersection
            boundleft = max(ssupp1[0], ssupp2[0])
            boundright = min(ssupp1[-1], ssupp2[-1])
            if boundleft < boundright:

                # find the singular supports contained in the intersection of psi_{j1, k1} and psi_{j2, k2}
                ssupint = np.hstack((ssupp1, ssupp2))
                ssupint = ssupint[ssupint >= boundleft]
                ssupint = ssupint[ssupint <= boundright]
                ssupint = np.sort(ssupint[~(np.triu(np.abs(ssupint[:, None] - ssupint) <= tol, 1)).any(0)])

                val = 0
                if omega1 == omega2:
                    def integrand(t):
                        return Bspline(m, (t - x1) / beta1) * Bspline(m, (t - x2) / beta2)

                    for counter in range(len(ssupint) - 1):
                        val += quad(integrand, ssupint[counter], ssupint[counter + 1])[0]

                else:
                    val = 0
                    for counter in range(len(ssupint) - 1):
                        for k in range(2 * m - 1):
                            left, right = 0, 0
                            t1, t2 = ssupint[counter], ssupint[counter+1]
                            for ell in range(k+1):
                                fac = math.comb(k, ell) * beta1 ** (-(k - ell)) * beta2 ** (- ell)
                                left += fac * Bspline_deriv(m, k - ell, t1/beta1 - epsilon * k1, 0) * Bspline_deriv(m, ell, t1/beta2 - epsilon * k2, 0)
                                right += fac * Bspline_deriv(m, k - ell, t2/beta1 - epsilon * k1, 1) * Bspline_deriv(m, ell, t2/beta2 - epsilon * k2, 1)
                            left *= np.exp(2 * np.pi * 1j * t1 * (omega1 - omega2))
                            right *= np.exp(2 * np.pi * 1j * t2 * (omega1 - omega2))
                            val += (-1) ** k * (right - left) * (2 * np.pi * 1j * (omega1 - omega2)) ** (-(k + 1))

                gramian[n, r] = val * (beta1 * beta2) ** (-1 / 2) * np.exp(2 * np.pi * 1j * (-omega1 * x1 + omega2 * x2))
                gramian[r, n] = np.conj(gramian[n, r])  # use that the gramian is hermitian
    return sparse.csr_matrix(gramian)


def frame_algorithm(relax, iters, c0, gramian):
    c = relax * c0
    for n in range(iters - 1):
        step = c0 - gramian.dot(c)
        c = c + relax * step
        print(n, np.linalg.norm(step))
    return c
