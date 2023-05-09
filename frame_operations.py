import numpy as np
from scipy.integrate import quad
from scipy import integrate
from frame_elements import eval_frame_element, singsupp


# returns the coefficient sequence from applying the analysis operator
def analysis_operator(m, indexSet, f, alpha, epsilon):
    indexsetlen = len(indexSet)
    result = np.zeros(indexsetlen, dtype=complex)

    for n in range(indexsetlen):
        j = indexSet[n][0]
        k = indexSet[n][1]
        omega = np.sign(j) * ((1 + (1 - alpha) * abs(epsilon * j)) ** (1 / (1 - alpha)) - 1)
        beta = (1 + abs(omega)) ** (-alpha)
        x = epsilon * beta * k

        # split inner product in real and imaginary part
        def integrand1(t):
            return np.real(f(t) * eval_frame_element(t, j, k, m, alpha, epsilon))

        def integrand2(t):
            return np.imag(f(t) * eval_frame_element(t, j, k, m, alpha, epsilon))

        # integrate w.r.t. singular support
        for counter in range(m):
            val = quad(integrand1, x + beta * counter, x + beta * (counter + 1))[0] - 1j * \
                  quad(integrand2, x + beta * counter, x + beta * (counter + 1))[0]
            result[n] = result[n] + val
    return result


# returns the coefficient sequence from applying the analysis operator whereby f is sampled at points t
def analysis_operator_sample(m, indexSet, f, t, alpha, epsilon):
    indexsetlen = len(indexSet)
    result = np.zeros(indexsetlen, dtype=complex)

    for n in range(indexsetlen):
        j = indexSet[n][0]
        k = indexSet[n][1]
        omega = np.sign(j) * ((1 + (1 - alpha) * abs(epsilon * j)) ** (1 / (1 - alpha)) - 1)
        beta = (1 + abs(omega)) ** (-alpha)
        x = epsilon * beta * k

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
        singsupp1 = singsupp(j1, k1, m, alpha, epsilon)
        for n in range(r, indexsetlen):
            j2 = indexSet[n][0]
            k2 = indexSet[n][1]
            singsupp2 = singsupp(j1, k1, m, alpha, epsilon)

            # find the boundary of the intersection
            boundleft = max(singsupp1[0], singsupp2[0])
            boundright = min(singsupp1[-1], singsupp2[-1])
            if boundleft < boundright:

                # find the singular supports contained in the intersection of psi_{j1, k1} and psi_{j2, k2}
                singsupp12 = np.hstack((singsupp1, singsupp2))
                singsupp12 = singsupp12[singsupp12 >= boundleft]
                singsupp12 = singsupp12[singsupp12 <= boundright]
                singsupp12 = np.sort(np.unique(singsupp12))

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
                for counter in range(len(singsupp12) - 1):
                    val1 = quad(integrand1, singsupp12[counter], singsupp12[counter + 1])[0]
                    val2 = quad(integrand2, singsupp12[counter], singsupp12[counter + 1])[0]
                    val3 = quad(integrand3, singsupp12[counter], singsupp12[counter + 1])[0]
                    val4 = quad(integrand4, singsupp12[counter], singsupp12[counter + 1])[0]
                    gramian[n][r] = gramian[n][r] + val1 - 1j * val2 + 1j * val3 + val4
                gramian[r][n] = np.conj(gramian[n][r])  # use that the gramian is hermitian
    return gramian


def frame_algorithm(relax, iters, c0, gramian):
    c = relax * c0
    for n in range(iters - 1):
        c = c + relax * (c0 - gramian.dot(c))
    return c
