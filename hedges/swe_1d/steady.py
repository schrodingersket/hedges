import numpy as np
import scipy.optimize


def subcritical_height(gravity, b, q_l, h_r, b_r=0):
    """
    Computes the steady state height at set of points in the computational domain of
    subcritical Shallow Water flow over a bump.

    See Section 3.1.3 of https://doi.org/10.1002/fld.3741 for more information.

    :param gravity: Gravitational constant
    :param b: Array of bathymetry values at computational domain points
    :param q_l: Prescribed steady state flow at x = x_l (left boundary)
    :param h_r: Prescribed height at x = x_r (right boundary)
    :param b_r: Bathymetry height at x = x_r (right boundary)
    :return:
    """
    c = (b - (b_r + h_r) - np.square(q_l / h_r) / (2 * gravity))
    k = np.square(q_l) / (2 * gravity)

    return np.array([scipy.optimize.newton(
        lambda h: np.power(h, 3) + c[j] * np.square(h) + k,
        x0=h_r,
        maxiter=500
    ) for j, _ in enumerate(b)])
