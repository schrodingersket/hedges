import numpy as np

from scipy import optimize


def subcritical_height(g, bb, q0, hL):
    """
    Computes the steady state height at set of points in the computational domain of
    subcritical Shallow Water flow over a bump.

    See Section 3.1.3 of https://doi.org/10.1002/fld.3741 for more information.

    :param g: Gravitational constant
    :param bb: Bathymetry values at computational domain points
    :param q0: Prescribed steady state flow value (steady-state solutions require constant flow
               across domain)
    :param hL: Prescribed height at x=L (right boundary)
    :return:
    """
    return np.array(list(map(lambda z: optimize.newton(
        lambda h: np.power(h, 3) + (z - np.square(q0/hL)/(2*g) - hL) * np.square(h) + np.square(q0)/(2*g),
        x0=hL,
        maxiter=500
    ), bb)))
