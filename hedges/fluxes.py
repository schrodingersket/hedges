import numpy as np


def lax_friedrichs_flux(u_l, u_r, xx, t, f, f_prime):
    """
    See http://www.clawpack.org/riemann_book/html/Approximate_solvers.html#Lax-Friedrichs-(LF)-and-local-Lax-Friedrichs-(LLF)

    :param u_l: Left state
    :param u_r: Right state
    :param xx: Domain
    :param t: Current time
    :param f: Hyperbolic flux function
    :param f_prime: Jacobian of hyperbolic flux function
    :return:
    """
    eig_left = np.abs(np.linalg.eigvalsh(f_prime(u_l, xx, t))).max()
    eig_right = np.abs(np.linalg.eigvalsh(f_prime(u_r, xx, t))).max()

    alpha = max(eig_left, eig_right)

    return 0.5 * (f(u_r, xx, t) + f(u_l, xx, t) + alpha * (u_l - u_r))