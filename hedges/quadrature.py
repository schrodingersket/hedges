import numpy as np

from scipy.interpolate import lagrange


def _generate_lagrange_basis(x):
    """
    Generate a Lagrange basis consisting of characteristic polynomials of degree (len(x) - 1) for
    each collocation point. The j-th basis polynomial is computed by determining the interpolating
    polynomial for the points (x_i, delta_ij) where i represents the index of the collocation nodes
    and delta_ij is the Kronecker Delta function.

    :param x:
    :return:
    """
    basis = []
    for i in range(len(x)):
        samples = np.zeros(len(x))
        samples[i] = 1  # set to 1 to obtain a characteristic polynomial for the Lagrange basis
        basis.append(lagrange(x, samples))

    return basis


def lagrange_differentiation_matrix(x):
    """
    Computes the differentiation matrix for a Lagrange interpolating polynomial.

    :param x:
    :return:
    """
    basis = _generate_lagrange_basis(x)
    d = np.empty((len(x), len(x)))

    for j, b, in enumerate(basis):
        # b represents the j-th Lagrange basis function. We set the j-th column of the
        # differentiation matrix d to the derivative of this function evaluated at the provided
        # nodes.
        #
        d[:, j] = b.deriv()(x)

    return d


def gl(n, *args):
    """
    TODO: Figure out what changes need to be made to the hyperbolic solver for these points to work.
          You will be disappointed if you try to use these points in their current state (the points
          and weights are accurate).

    Computes N Gauss-Legendre nodes (intervals) on the interval [-1, 1] by using Numpy's built-in
    libraries for finding roots of Legendre polynomials.
    """
    # Get n-th Legendre polynomial basis function
    #
    legendre_basis_n = np.polynomial.Legendre.basis(n)

    # Find zeros of derivative of Legendre polynomials
    #
    x = legendre_basis_n.roots()

    # Compute weights
    #
    w = 2.0 / ((1 - np.square(x)) * (legendre_basis_n.deriv()(x))**2)

    return x, w


def gll(n, *args):
    """
    Computes N Gauss-Legendre-Lobatto nodes (intervals) on the interval [-1, 1] by using Numpy's
    built-in libraries for finding roots of differentiated Legendre polynomials.
    """
    # Get n-th Legendre polynomial basis function
    #
    legendre_basis_n = np.polynomial.Legendre.basis(n)

    # Find zeros of derivative of Legendre polynomials
    #
    x = legendre_basis_n.deriv().roots()
    x = np.insert(x, 0, -1.0)
    x = np.append(x, 1.0)

    # Compute weights
    #
    w = 2.0 / ((n*(n + 1)) * (legendre_basis_n(x)**2))

    return x, w


def _gll_newton_raphson(n, tol=10e-8):
    """
    Computes N Gauss-Legendre-Lobatto nodes (intervals) on the interval [-1, 1] by using the
    Newton-Raphson method to compute zeros of (1 - x^2)P_N'(x), where P_N' denotes the first
    derivative of the N-th degree Legendre polynomial.

    To use the Newton-Raphson method, we need d/dt [(1 - x^2)P_N'(x)].

    This function takes 1-2 orders of magnitude longer than the Numpy-based gll computation, but
    this function is included for reference as a secondary way to compute these points.

    See https://doi.org/10.1007/978-88-470-5522-3_10
    """

    # Initial guess
    #
    x = np.polynomial.chebyshev.chebpts2(n+1)

    # Legendre polynomials
    #
    legendre_bases = np.array((
        np.polynomial.Legendre.basis(n),
        np.polynomial.Legendre.basis(n-1),
    ))

    # Generate Legendre polynomials
    #
    x_prev = x
    for i in range(100):
        x = x + (legendre_bases[-1](x) - x * legendre_bases[0](x)) / ((n + 1) * legendre_bases[0](x))

        if np.linalg.norm(x - x_prev, np.inf) < tol:
            break

    # Compute weights
    #
    w = 2.0 / ((n*(n + 1)) * (legendre_bases[0](x)**2))

    return x, w


if __name__ == '__main__':
    """
    We add some basic testing here in order to verify agreement between the two GLL methods. This
    also provides a nice sanity check to compute GLL nodes/weights.
    
    """
    import time
    np.set_printoptions(suppress=True)
    print('Gauss-Legendre-Lobatto nodes/weights for 3rd-degree Legendre polynomial:')

    nodes, weights = gll(3)
    differentiation_matrix = lagrange_differentiation_matrix(nodes)

    print('x = {}'.format(nodes))
    print('w = {}'.format(weights))
    print('D = {}'.format(differentiation_matrix))

    # Compute GLL via Newton-Raphson
    #
    test_node_count = 100
    start = time.time()
    x_nr, w_nr = _gll_newton_raphson(test_node_count, 10e-8)
    end = time.time()
    print('Newton-Raphson took {} seconds for {} nodes'.format(end - start, test_node_count))
    start = time.time()

    # Compute GLL via Numpy
    #
    x_np, w_np = gll(test_node_count)
    end = time.time()
    print('Numpy differentiation took {} seconds for {} nodes'.format(end - start, test_node_count))

    # Assert equivalence
    #
    assert(np.all(np.isclose(x_nr, x_np)))
    assert(np.all(np.isclose(w_nr, w_np)))
