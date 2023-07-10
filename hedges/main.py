import numpy as np
from scipy import optimize

import swe_1d
import fluxes
import quadrature
import rk

# Physical parameters
#
g = 1.0  # gravity
inflow_rate = 0.1
outflow_resistance_coefficient = 0.0  # outflow resistance (1 for reflection, 0 for free flow)

tspan = (0.0, 2.0)
xspan = (-1.0, 1.0)


def q_bc(t):
    """
    Prescribed inflow rate as a function of time

    :param t:
    :return:
    """
    return inflow_rate * t if t < 1 else inflow_rate  # Constant inflow


def initial_condition(xx):
    """
    Creates initial conditions for (h, uh).

    :param xx:
    :return:
    """
    initial_height = 0.5 * np.ones(xx.shape) - swe_bathymetry(xx)  # water at rest
    initial_flow = np.zeros(xx.shape)

    ic = np.array((
        initial_height,
        initial_flow,
    ))

    # Verify consistency of initial condition
    #
    if not np.allclose(ic[1][0], q_bc(0)):
        raise ValueError('Initial flow condition must match prescribed inflow.')

    return ic


# Prescribed inflow flux (derived by equating backward characteristics at the leftmost interface)
#
def prescribed_inflow(q_in):
    """
    Maintains a (possibly time-dependent) prescribed rate of flow at the inflow boundary.

    :param q_in: A function with time as its single parameter, which should return the rate of flow
                 at a particular time.
    :return:
    """
    def _flux_function(u_l, u_r, xx, t, f, f_prime):
        h_r, q_r = u_r

        # Backward characteristic
        #
        w_b = q_r / h_r - 2 * np.sqrt(g * h_r)

        # Evaluate prescribed bathymetry value at t
        #
        q = q_in(t)

        # We solve for the square root of h to avoid numerical issues with Newton's method
        #
        sqrt_h = optimize.newton(
            lambda hh: q / np.square(hh) - 2 * np.sqrt(g) * hh - w_b,
            x0=np.square(h_r),
            fprime=lambda hh: -2 * q / hh ** 3 - 2 * np.sqrt(g),
            maxiter=500
        )

        h = np.square(sqrt_h)

        return fluxes.lax_friedrichs_flux(np.array((h, q)), u_r, xx, t, f, f_prime)

    return _flux_function


def resistive_outflow(resistance_coefficient):
    def _flux_function(u_l, u_r, xx, t, f, f_prime):
        h_l, q_l = u_l

        # Forward characteristic
        #
        w_f = q_l / h_l + 2 * np.sqrt(g * h_l)

        # Compute "ghost cell" values
        #
        h_r = h_l
        q_r = h_l * w_f * (1 - resistance_coefficient) - q_l

        # Backward characteristic
        #
        w_b = q_r/h_r - 2 * np.sqrt(g * h_r)

        # Upwinded values
        #
        # TODO: Figure out why these don't work when local LFF with ghost cell values does
        #
        h = (w_f - w_b)**2 / (2*g)
        q = h/2 * (w_f + w_b)

        return fluxes.lax_friedrichs_flux(u_l, u_r, xx, t, f, f_prime)

    return _flux_function


eps = 20


def swe_bathymetry(xx):
    return -0.05 * xx + 0.1 * np.exp(-np.square(eps*xx))


def swe_bathymetry_derivative(xx):
    return -0.05 * np.ones(xx.shape) + 0.1 * -2 * np.square(eps) * xx * np.exp(-np.square(eps*xx))


# Instantiate solver with bathymetry
#
solver = swe_1d.ShallowWater1D(
    b=swe_bathymetry,
    b_x=swe_bathymetry_derivative,
    g=g
)


t_interval_ms = 20
dt = t_interval_ms / 1000
solution = solver.solve(
    tspan=tspan,
    xspan=xspan,
    cell_count=16,
    polydeg=5,
    initial_condition=initial_condition,
    intercell_flux=fluxes.lax_friedrichs_flux,
    left_boundary_flux=prescribed_inflow(q_bc),
    right_boundary_flux=resistive_outflow(outflow_resistance_coefficient),
    quad_rule=quadrature.gll,
    **{
        'method': 'RK45',
        # 'method': rk.SSPRK33,  # Uncomment to use a strong-stability preserving RK method
        't_eval': np.arange(tspan[0], tspan[1], dt),
        'max_step': dt,  # max time step for ODE solver
        'rtol': 1.0e-6,
        'atol': 1.0e-6,

    }
)

ani, plt = solver.plot_animation(solution, frame_interval=t_interval_ms)

movie_name = 'swe_1d.gif'
print('Writing movie to {}...'.format(movie_name))

ani.save(movie_name, progress_callback=lambda i, n: print(
    f'Saving animation frame {i + 1}/{n}'
) if i % 25 == 0 else None)

plt.show()
