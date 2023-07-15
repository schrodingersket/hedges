import numpy as np

import bc
import fluxes
import quadrature
import rk
import swe_1d

# Physical parameters
#
g = 1.0  # gravity

# Domain
#
tspan = (0.0, 30.0)
xspan = (-1, 1)

# Bathymetry parameters
#
b_smoothness = 0.1
b_amplitude = 0.005
b_slope = 0.05
assert(b_smoothness > 0)

# Inflow parameters
#
inflow_amplitude = 0.05
inflow_smoothness = 1.0
inflow_peak_time = 1.0
assert(inflow_amplitude > 0)

# Initial waveheight
#
h0 = 0.2
assert(h0 > 0)


def swe_bathymetry(xx):
    """
    Describes bathymetry with an upslope which is perturbed by a hyperbolic tangent function.
    """
    return b_slope * xx + b_amplitude * np.arctan(xx/b_smoothness)


def swe_bathymetry_derivative(xx):
    """
    Derivative of swe_bathymetry
    """
    return b_slope * np.ones(xx.shape) + b_amplitude * b_smoothness/(1 + np.square(b_smoothness*xx)) * np.exp(-np.square(b_smoothness*xx))


def q_bc(t):
    """
    Describes a Gaussian inflow, where the function transitions to a constant value upon reaching its maximum value.

    :param t: Time
    :return:
    """
    return inflow_amplitude * np.exp( -np.square(min(t, inflow_peak_time) - inflow_peak_time) / (2 * np.square(inflow_smoothness)) )


def initial_condition(xx):
    """
    Creates initial conditions for (h, uh).

    :param xx: Computational domain
    :return:
    """
    initial_height = h0 * np.ones(xx.shape) - swe_bathymetry(xx)  # horizontal water surface
    initial_flow = q_bc(0) * np.ones(xx.shape)  # Start with whatever flow is prescribed by our inflow BC

    ic = np.array((
        initial_height,
        initial_flow,
    ))

    # Verify consistency of initial condition
    #
    if not np.allclose(ic[1][0], q_bc(0)):
        raise ValueError('Initial flow condition must match prescribed inflow.')

    return ic


if __name__ == '__main__':
    # Instantiate solver with bathymetry
    #
    solver = swe_1d.ShallowWater1D(
        b=swe_bathymetry,
        b_x=swe_bathymetry_derivative,
        gravity=g
    )
    
    
    t_interval_ms = 20
    dt = t_interval_ms / 1000
    surface_flux = fluxes.lax_friedrichs_flux
    solution = solver.solve(
        tspan=tspan,
        xspan=xspan,
        cell_count=16,
        polydeg=4,
        initial_condition=initial_condition,
        intercell_flux=surface_flux,
        left_boundary_flux=swe_1d.ShallowWater1D.bc_prescribed_inflow(q_bc, gravity=g, surface_flux=surface_flux),
        right_boundary_flux=bc.transmissive_outflow(surface_flux=surface_flux),
        quad_rule=quadrature.gll,
        **{
            # 'method': 'RK45',
            'method': rk.SSPRK33,  # Uncomment to use a strong-stability preserving RK method
            't_eval': np.arange(tspan[0], tspan[1], dt),
            # 'max_step': dt,  # max time step for ODE solver
            'rtol': 1.0e-6,
            'atol': 1.0e-6,
        }
    )
    
    ani, plt = solver.plot_animation(solution, frame_interval=t_interval_ms)
    
    # movie_name = 'swe_1d.gif'
    # print('Writing movie to {}...'.format(movie_name))
    
    # ani.save(movie_name, progress_callback=lambda i, n: print(
    #     f'Saving animation frame {i + 1}/{n}'
    # ) if i % 25 == 0 else None)
    
    plt.show()
