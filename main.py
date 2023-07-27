#!/usr/bin/env python
# coding: utf-8

# # 1D Discontinuous Galerkin Shallow Water Solver
#
# We solve the 1D Shallow Water Equations in conservative form:
#
# \begin{align*}
#     h_t + q_x &= 0 \\
#     q_t + \left[ \frac{q^2}{h} + \frac{1}{2} g h^2 \right]_x &= -g h b_x - C_f \left(\frac{q}{h}\right)^2
# \end{align*}
#
# We neglect friction so that $C_f = 0$.
#

import numpy as np
import matplotlib.pyplot as plt


import hedges.bc as bc
import hedges.fluxes as fluxes
import hedges.quadrature as quadrature
import hedges.rk as rk
import hedges.swe_1d.pde as pde


# Physical parameters
#
g = 1.0  # gravity

# Domain
#
tspan = (0.0, 4.0)
xspan = (-1, 1)

# Bathymetry parameters
#
b_smoothness = 0.1
b_amplitude = 0.02
b_slope = 0.05
assert(b_smoothness > 0)

# Inflow parameters
#
inflow_amplitude = 0.05
inflow_smoothness = 1.0
inflow_peak_time = 2.0
assert(inflow_amplitude > 0)

# Initial waveheight
#
h0 = 0.2
assert(h0 > 0)


def swe_bathymetry(x):
    """
    Describes bathymetry with an upslope which is perturbed by a hyperbolic tangent function.
    """
    return b_slope * x + b_amplitude * np.arctan(x / b_smoothness)


def swe_bathymetry_derivative(x):
    """
    Derivative of swe_bathymetry
    """
    return b_slope + b_amplitude / (
            b_smoothness * (1 + np.square(x / b_smoothness))
    )


def q_bc(t):
    """
    Describes a Gaussian inflow, where the function transitions to a constant value upon attaining
    its maximum value.

    :param t: Time
    :return:
    """
    t_np = np.array(t)
    return inflow_amplitude * np.exp(
        -np.square(
            np.minimum(t_np, inflow_peak_time * np.ones(t_np.shape)) - inflow_peak_time
        ) / (2 * np.square(inflow_smoothness))
    )


def initial_condition(x):
    """
    Creates initial conditions for (h, uh).

    :param x: Computational domain
    :return:
    """
    initial_height = h0 * np.ones(x.shape) - swe_bathymetry(x)  # horizontal water surface
    initial_flow = q_bc(0) * np.ones(x.shape)  # Start with inflow BC

    initial_values = np.array((
        initial_height,
        initial_flow,
    ))

    # Verify consistency of initial condition
    #
    if not np.allclose(initial_values[1][0], q_bc(0)):
        raise ValueError('Initial flow condition must match prescribed inflow.')

    return initial_values


# Plot bathymetry and ICs
#
xl, xr = xspan
t0, tf = tspan

xx = np.linspace(xl, xr, num=100)
tt = np.linspace(t0, tf, num=100)

fig, (h_ax, hv_ax, q_bc_ax) = plt.subplots(3, 1)

ic = initial_condition(xx)
bb = swe_bathymetry(xx)
qq_bc = q_bc(tt)

# Plot initial wave height and bathymetry
#
h_ax.plot(xx, ic[0] + bb)
h_ax.plot(xx, bb)
h_ax.set_title('Initial wave height $h(x, 0)$')

# Plot initial flow rate
#
hv_ax.plot(xx, ic[1])
hv_ax.set_title('Initial flow rate $q(x, 0)$')

# Plot flow rate at left boundary over simulation time
#
q_bc_ax.plot(tt, qq_bc)
q_bc_ax.set_title('Boundary flow rate $q({}, t)$'.format(xl))

plt.tight_layout()
plt.show()

# Instantiate solver with bathymetry
#
solver = pde.ShallowWater1D(
    b=swe_bathymetry,
    b_x=swe_bathymetry_derivative,
    gravity=g
)


t_interval_ms = 20
dt = t_interval_ms / 1000
surface_flux = fluxes.lax_friedrichs_flux
print('Integrating ODE system...')
solution = solver.solve(
    tspan=tspan,
    xspan=xspan,
    cell_count=16,
    polydeg=4,
    initial_condition=initial_condition,
    intercell_flux=surface_flux,
    left_boundary_flux=pde.ShallowWater1D.bc_prescribed_inflow(
        q_bc,
        gravity=g,
        surface_flux=surface_flux,
    ),
    right_boundary_flux=bc.transmissive_boundary(
        surface_flux=surface_flux,
        direction=bc.Direction.DOWNSTREAM,
    ),
    quad_rule=quadrature.gll,
    **{
        'method': rk.SSPRK33,
        't_eval': np.arange(tspan[0], tspan[1], dt),
        'max_step': dt,  # max time step for ODE solver
        'rtol': 1.0e-6,
        'atol': 1.0e-6,
    }
)

# Plot solution animation
#
ani, plt = solver.plot_animation(solution, frame_interval=t_interval_ms)

# Save animation to file
#
movie_name = 'swe_1d.gif'
print('Writing movie to {}...'.format(movie_name))

ani.save(movie_name, progress_callback=lambda i, n: print(
    f'Saving animation frame {i + 1}/{n}'
) if i % 50 == 0 else None)
print('Animation written to {}.'.format(movie_name))

plt.show()
