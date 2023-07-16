import numpy as np

import simulation

# Physical parameters
#
g = 1.0  # gravity

# Domain
#
tspan = (0.0, 5.0)
xspan = (-1, 1)

# Bathymetry parameters
#
b_smoothness = 0.1
b_amplitude = 0.005
b_slope = 0.05
assert(b_smoothness > 0)

# Initial wave height (including bathymetry)
#
H0 = 0.2
assert(H0 > 0)


def swe_bathymetry(x):
    """
    Describes bathymetry with an upslope which is perturbed by a hyperbolic tangent function.
    """
    return b_slope * x + b_amplitude * np.arctan(x/b_smoothness)


def swe_bathymetry_derivative(x):
    """
    Derivative of swe_bathymetry
    """
    return b_slope * np.ones(x.shape) + b_amplitude * b_smoothness/(1 + np.square(b_smoothness*x)) * np.exp(-np.square(b_smoothness*x))


runner = simulation.SWEFlowRunner(
    b=swe_bathymetry,
    b_x=swe_bathymetry_derivative,
    initial_height=H0,
    xlims=xspan,
    tlims=tspan,
    gravity=g,
    inflow_amplitude=0.05,
    inflow_smoothness=1.0,
    inflow_peak_time=1.0,
)
runner.run()
