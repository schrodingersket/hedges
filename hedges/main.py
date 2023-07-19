import numpy as np

from . import simulation

# Physical parameters
#
g = 1.0  # gravity

# Domain
#
tspan = (0.0, 10.0)
xspan = (-1, 1)

# Bathymetry parameters
#
b_smoothness = 0.1
b_amplitude = 0.02
b_slope = 0.05
assert(b_smoothness > 0)

# Initial wave height (including bathymetry)
#
H0 = 0.5
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


amplitude = np.linspace(0.01, 0.1, num=9)
smoothness = np.linspace(0.5, 2.0, num=16)

for j, a in enumerate(amplitude):
    for k, s in enumerate(smoothness):
        runner = simulation.SWEFlowRunner(
            b=swe_bathymetry,
            b_x=swe_bathymetry_derivative,
            initial_height=H0,
            xlims=xspan,
            tlims=tspan,
            gravity=g,
            inflow_amplitude=a,
            inflow_smoothness=s,
            inflow_peak_time=4.0,
        )
        print('Running simulation {}/{} with a={}, s={}...'.format(
            (j+1)*(k+1),
            len(amplitude) * len(smoothness),
            a,
            s,
        ))
        runner.run()

print('Simulation execution complete.')
