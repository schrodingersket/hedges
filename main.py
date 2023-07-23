import os
import csv
import numpy as np

import hedges.simulation

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
h0 = 0.5
assert(h0 > 0)

# Boolean flag indicating whether solution should be saved to file.
#
save_animation = True


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


if __name__ == '__main__':
    amplitude = np.linspace(0.01, 0.1, num=9)
    smoothness = np.linspace(0.5, 2.0, num=16)

    inflow_peak_time = 4.0

    for j, a in enumerate(amplitude):
        for k, s in enumerate(smoothness):

            def q_bc(t):
                """
                Describes a Gaussian inflow, where the function transitions to a constant value upon
                attaining its maximum value.

                :param t: Time
                :return:
                """
                tt = np.array(t)
                return a * np.exp(
                    -np.square(
                        np.minimum(tt, inflow_peak_time * np.ones(tt.shape)) - inflow_peak_time
                    ) / (2 * np.square(s))
                )

            # Instantiate and run DG solver
            #
            runner = hedges.simulation.SWEFlowRunner(
                b=swe_bathymetry,
                b_x=swe_bathymetry_derivative,
                h0=h0,
                xlims=xspan,
                tlims=tspan,
                gravity=g,
                q_bc=q_bc,
            )
            print('Running simulation {}/{} with a={}, s={}...'.format(
                (j+1)*(k+1),
                len(amplitude) * len(smoothness),
                a,
                s,
            ))
            solution, ani = runner.run()

            # Save solver parameters to file
            #
            with open(os.path.join(runner.save_dir, 'parameters.csv'), 'w', newline='') as csvfile:
                params = {
                    'gravity': g,
                    'inflow_amplitude': a,
                    'inflow_smoothness': s,
                    'inflow_peak_time': inflow_peak_time,
                    'xl': xspan[0],
                    'xr': xspan[1],
                    't0': tspan[0],
                    'tf': tspan[1],
                }
                writer = csv.DictWriter(csvfile, fieldnames=params.keys())
                writer.writeheader()
                writer.writerow(params)

            # Write animation to file
            #
            if save_animation:
                movie_name = os.path.join(runner.save_dir, 'swe_1d.gif')
                print('Writing movie to {}...'.format(movie_name))

                ani.save(movie_name, progress_callback=lambda i, n: print(
                    f'Saving animation frame {i + 1}/{n}'
                ) if i % 50 == 0 else None)
                print('Animation written to {}.'.format(movie_name))

    print('Simulation execution complete.')
