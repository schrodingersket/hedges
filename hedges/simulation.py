import binascii
import csv
import os
import time


import numpy as np


from . import bc
from . import fluxes
from . import quadrature
from . import rk
from . import swe_1d


class SWEFlowRunner:
    """
    Helper class for running several simulations in sequence or parallel.
    """
    def __init__(
            self,
            b,
            b_x,
            initial_height,
            xlims,
            tlims,
            gravity=9.81,
            inflow_amplitude=0.05,
            inflow_smoothness=1.0,
            inflow_peak_time=1.0,
            save_dir='./out',
    ) -> None:
        self.gravity = gravity
        self.inflow_amplitude = inflow_amplitude
        self.inflow_smoothness = inflow_smoothness
        self.inflow_peak_time = inflow_peak_time
        self.b = b
        self.b_x = b_x
        self.initial_height = initial_height
        self.xlims = xlims
        self.tlims = tlims

        # Used to save files in unique folders organized by time. Folder name format is
        # "<epoch timestamp>_<six-digit random hex>"
        #
        self._save_dir = os.path.join(
            save_dir,
            '{}_{}'.format(int(time.time()), binascii.b2a_hex(os.urandom(4)).decode())
        )

        assert(inflow_amplitude > 0)

    def q_bc(self, t):
        """
        Describes a Gaussian inflow, where the function transitions to a constant value upon
        attaining its maximum value.

        :param t: Time
        :return:
        """
        tt = np.array(t)
        return self.inflow_amplitude * np.exp(
            -np.square(
                np.minimum(tt, self.inflow_peak_time * np.ones(tt.shape)) - self.inflow_peak_time
            ) / (2 * np.square(self.inflow_smoothness))
        )

    def initial_condition(self, x):
        """
        Creates initial conditions for (h, uh).

        :param x: Computational domain
        :return:
        """
        # Horizontal water surface
        #
        initial_height = self.initial_height * np.ones(x.shape) - self.b(x)

        # Start with whatever flow is prescribed by our inflow BC
        #
        initial_flow = self.q_bc(0) * np.ones(x.shape)

        ic = np.array((
            initial_height,
            initial_flow,
        ))

        # Verify consistency of initial condition
        #
        if not np.allclose(ic[1][0], self.q_bc(0)):
            raise ValueError('Initial flow condition must match prescribed inflow.')

        return ic

    def run(self):
        print('Writing output files to {}'.format(self._save_dir))
        os.makedirs(self._save_dir, exist_ok=True)

        # Write solver parameters to file
        #
        with open(os.path.join(self._save_dir, 'parameters.csv'), 'w', newline='') as csvfile:
            params = {
                'gravity': self.gravity,
                'inflow_amplitude': self.inflow_amplitude,
                'inflow_smoothness': self.inflow_smoothness,
                'inflow_peak_time': self.inflow_peak_time,
                'xl': self.xlims[0],
                'xr': self.xlims[1],
                't0': self.tlims[0],
                'tf': self.tlims[1],
            }
            writer = csv.DictWriter(csvfile, fieldnames=params.keys())
            writer.writeheader()
            writer.writerow(params)

        # Instantiate solver with bathymetry
        #
        solver = swe_1d.ShallowWater1D(
            b=self.b,
            b_x=self.b_x,
            gravity=self.gravity
        )

        t_interval_ms = 20
        dt = t_interval_ms / 1000
        surface_flux = fluxes.lax_friedrichs_flux
        solution = solver.solve(
            tspan=self.tlims,
            xspan=self.xlims,
            cell_count=16,
            polydeg=4,
            initial_condition=self.initial_condition,
            intercell_flux=surface_flux,
            left_boundary_flux=swe_1d.ShallowWater1D.bc_prescribed_inflow(
                self.q_bc,
                gravity=self.gravity,
                surface_flux=surface_flux
            ),
            right_boundary_flux=bc.transmissive_outflow(surface_flux=surface_flux),
            quad_rule=quadrature.gll,
            **{
                'method': rk.SSPRK33,
                't_eval': np.arange(self.tlims[0], self.tlims[1] + dt, dt),
                'max_step': dt,  # max time step for ODE solver
                'rtol': 1.0e-6,
                'atol': 1.0e-6,
            }
        )

        sol, XX = solution  # Uppercase variables indicate matrix; lowercase indicates vector
        xx = XX.flatten()
        uu = sol.y.T.reshape(len(sol.t), solver.solution_vars, XX.shape[0], XX.shape[1])
        bb = self.b(xx)
        bb_x = self.b_x(xx)
        with open(os.path.join(self._save_dir, 'solution.csv'), 'w', newline='') as csvfile:
            params = {
                't': None,
                'x': None,
                'h': None,
                'q': None,
                'b': None,
                'b_x': None,
            }
            writer = csv.DictWriter(csvfile, fieldnames=params.keys())
            writer.writeheader()
            for n, t in enumerate(sol.t):
                params['t'] = t
                hh = uu[n, 0, :, :].flatten()
                qq = uu[n, 1, :, :].flatten()

                for k, x in enumerate(xx):
                    params['x'] = x
                    params['h'] = hh[k]
                    params['q'] = qq[k]
                    params['b'] = bb[k]
                    params['b_x'] = bb_x[k]
                    writer.writerow(params)

        # Plot solution animation
        #
        ani, plt = solver.plot_animation(solution, frame_interval=t_interval_ms)

        # Save animation to file
        #
        movie_name = os.path.join(self._save_dir, 'swe_1d.gif')
        print('Writing movie to {}...'.format(movie_name))

        ani.save(movie_name, progress_callback=lambda i, n: print(
            f'Saving animation frame {i + 1}/{n}'
        ) if i % 50 == 0 else None)
        print('Animation written to {}.'.format(movie_name))
