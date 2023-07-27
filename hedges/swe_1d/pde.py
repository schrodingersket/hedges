import numpy as np
from scipy import optimize

from hedges import bc
from hedges import fluxes
from hedges import hyperbolic_solver_1d


class ShallowWater1D(hyperbolic_solver_1d.Hyperbolic1DSolver):
    solution_vars = 2

    def __init__(self, b=None, b_x=None, gravity=9.81):
        """
        :param b: Bathymetry function. If not provided, flat bathymetry is assumed.
        :param b_x: Derivative of bathymetry function. If not provided, flat bathymetry is assumed.
        :param gravity: Gravitational acceleration
        """
        self.b = b or (lambda x: np.zeros(x.shape))
        self.b_x = b_x or (lambda x: np.zeros(x.shape))
        self.gravity = gravity

    def hyperbolic_flux_term(self, u, xx, t):
        """
        SWE hyperbolic flux term:

        F(x) = [
          q
          q^2 / h + 1/2 * g * h^2
        ]

        :param u:
        :param xx:
        :param t:
        :return:
        """
        height, flow = u

        return np.array((
            flow,
            np.square(flow) / height + 0.5 * self.gravity * np.square(height)
        ))

    def hyperbolic_flux_jacobian(self, u, xx, t):
        """
        SWE hyperbolic flux Jacobian:

        F'(x) = [
          0                     1
          -q^2 / h^2 + g*h      2 * q / h
        ]

        :param u:
        :param xx:
        :param t:
        :return:
        """
        height, flow = u

        return np.array((
            (0,                                         1),
            (-(flow/height)**2 + self.gravity * height, 2 * flow/height)
        ))

    def source_term(self, u, xx, t):
        """
        SWE source term:

        S(x) = [
          0
          -g * h * b_x
        ]

        :param u:
        :param xx:
        :param t:
        :return:
        """
        height, flow = u

        return np.array((
            np.zeros(height.shape),
            -self.gravity * height * self.b_x(xx)
        ))

    def plot_animation(self, solution, frame_interval=20, animate_frame=None, initialize_plots=None):
        """
        Overrides base class implementation in order to add bathymetry to animations.

        :param solution:
        :param frame_interval:
        :param animate_frame:
        :param initialize_plots:
        :return:
        """
        return super().plot_animation(
            solution,
            frame_interval,
            animate_frame=self._animate_swe_frame,
            initialize_plots=self._initialize_swe_plots
        )

    def _initialize_swe_plots(self, axes, domain, soln, soln_shape):
        """
        Initializes plots for animation. Axis limits for each subplot are set to max/min values
        across entire time domain with respect to the subplot's solution variable.

        :param axes: Subplot axes.
        :param domain:
        :param soln: Solution indexed by time step
        :param soln_shape: Shape of solution variable
        :return:
        """
        plot_lines = []

        # Reshape solution into something a little more intuitive to work with. First dimension
        # is the time step index, second is the solution variable of interest, third dimension
        # is the cell index, and fourth is the node.
        #
        uu = soln.y.T.reshape(len(soln.t), *soln_shape)

        for i in range(0, self.solution_vars):
            # Plot solution variables
            #
            if i == 0:
                bb = self.b(domain)
                # Plot bathymetry
                #
                line, = axes[i].plot(domain.flatten(), uu[0, i].flatten() + bb.flatten())
                axes[i].plot(domain.flatten(), bb.flatten())
                plot_min = min(uu[:, i].min(), bb.min())
                plot_max = max(uu[:, i].max(), bb.max())
            else:
                # Plot flow
                #
                plot_min = (uu[:, i]).min()
                plot_max = (uu[:, i]).max()
                line, = axes[i].plot(domain.flatten(), (uu[0, i]).flatten())

                # Plot velocity
                #
                # plot_min = (uu[:, i]/uu[:, 0]).min()
                # plot_max = (uu[:, i]/uu[:, 0]).max()
                # line, = axes[i].plot(domain.flatten(), (uu[0, i]/uu[0, 0]).flatten())

            plot_lines.append(line)

            # Add a little extra padding to top and bottom of plots
            #
            padding = (0.1 * (plot_max - plot_min)) or 0.1
            axes[i].set_ylim(plot_min - padding, plot_max + padding)

        return plot_lines

    def _animate_swe_frame(self, n, axes, plot_lines, domain, soln, soln_shape):
        """
        Animate plot according to:

        https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html

        :param n: Frame index
        :param axes: Subplot axes.
        :param plot_lines: Plot lines to update. This should contain one line per solution.
        :param domain:
        :param soln: Solution indexed by time step
        :param soln_shape: Shape of solution variable

        :return:
        """
        u_i = soln.y[:, n].reshape(soln_shape)
        var_names = {
            0: 'h',
            1: 'hv'
        }

        # Plot each solution quantity
        #
        for k, line in enumerate(plot_lines):
            axes[k].set_title('${}; t = {:0.2f}$'.format(var_names[k], soln.t[n]))
            line.set_xdata(domain.flatten())
            if k == 0:
                bb = self.b(domain)
                # Add bathymetry plot
                #
                line.set_ydata(u_i[k].flatten() + bb.flatten())
            elif k == 1:
                # Plot flow
                #
                line.set_ydata((u_i[k]).flatten())

                # Plot velocity
                #
                # line.set_ydata((u_i[k]/u_i[0]).flatten())

        # This is the format required by the animate callback for plots.
        #
        return [(line,) for line in plot_lines]

    @staticmethod
    def surface_flux_fjordholm_etal(gravity=9.81):
        """
        Total energy conservative (mathematical entropy for shallow water equations). When the bottom topography
        is nonzero this should only be used as a surface flux otherwise the scheme will not be well-balanced.
        For well-balancedness in the volume flux use [`flux_wintermeyer_etal`](@ref).

        Details are available in Eq. (4.1) in the paper:
        - Ulrik S. Fjordholm, Siddhartha Mishr and Eitan Tadmor (2011)
          Well-balanced and energy stable schemes for the shallow water equations with discontinuous topography
          [DOI: 10.1016/j.jcp.2011.03.042](https://doi.org/10.1016/j.jcp.2011.03.042)

        :param gravity: Gravitational constant.
        """
        def _flux_function(u_l, u_r, xx, t, f, f_prime):
            # Unpack left and right state
            h_l, q_l = u_l
            h_r, q_r = u_r

            v_l = q_l / h_l
            v_r = q_r / h_r

            # Average each factor of products in flux
            h_avg = 0.5 * (h_l + h_r)
            v_avg = 0.5 * (v_l + v_r)
            p_avg = 0.25 * gravity * (np.square(h_l) + np.square(h_r))

            # Calculate fluxes depending on orientation
            f1 = h_avg * v_avg
            f2 = f1 * v_avg + p_avg

            return np.array((f1, f2))
        return _flux_function

    @staticmethod
    def volume_flux_wintermeyer_etal(gravity=9.81):
        """
        Total energy conservative (mathematical entropy for shallow water equations) split form.
        When the bottom topography is nonzero this scheme will be well-balanced when used as a `volume_flux`.
        The `surface_flux` should still use, e.g., [`flux_fjordholm_etal`](@ref).

        Further details are available in Theorem 1 of the paper:
        - Niklas Wintermeyer, Andrew R. Winters, Gregor J. Gassner and David A. Kopriva (2017)
          An entropy stable nodal discontinuous Galerkin method for the two dimensional
          shallow water equations on unstructured curvilinear meshes with discontinuous bathymetry
          [DOI: 10.1016/j.jcp.2017.03.036](https://doi.org/10.1016/j.jcp.2017.03.036)

        :param gravity: Gravitational constant.
        """
        def _flux_function(u_l, u_r, xx, t, f, f_prime):

            # Unpack left and right state
            #
            h_l, q_l = u_l
            h_r, q_r = u_r

            # Get the velocities on either side
            #
            v_l = q_l / h_l
            v_r = q_r / h_r

            # Average each factor of products in flux
            #
            v_avg = 0.5 * (v_l + v_r)
            p_avg = 0.5 * gravity * h_l * h_r

            # Calculate fluxes depending on orientation
            #
            f1 = 0.5 * (q_l + q_r)
            f2 = f1 * v_avg + p_avg

            return np.array((f1, f2))
        return _flux_function

    @staticmethod
    def bc_prescribed_inflow(q_in, gravity=9.81, surface_flux=fluxes.lax_friedrichs_flux):
        """
        Maintains a (possibly time-dependent) prescribed rate of flow at the inflow boundary by
        setting backward characteristics for upwinded values and cell values equal at the leftmost
        boundary.

        :param q_in: A function with time as its single parameter which should return the rate of
                     flow at a particular time.
        :param gravity: Gravitational constant.
        :param surface_flux: Function used to compute numerical flux between adjacent cell states.
        :return:
        """
        def qbc(t, u_l, u_r):
            h_r, q_r = u_r

            # Backward characteristic
            #
            w_b = q_r / h_r - 2 * np.sqrt(gravity * h_r)

            # Evaluate prescribed bathymetry value at t
            #
            q = q_in(t)

            # We solve for the square root of h to mitigate numerical issues with Newton's method
            #
            sqrt_h = optimize.newton(
                lambda hh: q / np.square(hh) - 2 * np.sqrt(gravity) * hh - w_b,
                x0=np.square(h_r),
                # fprime=lambda hh: -2 * q / hh ** 3 - 2 * np.sqrt(g),
                maxiter=500
            )
            h = np.square(sqrt_h)

            return h, q

        return bc.dirichlet_boundary(
            qbc,
            surface_flux=surface_flux,
            direction=bc.Direction.UPSTREAM,
        )
