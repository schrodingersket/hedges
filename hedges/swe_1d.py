import numpy as np


import hyperbolic_solver_1d


class ShallowWater1D(hyperbolic_solver_1d.Hyperbolic1DSolver):
    solution_vars = 2

    def __init__(self, b=None, b_x=None, g=9.81):
        """
        :param b: Bathymetry function. If not provided, flat bathymetry is assumed.
        :param b_x: Derivative of bathymetry function. If not provided, flat bathymetry is assumed.
        :param g: Gravitational acceleration
        """
        self.b = b or (lambda x: np.zeros(x.shape))
        self.b_x = b_x or (lambda x: np.zeros(x.shape))
        self.g = g

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
            np.square(flow) / height + 0.5 * self.g * np.square(height)
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
            (0,                                   1),
            (-(flow/height)**2 + self.g * height, 2 * flow/height)
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
            -self.g * height * self.b_x(xx)
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
                plot_min = uu[:, i].min()
                plot_max = uu[:, i].max()
                line, = axes[i].plot(domain.flatten(), uu[0, i].flatten())

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
            else:
                line.set_ydata(u_i[k].flatten())

        # This is the format required by the animate callback for plots.
        #
        return [(line,) for line in plot_lines]
