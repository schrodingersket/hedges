import abc
import functools


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy import integrate

from . import quadrature


class Hyperbolic1DSolver(abc.ABC):
    """
    This base class serves as a basic Discontinuous Galerkin solver for 1D hyperbolic PDEs and is
    written to be as clear as possible to those who are new to such solvers. To add support for a
    particular PDE (e.g., advection, SWE, etc.), the three abstract methods
    :func:`dg.hyperbolic_solver_1d.Hyperbolic1DSolver.hyperbolic_flux_term`,
    :func:`dg.hyperbolic_solver_1d.Hyperbolic1DSolver.hyperbolic_flux_jacobian`, and
    :func:`dg.hyperbolic_solver_1d.Hyperbolic1DSolver.source_term` defined in this class must be
    implemented. See :class:`dg.swe_1d.ShallowWater1D` for an example.

    This class also contains a basic function for animating numerical solutions; by default, all
    solution variables are plotted as subplots in a single animation, though this behavior can be
    overridden as desired by the subclasses implementation. See :class:`dg.swe_1d.ShallowWater1D`
    for an example.
    """
    solution_vars = NotImplemented  # Indicates the number of variables in the PDE solution.

    @abc.abstractmethod
    def hyperbolic_flux_term(self, u, xx, t):
        """
        Compute flux term from solution values (u) at a particular time step. This corresponds to
        F(u) in the conservative PDE form:

        u_t + [F(u)]_x = S(u)

        :param u: Solution values at t
        :param xx: Domain values
        :param t: Current timestep value
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def hyperbolic_flux_jacobian(self, u, xx, t):
        """
        Compute flux Jacobian from solution values (u) at a particular time step. This corresponds
        to F'(u) in the quasi-linear form:

        u_t + F'(u) u_x = S(u)

        :param u: Solution values at t
        :param xx: Domain values
        :param t: Current timestep value
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def source_term(self, u, xx, t):
        """
        Compute source term from solution values (u) at a particular time step. This corresponds to
        S(u) in the conservative PDE form:

        u_t + [F(u)]_x = S(u)

        :param u: Solution values at t
        :param xx: Domain values
        :param t: Current timestep value
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def _to_physical_domain(x_l, x_r, xi):
        """
        Transforms a set of reference coordinates (i.e., points between [-1, 1]) from a cell's
        reference domain to a particular element in the physical domain.

        In other words, for a cell Q_l with bounds x_l and x_r, this function
        makes the affine transformation:

        [-1, 1] -> [x_l, x_r].

        :param x_l: Left boundary of physical domain cell
        :param x_l: Right boundary of physical domain cell
        :param xi: Collection of points contained in the reference domain [-1, 1].
        :return: Tuple of (x_k, J_k), where x_k are the mapped points and J_k is the Jacobian of
                 the transformation variables (used for quadrature evaluations on arbitrary
                 intervals)
        """
        return (x_l + x_r + xi * (x_r - x_l))/2, (x_r - x_l)/2

    def solve(self, tspan, xspan, cell_count, polydeg, left_boundary_flux, right_boundary_flux,
              intercell_flux, initial_condition, t_eval_interval=20,
              quad_rule=quadrature.gll, **ivp_options):
        """
        Solves the hyperbolic PDE via the Method of Lines (MoL) approach with standard SciPy ODE
        solvers.

        :param tspan: Time domain on which the PDE is to be solved.
        :param xspan: Physical domain on which the PDE is to be solved.
        :param cell_count: Number of cells to split domain into
        :param polydeg: Degree of approximating polynomial (same across all cells).
        :param left_boundary_flux: Function used to compute flux at left boundary (inflow)
        :param right_boundary_flux: Function used to compute flux at right boundary (outflow)
        :param intercell_flux: Function used to compute flux in between adjacent cells.
        :param initial_condition: Function used to compute the solution's initial condition.
        :param method: ODE Integration scheme used to solve semidiscretized problem. Should
                       correspond to an existing SciPy ODE solver.
        :param t_eval_interval: Interval (in ms) to evaluate interpolated solution. For instance,
                                if tspan=[0, 0.5] and this value is set to 100, this function
                                returns the solution at points {0, 0.1, 0.2, 0.3, 0.4, 0.5}.
        :param quad_rule: Quadrature rule. Given a polynomial degree, this function should
                          return a tuple (x, w) where x and w are arrays containing the
                          quadrature nodes and weights.
        :return: (sol, xx) tuple, where sol is the system solution and xx is the domain (in the
                 shape that each solution variable at each timestep should be reshaped to). The
                 first dimension represents a particular cell and the second the GLL nodes (as
                 mapped from the reference domain [-1, 1] into the physical domain) in each cell.
        """
        x_l, x_r = xspan
        t_0, t_f = tspan

        # Gimme some SWEet Gauss-Legendre-Lobatto points/weights on [-1, 1]
        #
        x, w = quad_rule(polydeg)

        # Get Lagrange differentiation matrix to compute derivative of polynomial interpolant
        #
        differentiation_matrix = quadrature.lagrange_differentiation_matrix(x)

        # Mass matrix; basically just a convenient way to implement a Gaussian quadrature rule.
        #
        mass_matrix = np.diag(w)

        # Boundary matrix for assigning numerical flux values
        #
        boundary_matrix = np.zeros(mass_matrix.shape)
        boundary_matrix[0, 0] = -1
        boundary_matrix[-1, -1] = 1

        # First dimension is the solution variable of interest (e.g., length 1 for scalar 1D
        # advection, length 2 for 1D SWE, etc.), the second dimension represents the domain
        # cells, and the third dimension represents the cell (GLL) nodes.
        #
        u_shape = (self.solution_vars, cell_count, len(x))

        # Shape of domain is a tuple of the latter two elements of u_shape.
        #
        xx_shape = u_shape[1:]

        # Split domain into evenly-spaced cells
        #
        dx = (x_r - x_l) / cell_count

        # Stores node values across entire computational domain (i.e., [x_k^L, x_k^R])
        #
        xx = np.empty(xx_shape)

        # Stores the length of each domain cell.
        #
        cell_lengths = np.empty(cell_count)

        for cell_k in range(0, cell_count):
            cell_left = x_l + cell_k * dx
            cell_right = cell_left + dx
            xx[cell_k], cell_lengths[cell_k] = self._to_physical_domain(cell_left, cell_right, x)

        # This allows us to get a feel for the status of the integration scheme. This will print our
        # the status no more than 20 times over the course of integration.
        #
        time_buckets = np.linspace(t_0, t_f, num=20)

        def uprime(t, u_col, ivp_args):
            if t > time_buckets[ivp_args['bucket']]:
                ivp_args['bucket'] = ivp_args['bucket'] + 1
                print('Computing du/dt at t={}'.format(t))

            # Reshape column vector into cell-node matrix.
            #
            u = u_col.reshape(u_shape)

            # Stores computed value of u' at this time step
            #
            du_t = np.zeros(u_shape)

            # Placeholder for numerical flux values
            #
            numerical_flux = np.zeros(u_shape)

            # Evaluate hyperbolic flux
            #
            f_u = self.hyperbolic_flux_term(u, xx, t)

            # Evaluate hyperbolic source term
            #
            s_u = self.source_term(u, xx, t)

            for cell_k in range(1, cell_count - 1):
                left_flux = intercell_flux(
                    u[:, cell_k - 1, -1],
                    u[:, cell_k, 0],
                    xx,
                    t,
                    f=self.hyperbolic_flux_term,
                    f_prime=self.hyperbolic_flux_jacobian
                )

                numerical_flux[:, cell_k, 0] = left_flux
                numerical_flux[:, cell_k - 1, -1] = left_flux

                right_flux = intercell_flux(
                    u[:, cell_k, -1],
                    u[:, cell_k + 1, 0],
                    xx,
                    t,
                    f=self.hyperbolic_flux_term,
                    f_prime=self.hyperbolic_flux_jacobian
                )
                numerical_flux[:, cell_k, -1] = right_flux
                numerical_flux[:, cell_k + 1, 0] = right_flux

            # Compute and assign numerical flux for boundary interfaces. This is where boundary
            # conditions are implemented.
            #
            numerical_flux[:, 0, 0] = left_boundary_flux(
                u[:, -1, -1],
                u[:, 0, 0],
                xx,
                t,
                f=self.hyperbolic_flux_term,
                f_prime=self.hyperbolic_flux_jacobian
            )
            numerical_flux[:, -1, -1] = right_boundary_flux(
                u[:, -1, -1],
                u[:, 0, 0],
                xx,
                t,
                f=self.hyperbolic_flux_term,
                f_prime=self.hyperbolic_flux_jacobian
            )

            # Compute integrals via quadrature
            #
            for var in range(0, self.solution_vars):
                # Update values in each cell
                #
                for cell_k in range(0, cell_count):
                    # Compute volume integrals
                    #
                    du_t[var][cell_k] += np.linalg.solve(mass_matrix, differentiation_matrix.T) \
                                         @ mass_matrix \
                                         @ f_u[var][cell_k]

                    # Compute surface integrals
                    #
                    du_t[var][cell_k] -= np.linalg.solve(mass_matrix, boundary_matrix) \
                                         @ (numerical_flux[var][cell_k])

                    # Apply Jacobian from mapping to reference element
                    #
                    du_t[var][cell_k] /= cell_lengths[cell_k]

                    # Add source term
                    #
                    du_t[var][cell_k] += s_u[var][cell_k]

            # Flatten column-wise
            #
            return du_t.flatten()

        # Flatten initial condition cell grid into vector
        #
        ic = initial_condition(xx).flatten()

        # We wrap this value in a dict so that uprime can actually mutate t_uniq. t_uniq is used
        # to print ODE solver progress.
        #
        ode_args = {'bucket': 0}

        # Solve ODE system
        #
        sol = integrate.solve_ivp(
            uprime,
            (t_0, t_f),
            ic,
            args=(ode_args,),
            **ivp_options
        )

        return sol, xx

    def plot_animation(self, solution, frame_interval=20, animate_frame=None, initialize_plots=None):
        """
        Animates plots of time-dependent solutions over time. This function plots all solution
        variables as subplots in a single plot.

        :param solution: Solution tuple in the form returned by
                         :func:`~hyperbolic_solver_1d.Hyperbolic1DSolver.solve`
        :param frame_interval: Interval between animation frames in milliseconds
        :param animate_frame: Optional custom animation function (e.g., to draw SWE bathymetry)
        :param initialize_plots: Optional function for initializing plots. If not provided,
                                 solutions will be plotted according to their respective min/max
                                 values over the entire time domain.
        :return:
        """
        sol, xx = solution

        # Plot animated solution
        #
        u_shape = (self.solution_vars, xx.shape[0], xx.shape[1])
        fig, ax = plt.subplots(self.solution_vars, 1)
        fig.tight_layout()

        if initialize_plots:
            lines = initialize_plots(ax, xx, sol, u_shape)
        else:
            lines = self._initialize_plots(ax, xx, sol, u_shape)

        ani = animation.FuncAnimation(
            fig,
            functools.partial(
                animate_frame or self._default_frame_animation,
                domain=xx,
                soln=sol,
                plot_lines=lines,
                axes=ax,
                soln_shape=u_shape,
            ),
            frames=len(sol.t),
            interval=frame_interval,
            repeat=True,
            repeat_delay=1.0 * 1000,
        )

        return ani, plt

    def _initialize_plots(self, axes, domain, soln, soln_shape):
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
            axes[i].set_ylim(uu[:, i].min(), uu[:, i].max())
            line, = axes[i].plot(domain.flatten(), uu[0, i].flatten())
            plot_lines.append(line)

        return plot_lines

    def _default_frame_animation(self, n, axes, plot_lines, domain, soln, soln_shape):
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

        # Plot each solution quantity
        #
        for k, line in enumerate(plot_lines):
            axes[k].set_title('$q_{}; t = {:0.2f}$'.format(k+1, soln.t[n]))
            line.set_xdata(domain.flatten())
            line.set_ydata(u_i[k].flatten())

        # This is the format required by the animate callback for plots.
        #
        return [(line,) for line in plot_lines]
