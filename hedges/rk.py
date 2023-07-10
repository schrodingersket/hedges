import numpy as np

from scipy.integrate import OdeSolver

from scipy.integrate._ivp.rk import rk_step, RkDenseOutput
from scipy.integrate._ivp.common import (validate_max_step, validate_tol, select_initial_step,
                                         warn_extraneous, validate_first_step)


class RungeKutta(OdeSolver):
    """Base class for explicit Runge-Kutta methods."""
    C: np.ndarray = NotImplemented
    A: np.ndarray = NotImplemented
    B: np.ndarray = NotImplemented
    P: np.ndarray = NotImplemented
    order: int = NotImplemented
    error_estimator_order: int = NotImplemented
    n_stages: int = NotImplemented

    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6, vectorized=False,
                 first_step=None, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized,
                         support_complex=True)
        self.y_old = None
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)
        self.f = self.fun(self.t, self.y)
        if first_step is None:
            self.h_abs = select_initial_step(
                self.fun, self.t, self.y, self.f, self.direction,
                self.error_estimator_order, self.rtol, self.atol)
        else:
            self.h_abs = validate_first_step(first_step, t0, t_bound)
        self.K = np.empty((self.n_stages + 1, self.n), dtype=self.y.dtype)
        self.error_exponent = -1 / (self.error_estimator_order + 1)
        self.h_previous = None

    def _step_impl(self):
        t = self.t
        y = self.y

        max_step = self.max_step
        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)

        if self.h_abs > max_step:
            h_abs = max_step
        elif self.h_abs < min_step:
            h_abs = min_step
        else:
            h_abs = self.h_abs

        if h_abs < min_step:
            return False, self.TOO_SMALL_STEP

        h = h_abs * self.direction
        t_new = t + h

        if self.direction * (t_new - self.t_bound) > 0:
            t_new = self.t_bound

        h = t_new - t
        h_abs = np.abs(h)

        y_new, f_new = rk_step(self.fun, t, y, self.f, h, self.A, self.B, self.C, self.K)

        self.h_previous = h
        self.y_old = y

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.f = f_new

        return True, None

    def _dense_output_impl(self):
        Q = self.K.T.dot(self.P)
        return RkDenseOutput(self.t_old, self.t, self.y_old, Q)


class SSPRK33(RungeKutta):
    order = 3
    error_estimator_order = 2
    n_stages = 3
    C = np.array([0, 1, 1/2])
    A = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1/4, 1/4, 0]
    ])
    B = np.array([1/6, 1/6, 2/3])
    P = np.array([[1, -4 / 3, 5 / 9],
                  [0, 1, -2/3],
                  [0, 4/3, -8/9],
                  [0, -1, 1]])
