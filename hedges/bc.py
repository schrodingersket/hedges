import enum

import numpy as np


from . import fluxes


class Direction(enum.Enum):
    UPSTREAM = 1
    DOWNSTREAM = 2


def transmissive_boundary(surface_flux=fluxes.lax_friedrichs_flux, direction=Direction.UPSTREAM):
    """
    Create a transmissive boundary by setting left and right ("ghost") states the same
    and computing the corresponding surface flux.

    For details see Section 9.2.5 of the book:
    - Eleuterio F. Toro (2001)
      Shock-Capturing Methods for Free-Surface Shallow Flows
      1st edition
      ISBN 0471987662

    :param surface_flux: Function used to compute numerical flux between adjacent cell states.
    :param direction: Boundary at which condition is implemented. Default is left boundary.
    """
    def _flux_function(u_l, u_r, xx, t, f, f_prime):
        if direction == Direction.UPSTREAM:
            return surface_flux(u_r, u_r, xx, t, f, f_prime)
        else:
            return surface_flux(u_l, u_l, xx, t, f, f_prime)

    return _flux_function


def reflective_boundary(surface_flux=fluxes.lax_friedrichs_flux, direction=Direction.UPSTREAM):
    """
    Create a transmissive boundary by reflecting the velocity between left and right ("ghost")
   and computing the corresponding surface flux.

    For details see Section 9.2.5 of the book:
    - Eleuterio F. Toro (2001)
      Shock-Capturing Methods for Free-Surface Shallow Flows
      1st edition
      ISBN 0471987662

    :param surface_flux: Function used to compute numerical flux between adjacent cell states.
    :param direction: Boundary at which condition is implemented. Default is left boundary.
    """
    def _flux_function(u_l, u_r, xx, t, f, f_prime):
        if direction == Direction.UPSTREAM:
            h_r, q_r = u_r
            return surface_flux(np.array((h_r, -q_r)), u_r, xx, t, f, f_prime)
        else:
            h_l, q_l = u_l
            return surface_flux(u_l, np.array((h_l, -q_l)), xx, t, f, f_prime)

    return _flux_function


def dirichlet_boundary(g, surface_flux=fluxes.lax_friedrichs_flux, direction=Direction.UPSTREAM):
    """
    Maintains a (possibly time-dependent) prescribed solution at the inflow boundary.

    :param g: A function with time as its first parameter, which should return the prescribed
              solution values at a particular time. The left and right cell interface values are
              passed to this function as named parameters.
    :param surface_flux: Function used to compute numerical flux between adjacent cell states.
    :param direction: Boundary at which condition is implemented. Default is left boundary.
    :return:
    """
    def _flux_function(u_l, u_r, xx, t, f, f_prime):
        # Evaluate prescribed solution at t
        #
        u = g(t, u_l=u_l, u_r=u_r)

        # Return numerical flux values
        #
        if direction == Direction.UPSTREAM:
            return surface_flux(np.array(u), u_r, xx, t, f, f_prime)
        else:
            return surface_flux(u_l, np.array(u), xx, t, f, f_prime)

    return _flux_function
