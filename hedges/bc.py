import numpy as np


from . import fluxes


def transmissive_outflow(surface_flux=fluxes.lax_friedrichs_flux):
    """
    Create a transmissive boundary by setting left and right ("ghost") states the same
    and computing the corresponding surface flux.

    For details see Section 9.2.5 of the book:
    - Eleuterio F. Toro (2001)
      Shock-Capturing Methods for Free-Surface Shallow Flows
      1st edition
      ISBN 0471987662

    :param surface_flux: Function used to compute numerical flux between adjacent cell states.
    """
    def _flux_function(u_l, u_r, xx, t, f, f_prime):
        """

        :param u_l:
        :param u_r:
        :param xx:
        :param t:
        :param f:
        :param f_prime:
        :return:
        """
        return surface_flux(u_l, u_l, xx, t, f, f_prime)

    return _flux_function

def reflective_outflow(surface_flux=fluxes.lax_friedrichs_flux):
    """
    Create a transmissive boundary by reflecting the velocity between left and right ("ghost")
   and computing the corresponding surface flux.

    For details see Section 9.2.5 of the book:
    - Eleuterio F. Toro (2001)
      Shock-Capturing Methods for Free-Surface Shallow Flows
      1st edition
      ISBN 0471987662

    :param surface_flux: Function used to compute numerical flux between adjacent cell states.
    """
    def _flux_function(u_l, u_r, xx, t, f, f_prime):
        h_l, q_l = u_l

        return surface_flux(u_l, np.array((h_l, -q_l)), xx, t, f, f_prime)

    return _flux_function
