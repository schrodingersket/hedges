[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/schrodingersket/hedges/HEAD?urlpath=notebooks%2Fhedges%2Fmain.ipynb)

# Hyperbolic Educational Discontinuous Galerkin Equation Solver

This repository contains a Discontinuous Galerkin Python solver for 1D Hyperbolic PDEs. It is 
intended to be used primarily for education and as such prioritizes clarity in the codebase over
computational efficiency or cleverness. It is designed to be extensible and modular so that support
for new hyperbolic systems is easy to implement, and comes out of the box with an example for
the 1D shallow water equations (SWE) with variable bathymetry, prescribed inflow, and free outflow.

## Usage

Once you've installed the dependencies listed in `requirements.txt` 
(`pip install -r requirements.txt`), you can simply run `hedges/main.py` to generate a full 
animation of the SWE:

```python hedges/main.py```

This also saves the animation to `swe_1d.gif` for whatever use your heart desires. The 
`plot_animation` function of the `hedges.hyperbolic_solver_1d.Hyperbolic1DSolver` base class returns 
a tuple of the form 
([FuncAnimation](https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.animation.FuncAnimation.html), 
[matplotlib](https://matplotlib.org/3.5.1/api/matplotlib_configuration_api.html#matplotlib)) which 
may be used to save individual frames or modify plots as desired. Subclass implementations may 
optionally override the default plotting behavior, with a reference implementation provided in 
`hedges.swe_1d.ShallowWater1D`.

A Jupyter notebook is provided in this repository and is available at 
[BinderHub](https://mybinder.org/v2/gh/schrodingersket/hedges/HEAD?urlpath=notebooks%2Fhedges%2Fmain.ipynb) to experiment with.

## Extending the Solver

Currently, only the 1D Shallow Water equations are implemented. However, support can easily be added
for other 1D hyperbolic PDE systems by simply subclassing `hedges.hyperbolic_solver_1d.Hyperbolic1DSolver`
and implementing the flux `F(u)`, source `S(u)`, and flux Jacobian `F'(u)` terms for the desired 
system in conservative form. See `hedges.swe_1d.ShallowWater1D` for a reference implementation.

A local Lax-Friedrichs flux is used as the approximate Riemann solver between cell 
interfaces, but the `solve` method of the `hedges.hyperbolic_solver_1d.Hyperbolic1DSolver` class accepts
a function reference which can be used to implement e.g. HLL flux (or exact Riemann solvers).
