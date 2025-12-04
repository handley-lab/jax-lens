"""
Functional light and mass profiles for gravitational lensing.

All functions are pure and JAX-compatible, designed for use with jax.vmap.
"""

from jax_lens.profiles import light, mass

__all__ = ["light", "mass"]
