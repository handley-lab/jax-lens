"""
JAX-Native Strong Gravitational Lensing Library

A functional, JAX-native reimplementation of PyAutoLens core functionality,
designed for efficient batched MCMC sampling via jax.vmap.

This library provides:
- Functional light profiles (Sersic, Gaussian, Exponential)
- Functional mass profiles (SIS, SIE, NFW)
- Multi-plane ray-tracing
- Imaging likelihood computation
- vmap-compatible parameter batching

Example usage:
    import jax.numpy as jnp
    import jax_lens as jl

    # Define parameters as a PyTree
    params = {
        'lens': {
            'centre': jnp.array([0.0, 0.0]),
            'einstein_radius': 1.0,
        },
        'source': {
            'centre': jnp.array([0.1, 0.1]),
            'intensity': 1.0,
            'effective_radius': 0.5,
            'sersic_index': 1.0,
        }
    }

    # Compute likelihood
    log_like = jl.log_likelihood(params, data, noise_map, grid, psf)

    # Batch over parameters with vmap
    batched_log_like = jax.vmap(jl.log_likelihood, in_axes=(0, None, None, None, None))
"""

from jax_lens.profiles import light, mass
from jax_lens.profiles.light import (
    sersic,
    gaussian,
    exponential,
    dev_vaucouleurs,
)
from jax_lens.profiles.mass import (
    sis_deflections,
    sie_deflections,
    nfw_deflections,
    sis_convergence,
    sie_convergence,
    nfw_convergence,
)
from jax_lens.lens.tracer import (
    trace_grid,
    tracer_image,
    traced_grid_list,
)
from jax_lens.fitting.imaging import (
    log_likelihood,
    chi_squared,
    residuals,
)
from jax_lens.cosmology.distances import (
    angular_diameter_distance,
    scaling_factor,
)
from jax_lens.pipeline import (
    create_likelihood_fn,
    batched_likelihood,
)

__version__ = "0.1.0"
__all__ = [
    # Light profiles
    "sersic",
    "gaussian",
    "exponential",
    "dev_vaucouleurs",
    # Mass profiles
    "sis_deflections",
    "sie_deflections",
    "nfw_deflections",
    "sis_convergence",
    "sie_convergence",
    "nfw_convergence",
    # Tracer
    "trace_grid",
    "tracer_image",
    "traced_grid_list",
    # Fitting
    "log_likelihood",
    "chi_squared",
    "residuals",
    # Cosmology
    "angular_diameter_distance",
    "scaling_factor",
    # Pipeline
    "create_likelihood_fn",
    "batched_likelihood",
]
