"""
Functional ray-tracing for gravitational lensing.

This module provides pure JAX functions for multi-plane ray-tracing.
All functions are designed to be compatible with jax.vmap for batched parameter evaluation.

The key insight for JAX compatibility is that the *structure* of the lens system
(number of planes, types of profiles) is determined at compile time, while only
the *parameter values* vary at runtime.

Reference: PyAutoLens tracer.py and tracer_util.py
"""

import jax.numpy as jnp
from jax import lax
from typing import List, Tuple, Dict, Callable, Optional, NamedTuple
from functools import partial

from jax_lens.profiles import light, mass
from jax_lens.cosmology.distances import scaling_factor, Cosmology, PLANCK15


class PlaneConfig(NamedTuple):
    """Configuration for a single lensing plane."""

    redshift: float
    light_profile_types: Tuple[str, ...]  # e.g., ('sersic', 'sersic')
    mass_profile_types: Tuple[str, ...]  # e.g., ('sie',)


class TracerConfig(NamedTuple):
    """
    Static configuration for a tracer.

    This defines the structure of the lens system at compile time.
    The actual parameter values are passed separately.
    """

    planes: Tuple[PlaneConfig, ...]
    cosmology: Cosmology = PLANCK15


# =============================================================================
# Core Ray-Tracing Functions
# =============================================================================


def trace_grid(
    grid: jnp.ndarray,
    deflections: jnp.ndarray,
) -> jnp.ndarray:
    """
    Apply deflection angles to trace a grid to a new position.

    Parameters
    ----------
    grid : jnp.ndarray
        Grid of (y, x) coordinates with shape (..., 2)
    deflections : jnp.ndarray
        Deflection angles (alpha_y, alpha_x) with shape (..., 2)

    Returns
    -------
    jnp.ndarray
        Traced grid positions
    """
    return grid - deflections


def compute_plane_deflections(
    grid: jnp.ndarray,
    mass_profile_types: Tuple[str, ...],
    mass_params_list: List[dict],
) -> jnp.ndarray:
    """
    Compute total deflection angles from all mass profiles in a plane.

    Parameters
    ----------
    grid : jnp.ndarray
        Grid of (y, x) coordinates
    mass_profile_types : Tuple[str, ...]
        Types of mass profiles in this plane
    mass_params_list : List[dict]
        Parameters for each mass profile

    Returns
    -------
    jnp.ndarray
        Total deflection angles from this plane
    """
    if len(mass_profile_types) == 0:
        return jnp.zeros_like(grid)

    total_deflections = jnp.zeros_like(grid)

    for profile_type, params in zip(mass_profile_types, mass_params_list):
        deflections = mass.evaluate_deflections(profile_type, grid, params)
        total_deflections = total_deflections + deflections

    return total_deflections


def compute_plane_image(
    grid: jnp.ndarray,
    light_profile_types: Tuple[str, ...],
    light_params_list: List[dict],
) -> jnp.ndarray:
    """
    Compute total image from all light profiles in a plane.

    Parameters
    ----------
    grid : jnp.ndarray
        Grid of (y, x) coordinates (already traced to this plane)
    light_profile_types : Tuple[str, ...]
        Types of light profiles in this plane
    light_params_list : List[dict]
        Parameters for each light profile

    Returns
    -------
    jnp.ndarray
        Total surface brightness at each grid point
    """
    if len(light_profile_types) == 0:
        return jnp.zeros(grid.shape[:-1])

    total_image = jnp.zeros(grid.shape[:-1])

    for profile_type, params in zip(light_profile_types, light_params_list):
        image = light.evaluate_light_profile(profile_type, grid, params)
        total_image = total_image + image

    return total_image


def traced_grid_list(
    grid: jnp.ndarray,
    config: TracerConfig,
    params: dict,
    scaling_factors: Optional[jnp.ndarray] = None,
) -> List[jnp.ndarray]:
    """
    Compute the ray-traced grid at each plane in the tracer.

    This implements multi-plane ray-tracing following PyAutoLens.

    Parameters
    ----------
    grid : jnp.ndarray
        Image-plane grid of (y, x) coordinates with shape (N, 2)
    config : TracerConfig
        Static configuration defining the tracer structure
    params : dict
        Nested dictionary of parameters for each plane
        Structure: params['planes'][plane_idx]['mass'][profile_idx]
    scaling_factors : jnp.ndarray, optional
        Pre-computed scaling factors. If None, computed on the fly.
        Shape: (n_planes, n_planes)

    Returns
    -------
    List[jnp.ndarray]
        List of traced grids, one for each plane
    """
    n_planes = len(config.planes)
    redshifts = jnp.array([p.redshift for p in config.planes])

    # Store traced grids and deflections
    traced_grids = []
    deflections_list = []

    for plane_idx in range(n_planes):
        plane_config = config.planes[plane_idx]

        # Start with the original grid
        current_grid = grid.copy()

        # Apply scaled deflections from all previous planes
        if plane_idx > 0:
            for prev_idx in range(plane_idx):
                # Get scaling factor
                if scaling_factors is not None:
                    scale = scaling_factors[prev_idx, plane_idx]
                else:
                    scale = scaling_factor(
                        z_lens=redshifts[prev_idx],
                        z_source=redshifts[plane_idx],
                        z_final=redshifts[-1],
                        cosmo=config.cosmology,
                    )

                # Apply scaled deflections
                scaled_deflections = scale * deflections_list[prev_idx]
                current_grid = current_grid - scaled_deflections

        # Store the traced grid for this plane
        traced_grids.append(current_grid)

        # Compute deflections at this plane (for subsequent planes)
        if plane_idx < n_planes - 1:  # No need for last plane
            mass_types = plane_config.mass_profile_types
            mass_params = params["planes"][plane_idx].get("mass", [])
            deflections = compute_plane_deflections(current_grid, mass_types, mass_params)
            deflections_list.append(deflections)

    return traced_grids


def tracer_image(
    grid: jnp.ndarray,
    config: TracerConfig,
    params: dict,
    scaling_factors: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Compute the total lensed image from all planes.

    Parameters
    ----------
    grid : jnp.ndarray
        Image-plane grid of (y, x) coordinates with shape (N, 2)
    config : TracerConfig
        Static configuration defining the tracer structure
    params : dict
        Nested dictionary of parameters
    scaling_factors : jnp.ndarray, optional
        Pre-computed scaling factors

    Returns
    -------
    jnp.ndarray
        Total surface brightness at each grid point
    """
    # Get traced grids
    traced_grids = traced_grid_list(grid, config, params, scaling_factors)

    # Sum images from all planes
    total_image = jnp.zeros(grid.shape[:-1])

    for plane_idx, plane_config in enumerate(config.planes):
        light_types = plane_config.light_profile_types
        light_params = params["planes"][plane_idx].get("light", [])

        if len(light_types) > 0:
            plane_image = compute_plane_image(
                traced_grids[plane_idx], light_types, light_params
            )
            total_image = total_image + plane_image

    return total_image


def tracer_deflections(
    grid: jnp.ndarray,
    config: TracerConfig,
    params: dict,
    scaling_factors: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Compute the total deflection angles from image plane to source plane.

    Parameters
    ----------
    grid : jnp.ndarray
        Image-plane grid of (y, x) coordinates
    config : TracerConfig
        Static configuration defining the tracer structure
    params : dict
        Nested dictionary of parameters
    scaling_factors : jnp.ndarray, optional
        Pre-computed scaling factors

    Returns
    -------
    jnp.ndarray
        Total deflection angles (difference between image and source positions)
    """
    traced_grids = traced_grid_list(grid, config, params, scaling_factors)

    # Total deflection is the difference between image and source plane positions
    return grid - traced_grids[-1]


def tracer_convergence(
    grid: jnp.ndarray,
    config: TracerConfig,
    params: dict,
) -> jnp.ndarray:
    """
    Compute the total convergence from all mass profiles.

    Note: Convergence is evaluated at the image plane positions, not traced positions.
    This is the standard convention for convergence maps.

    Parameters
    ----------
    grid : jnp.ndarray
        Image-plane grid of (y, x) coordinates
    config : TracerConfig
        Static configuration defining the tracer structure
    params : dict
        Nested dictionary of parameters

    Returns
    -------
    jnp.ndarray
        Total convergence at each grid point
    """
    total_convergence = jnp.zeros(grid.shape[:-1])

    for plane_idx, plane_config in enumerate(config.planes):
        mass_types = plane_config.mass_profile_types
        mass_params = params["planes"][plane_idx].get("mass", [])

        for profile_type, profile_params in zip(mass_types, mass_params):
            kappa = mass.evaluate_convergence(profile_type, grid, profile_params)
            total_convergence = total_convergence + kappa

    return total_convergence


# =============================================================================
# Simplified API for Common Use Cases
# =============================================================================


def simple_tracer_image(
    grid: jnp.ndarray,
    lens_mass_params: dict,
    source_light_params: dict,
    lens_light_params: Optional[dict] = None,
    lens_mass_type: str = "sie",
    source_light_type: str = "sersic",
    lens_light_type: str = "sersic",
    z_lens: float = 0.5,
    z_source: float = 1.0,
) -> jnp.ndarray:
    """
    Simplified interface for a common lens+source system.

    Parameters
    ----------
    grid : jnp.ndarray
        Image-plane grid
    lens_mass_params : dict
        Parameters for the lens mass profile
    source_light_params : dict
        Parameters for the source light profile
    lens_light_params : dict, optional
        Parameters for the lens light profile
    lens_mass_type : str
        Type of lens mass profile
    source_light_type : str
        Type of source light profile
    lens_light_type : str
        Type of lens light profile
    z_lens : float
        Lens redshift
    z_source : float
        Source redshift

    Returns
    -------
    jnp.ndarray
        Total lensed image
    """
    # Build config
    if lens_light_params is not None:
        lens_plane = PlaneConfig(
            redshift=z_lens,
            light_profile_types=(lens_light_type,),
            mass_profile_types=(lens_mass_type,),
        )
    else:
        lens_plane = PlaneConfig(
            redshift=z_lens,
            light_profile_types=(),
            mass_profile_types=(lens_mass_type,),
        )

    source_plane = PlaneConfig(
        redshift=z_source,
        light_profile_types=(source_light_type,),
        mass_profile_types=(),
    )

    config = TracerConfig(planes=(lens_plane, source_plane))

    # Build params
    params = {
        "planes": [
            {
                "mass": [lens_mass_params],
                "light": [lens_light_params] if lens_light_params else [],
            },
            {
                "mass": [],
                "light": [source_light_params],
            },
        ]
    }

    return tracer_image(grid, config, params)


def simple_tracer_deflections(
    grid: jnp.ndarray,
    lens_mass_params: dict,
    lens_mass_type: str = "sie",
    z_lens: float = 0.5,
    z_source: float = 1.0,
) -> jnp.ndarray:
    """
    Simplified interface for computing deflection angles.

    Parameters
    ----------
    grid : jnp.ndarray
        Image-plane grid
    lens_mass_params : dict
        Parameters for the lens mass profile
    lens_mass_type : str
        Type of lens mass profile
    z_lens : float
        Lens redshift
    z_source : float
        Source redshift

    Returns
    -------
    jnp.ndarray
        Deflection angles
    """
    # For single-plane lensing, deflections are just from the mass profile
    return mass.evaluate_deflections(lens_mass_type, grid, lens_mass_params)
