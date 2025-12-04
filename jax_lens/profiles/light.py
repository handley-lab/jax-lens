"""
Functional light profiles for gravitational lensing.

All functions are pure JAX functions that can be differentiated and vmapped.
Grid coordinates are in (y, x) format with units of arcseconds.

Reference: PyAutoLens/autogalaxy light profile implementations
"""

import jax.numpy as jnp
from jax import lax
from typing import Tuple
from functools import partial


def _elliptical_radius(
    grid: jnp.ndarray,
    centre: jnp.ndarray,
    axis_ratio: float,
    angle: float,
) -> jnp.ndarray:
    """
    Compute elliptical radius for each grid point.

    Parameters
    ----------
    grid : jnp.ndarray
        Grid of (y, x) coordinates with shape (..., 2)
    centre : jnp.ndarray
        Centre of the profile (y, x) in arcseconds
    axis_ratio : float
        Ratio of semi-minor to semi-major axis (b/a), 0 < q <= 1
    angle : float
        Position angle in radians, measured counter-clockwise from x-axis

    Returns
    -------
    jnp.ndarray
        Elliptical radius at each grid point
    """
    # Shift to profile centre
    shifted = grid - centre

    # Rotate to align with major axis
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)

    y_rot = shifted[..., 0] * cos_angle - shifted[..., 1] * sin_angle
    x_rot = shifted[..., 0] * sin_angle + shifted[..., 1] * cos_angle

    # Compute elliptical radius: r_ell = sqrt(x^2 + y^2/q^2)
    # Using x along major axis, y along minor axis
    r_ell = jnp.sqrt(x_rot**2 + (y_rot / axis_ratio) ** 2)

    return r_ell


def _sersic_bn(n: float) -> float:
    """
    Compute the Sersic b_n parameter.

    This approximation is accurate to better than 0.1% for n > 0.36.

    Parameters
    ----------
    n : float
        Sersic index

    Returns
    -------
    float
        The b_n parameter
    """
    # Ciotti & Bertin (1999) approximation
    return 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n**2)


def sersic(
    grid: jnp.ndarray,
    centre: jnp.ndarray,
    intensity: float,
    effective_radius: float,
    sersic_index: float,
    axis_ratio: float = 1.0,
    angle: float = 0.0,
) -> jnp.ndarray:
    """
    Evaluate the Sersic light profile.

    I(r) = I_e * exp(-b_n * [(r/r_e)^(1/n) - 1])

    where b_n is chosen so that half the total light is within r_e.

    Parameters
    ----------
    grid : jnp.ndarray
        Grid of (y, x) coordinates with shape (..., 2) in arcseconds
    centre : jnp.ndarray
        Centre of the profile (y, x) in arcseconds
    intensity : float
        Intensity at the effective radius
    effective_radius : float
        Effective (half-light) radius in arcseconds
    sersic_index : float
        Sersic index (n=1 exponential, n=4 de Vaucouleurs)
    axis_ratio : float
        Ratio of semi-minor to semi-major axis (b/a), default 1.0 (circular)
    angle : float
        Position angle in radians, default 0.0

    Returns
    -------
    jnp.ndarray
        Surface brightness at each grid point
    """
    r_ell = _elliptical_radius(grid, centre, axis_ratio, angle)

    # Avoid division by zero for very small radii
    r_ell = jnp.maximum(r_ell, 1e-12)

    bn = _sersic_bn(sersic_index)

    # Sersic profile
    exponent = -bn * ((r_ell / effective_radius) ** (1.0 / sersic_index) - 1.0)

    return intensity * jnp.exp(exponent)


def exponential(
    grid: jnp.ndarray,
    centre: jnp.ndarray,
    intensity: float,
    effective_radius: float,
    axis_ratio: float = 1.0,
    angle: float = 0.0,
) -> jnp.ndarray:
    """
    Evaluate the exponential light profile (Sersic with n=1).

    Parameters
    ----------
    grid : jnp.ndarray
        Grid of (y, x) coordinates with shape (..., 2) in arcseconds
    centre : jnp.ndarray
        Centre of the profile (y, x) in arcseconds
    intensity : float
        Intensity at the effective radius
    effective_radius : float
        Effective (half-light) radius in arcseconds
    axis_ratio : float
        Ratio of semi-minor to semi-major axis (b/a), default 1.0
    angle : float
        Position angle in radians, default 0.0

    Returns
    -------
    jnp.ndarray
        Surface brightness at each grid point
    """
    return sersic(
        grid=grid,
        centre=centre,
        intensity=intensity,
        effective_radius=effective_radius,
        sersic_index=1.0,
        axis_ratio=axis_ratio,
        angle=angle,
    )


def dev_vaucouleurs(
    grid: jnp.ndarray,
    centre: jnp.ndarray,
    intensity: float,
    effective_radius: float,
    axis_ratio: float = 1.0,
    angle: float = 0.0,
) -> jnp.ndarray:
    """
    Evaluate the de Vaucouleurs light profile (Sersic with n=4).

    Parameters
    ----------
    grid : jnp.ndarray
        Grid of (y, x) coordinates with shape (..., 2) in arcseconds
    centre : jnp.ndarray
        Centre of the profile (y, x) in arcseconds
    intensity : float
        Intensity at the effective radius
    effective_radius : float
        Effective (half-light) radius in arcseconds
    axis_ratio : float
        Ratio of semi-minor to semi-major axis (b/a), default 1.0
    angle : float
        Position angle in radians, default 0.0

    Returns
    -------
    jnp.ndarray
        Surface brightness at each grid point
    """
    return sersic(
        grid=grid,
        centre=centre,
        intensity=intensity,
        effective_radius=effective_radius,
        sersic_index=4.0,
        axis_ratio=axis_ratio,
        angle=angle,
    )


def gaussian(
    grid: jnp.ndarray,
    centre: jnp.ndarray,
    intensity: float,
    sigma: float,
    axis_ratio: float = 1.0,
    angle: float = 0.0,
) -> jnp.ndarray:
    """
    Evaluate a 2D Gaussian light profile.

    I(r) = I_0 * exp(-r^2 / (2 * sigma^2))

    Parameters
    ----------
    grid : jnp.ndarray
        Grid of (y, x) coordinates with shape (..., 2) in arcseconds
    centre : jnp.ndarray
        Centre of the profile (y, x) in arcseconds
    intensity : float
        Peak intensity at the centre
    sigma : float
        Standard deviation (width) of the Gaussian in arcseconds
    axis_ratio : float
        Ratio of semi-minor to semi-major axis (b/a), default 1.0
    angle : float
        Position angle in radians, default 0.0

    Returns
    -------
    jnp.ndarray
        Surface brightness at each grid point
    """
    r_ell = _elliptical_radius(grid, centre, axis_ratio, angle)

    return intensity * jnp.exp(-0.5 * (r_ell / sigma) ** 2)


def evaluate_light_profile(
    profile_type: str,
    grid: jnp.ndarray,
    params: dict,
) -> jnp.ndarray:
    """
    Evaluate a light profile given its type and parameters.

    This is a dispatcher function for use in the pipeline.

    Parameters
    ----------
    profile_type : str
        One of 'sersic', 'exponential', 'dev_vaucouleurs', 'gaussian'
    grid : jnp.ndarray
        Grid of (y, x) coordinates
    params : dict
        Dictionary of profile parameters

    Returns
    -------
    jnp.ndarray
        Surface brightness at each grid point
    """
    if profile_type == "sersic":
        return sersic(
            grid=grid,
            centre=params["centre"],
            intensity=params["intensity"],
            effective_radius=params["effective_radius"],
            sersic_index=params["sersic_index"],
            axis_ratio=params.get("axis_ratio", 1.0),
            angle=params.get("angle", 0.0),
        )
    elif profile_type == "exponential":
        return exponential(
            grid=grid,
            centre=params["centre"],
            intensity=params["intensity"],
            effective_radius=params["effective_radius"],
            axis_ratio=params.get("axis_ratio", 1.0),
            angle=params.get("angle", 0.0),
        )
    elif profile_type == "dev_vaucouleurs":
        return dev_vaucouleurs(
            grid=grid,
            centre=params["centre"],
            intensity=params["intensity"],
            effective_radius=params["effective_radius"],
            axis_ratio=params.get("axis_ratio", 1.0),
            angle=params.get("angle", 0.0),
        )
    elif profile_type == "gaussian":
        return gaussian(
            grid=grid,
            centre=params["centre"],
            intensity=params["intensity"],
            sigma=params["sigma"],
            axis_ratio=params.get("axis_ratio", 1.0),
            angle=params.get("angle", 0.0),
        )
    else:
        raise ValueError(f"Unknown light profile type: {profile_type}")
