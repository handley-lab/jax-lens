"""
Functional mass profiles for gravitational lensing.

All functions compute deflection angles and convergence in a pure, JAX-compatible way.
Grid coordinates are in (y, x) format with units of arcseconds.
Deflection angles are returned as (alpha_y, alpha_x) in arcseconds.

Reference: PyAutoLens/autogalaxy mass profile implementations
"""

import jax.numpy as jnp
from jax import lax
from typing import Tuple


def _elliptical_coords(
    grid: jnp.ndarray,
    centre: jnp.ndarray,
    axis_ratio: float,
    angle: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Transform grid coordinates to elliptical coordinates aligned with the profile.

    Parameters
    ----------
    grid : jnp.ndarray
        Grid of (y, x) coordinates with shape (..., 2)
    centre : jnp.ndarray
        Centre of the profile (y, x) in arcseconds
    axis_ratio : float
        Ratio of semi-minor to semi-major axis (b/a)
    angle : float
        Position angle in radians

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        (eta, phi) elliptical coordinates where eta is the elliptical radius
    """
    # Shift to profile centre
    shifted = grid - centre

    # Rotate to align with major axis
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)

    y_rot = shifted[..., 0] * cos_angle - shifted[..., 1] * sin_angle
    x_rot = shifted[..., 0] * sin_angle + shifted[..., 1] * cos_angle

    return y_rot, x_rot


# =============================================================================
# Singular Isothermal Sphere (SIS)
# =============================================================================


def sis_deflections(
    grid: jnp.ndarray,
    centre: jnp.ndarray,
    einstein_radius: float,
) -> jnp.ndarray:
    """
    Compute deflection angles for a Singular Isothermal Sphere (SIS).

    The SIS has constant deflection magnitude equal to the Einstein radius:
    |alpha| = theta_E

    The deflection points radially toward the centre.

    Parameters
    ----------
    grid : jnp.ndarray
        Grid of (y, x) coordinates with shape (..., 2) in arcseconds
    centre : jnp.ndarray
        Centre of the mass profile (y, x) in arcseconds
    einstein_radius : float
        Einstein radius in arcseconds

    Returns
    -------
    jnp.ndarray
        Deflection angles (alpha_y, alpha_x) at each grid point, shape (..., 2)
    """
    # Vector from centre to grid point
    shifted = grid - centre

    # Distance from centre
    r = jnp.sqrt(shifted[..., 0] ** 2 + shifted[..., 1] ** 2)
    r = jnp.maximum(r, 1e-12)  # Avoid division by zero

    # Unit vector pointing radially outward
    unit_y = shifted[..., 0] / r
    unit_x = shifted[..., 1] / r

    # Deflection has constant magnitude, points toward centre
    alpha_y = einstein_radius * unit_y
    alpha_x = einstein_radius * unit_x

    return jnp.stack([alpha_y, alpha_x], axis=-1)


def sis_convergence(
    grid: jnp.ndarray,
    centre: jnp.ndarray,
    einstein_radius: float,
) -> jnp.ndarray:
    """
    Compute convergence (dimensionless surface mass density) for a SIS.

    kappa(r) = theta_E / (2 * r)

    Parameters
    ----------
    grid : jnp.ndarray
        Grid of (y, x) coordinates with shape (..., 2) in arcseconds
    centre : jnp.ndarray
        Centre of the mass profile (y, x) in arcseconds
    einstein_radius : float
        Einstein radius in arcseconds

    Returns
    -------
    jnp.ndarray
        Convergence at each grid point
    """
    shifted = grid - centre
    r = jnp.sqrt(shifted[..., 0] ** 2 + shifted[..., 1] ** 2)
    r = jnp.maximum(r, 1e-12)

    return einstein_radius / (2.0 * r)


def sis_potential(
    grid: jnp.ndarray,
    centre: jnp.ndarray,
    einstein_radius: float,
) -> jnp.ndarray:
    """
    Compute the lensing potential for a SIS.

    psi(r) = theta_E * r

    Parameters
    ----------
    grid : jnp.ndarray
        Grid of (y, x) coordinates with shape (..., 2) in arcseconds
    centre : jnp.ndarray
        Centre of the mass profile (y, x) in arcseconds
    einstein_radius : float
        Einstein radius in arcseconds

    Returns
    -------
    jnp.ndarray
        Potential at each grid point
    """
    shifted = grid - centre
    r = jnp.sqrt(shifted[..., 0] ** 2 + shifted[..., 1] ** 2)

    return einstein_radius * r


# =============================================================================
# Singular Isothermal Ellipsoid (SIE)
# =============================================================================


def sie_deflections(
    grid: jnp.ndarray,
    centre: jnp.ndarray,
    einstein_radius: float,
    axis_ratio: float,
    angle: float = 0.0,
) -> jnp.ndarray:
    """
    Compute deflection angles for a Singular Isothermal Ellipsoid (SIE).

    Uses the Kormann, Schneider & Bartelmann (1994) formulation.

    Parameters
    ----------
    grid : jnp.ndarray
        Grid of (y, x) coordinates with shape (..., 2) in arcseconds
    centre : jnp.ndarray
        Centre of the mass profile (y, x) in arcseconds
    einstein_radius : float
        Einstein radius in arcseconds (defined as the circularized Einstein radius)
    axis_ratio : float
        Ratio of semi-minor to semi-major axis (b/a), 0 < q <= 1
    angle : float
        Position angle in radians, default 0.0

    Returns
    -------
    jnp.ndarray
        Deflection angles (alpha_y, alpha_x) at each grid point, shape (..., 2)
    """
    # Get rotated coordinates
    y_rot, x_rot = _elliptical_coords(grid, centre, axis_ratio, angle)

    # Ellipticity factor
    q = axis_ratio
    f = jnp.sqrt(1.0 - q**2)

    # Elliptical radius (avoid zero)
    psi = jnp.sqrt(q**2 * x_rot**2 + y_rot**2)
    psi = jnp.maximum(psi, 1e-12)

    # Deflection in rotated frame (Kormann et al. 1994)
    # For q < 1, we have deflections that depend on arctan
    # Handle the circular case (q=1) separately to avoid division by zero

    # Factor for deflection magnitude
    factor = einstein_radius * jnp.sqrt(q) / f

    # Deflections in rotated frame
    alpha_x_rot = factor * jnp.arctan(f * x_rot / psi)
    alpha_y_rot = factor * jnp.arctanh(f * y_rot / psi)

    # Handle near-circular case (q close to 1)
    is_circular = q > 0.9999
    r = jnp.sqrt(x_rot**2 + y_rot**2)
    r = jnp.maximum(r, 1e-12)

    alpha_x_rot_circ = einstein_radius * x_rot / r
    alpha_y_rot_circ = einstein_radius * y_rot / r

    alpha_x_rot = jnp.where(is_circular, alpha_x_rot_circ, alpha_x_rot)
    alpha_y_rot = jnp.where(is_circular, alpha_y_rot_circ, alpha_y_rot)

    # Rotate back to original frame
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)

    alpha_y = alpha_y_rot * cos_angle + alpha_x_rot * sin_angle
    alpha_x = -alpha_y_rot * sin_angle + alpha_x_rot * cos_angle

    return jnp.stack([alpha_y, alpha_x], axis=-1)


def sie_convergence(
    grid: jnp.ndarray,
    centre: jnp.ndarray,
    einstein_radius: float,
    axis_ratio: float,
    angle: float = 0.0,
) -> jnp.ndarray:
    """
    Compute convergence for a Singular Isothermal Ellipsoid (SIE).

    kappa = theta_E * sqrt(q) / (2 * sqrt(q^2 * x^2 + y^2))

    Parameters
    ----------
    grid : jnp.ndarray
        Grid of (y, x) coordinates with shape (..., 2) in arcseconds
    centre : jnp.ndarray
        Centre of the mass profile (y, x) in arcseconds
    einstein_radius : float
        Einstein radius in arcseconds
    axis_ratio : float
        Ratio of semi-minor to semi-major axis (b/a)
    angle : float
        Position angle in radians, default 0.0

    Returns
    -------
    jnp.ndarray
        Convergence at each grid point
    """
    y_rot, x_rot = _elliptical_coords(grid, centre, axis_ratio, angle)

    q = axis_ratio
    psi = jnp.sqrt(q**2 * x_rot**2 + y_rot**2)
    psi = jnp.maximum(psi, 1e-12)

    return einstein_radius * jnp.sqrt(q) / (2.0 * psi)


# =============================================================================
# External Shear
# =============================================================================


def external_shear_deflections(
    grid: jnp.ndarray,
    gamma_1: float,
    gamma_2: float,
) -> jnp.ndarray:
    """
    Compute deflection angles from external shear.

    External shear accounts for tidal forces from nearby massive structures
    (galaxy clusters, neighboring galaxies, etc.) that are not explicitly modeled.

    The shear is parameterized by two components:
    - gamma_1: Shear along the x-axis (stretches along x, compresses along y)
    - gamma_2: Shear at 45 degrees

    Parameters
    ----------
    grid : jnp.ndarray
        Grid of (y, x) coordinates with shape (..., 2) in arcseconds
    gamma_1 : float
        First shear component (dimensionless)
    gamma_2 : float
        Second shear component (dimensionless)

    Returns
    -------
    jnp.ndarray
        Deflection angles (alpha_y, alpha_x) at each grid point, shape (..., 2)
    """
    y = grid[..., 0]
    x = grid[..., 1]

    # Shear deflection angles from potential psi = (gamma_1/2)(x^2 - y^2) + gamma_2 * x * y
    # Taking gradient: alpha = nabla(psi)
    # alpha_x = d(psi)/dx = gamma_1 * x + gamma_2 * y
    # alpha_y = d(psi)/dy = -gamma_1 * y + gamma_2 * x
    # This ensures div(alpha) = 0 (pure shear, no convergence)
    alpha_y = -gamma_1 * y + gamma_2 * x
    alpha_x = gamma_1 * x + gamma_2 * y

    return jnp.stack([alpha_y, alpha_x], axis=-1)


def external_shear_convergence(
    grid: jnp.ndarray,
    gamma_1: float,
    gamma_2: float,
) -> jnp.ndarray:
    """
    Compute convergence for external shear.

    External shear has zero convergence (it's a pure shear field).

    Parameters
    ----------
    grid : jnp.ndarray
        Grid of (y, x) coordinates with shape (..., 2) in arcseconds
    gamma_1 : float
        First shear component (dimensionless)
    gamma_2 : float
        Second shear component (dimensionless)

    Returns
    -------
    jnp.ndarray
        Convergence at each grid point (all zeros)
    """
    return jnp.zeros(grid.shape[:-1])


# =============================================================================
# Navarro-Frenk-White (NFW) Profile
# =============================================================================


def _nfw_func(x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the NFW lensing function F(x) where x = r/r_s.

    F(x) = 1 - 2/sqrt(1-x^2) * arctanh(sqrt((1-x)/(1+x)))  for x < 1
    F(x) = 1 - 2/sqrt(x^2-1) * arctan(sqrt((x-1)/(1+x)))   for x > 1
    F(x) = 0                                                for x = 1
    """
    x_safe = jnp.maximum(x, 1e-12)

    # x < 1 case
    sqrt_term_lt = jnp.sqrt((1.0 - x_safe) / (1.0 + x_safe))
    f_lt = 1.0 - 2.0 / jnp.sqrt(1.0 - x_safe**2) * jnp.arctanh(sqrt_term_lt)

    # x > 1 case
    sqrt_term_gt = jnp.sqrt((x_safe - 1.0) / (1.0 + x_safe))
    f_gt = 1.0 - 2.0 / jnp.sqrt(x_safe**2 - 1.0) * jnp.arctan(sqrt_term_gt)

    # x = 1 case
    f_eq = jnp.zeros_like(x_safe)

    # Select appropriate case
    result = jnp.where(x_safe < 0.999, f_lt, jnp.where(x_safe > 1.001, f_gt, f_eq))

    return result


def _nfw_g_func(x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the NFW g(x) function for deflection angles.

    g(x) = ln(x/2) + F(x)
    """
    x_safe = jnp.maximum(x, 1e-12)
    return jnp.log(x_safe / 2.0) + _nfw_func(x_safe)


def nfw_deflections(
    grid: jnp.ndarray,
    centre: jnp.ndarray,
    kappa_s: float,
    scale_radius: float,
    axis_ratio: float = 1.0,
    angle: float = 0.0,
) -> jnp.ndarray:
    """
    Compute deflection angles for a spherical NFW profile.

    Parameters
    ----------
    grid : jnp.ndarray
        Grid of (y, x) coordinates with shape (..., 2) in arcseconds
    centre : jnp.ndarray
        Centre of the mass profile (y, x) in arcseconds
    kappa_s : float
        Characteristic convergence (dimensionless surface density at r_s)
    scale_radius : float
        Scale radius r_s in arcseconds
    axis_ratio : float
        Ratio of semi-minor to semi-major axis (b/a), default 1.0 (spherical)
    angle : float
        Position angle in radians, default 0.0

    Returns
    -------
    jnp.ndarray
        Deflection angles (alpha_y, alpha_x) at each grid point, shape (..., 2)
    """
    # For simplicity, implementing spherical NFW first
    # Elliptical NFW requires numerical integration

    shifted = grid - centre
    r = jnp.sqrt(shifted[..., 0] ** 2 + shifted[..., 1] ** 2)
    r = jnp.maximum(r, 1e-12)

    x = r / scale_radius

    # Deflection magnitude: alpha = 4 * kappa_s * r_s * g(x) / x
    g_x = _nfw_g_func(x)
    alpha_mag = 4.0 * kappa_s * scale_radius * g_x / x

    # Unit vector
    unit_y = shifted[..., 0] / r
    unit_x = shifted[..., 1] / r

    alpha_y = alpha_mag * unit_y
    alpha_x = alpha_mag * unit_x

    return jnp.stack([alpha_y, alpha_x], axis=-1)


def _nfw_convergence_func(x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the NFW convergence function f(x) where kappa = 2 * kappa_s * f(x).

    This is a dimensionless function that gives the convergence profile shape.

    The formula is:
        f(x) = 1/(x^2 - 1) * [1 - 2/sqrt(1-x^2) * arctanh(sqrt((1-x)/(1+x)))]  for x < 1
        f(x) = 1/3                                                               for x = 1
        f(x) = 1/(x^2 - 1) * [1 - 2/sqrt(x^2-1) * arctan(sqrt((x-1)/(1+x)))]    for x > 1
    """
    x_safe = jnp.maximum(x, 1e-12)

    # For x < 1: both (x^2 - 1) and the bracket term are negative, product is positive
    sqrt_1mx2_lt = jnp.sqrt(jnp.maximum(1.0 - x_safe**2, 1e-12))
    sqrt_term_lt = jnp.sqrt(jnp.maximum((1.0 - x_safe) / (1.0 + x_safe), 0.0))
    bracket_lt = 1.0 - 2.0 / sqrt_1mx2_lt * jnp.arctanh(jnp.minimum(sqrt_term_lt, 0.99999))
    # Use (x^2 - 1) which is negative for x < 1
    f_lt = bracket_lt / (x_safe**2 - 1.0)

    # For x > 1: (x^2 - 1) is positive, bracket term is also positive
    sqrt_x2m1_gt = jnp.sqrt(jnp.maximum(x_safe**2 - 1.0, 1e-12))
    sqrt_term_gt = jnp.sqrt(jnp.maximum((x_safe - 1.0) / (1.0 + x_safe), 0.0))
    bracket_gt = 1.0 - 2.0 / sqrt_x2m1_gt * jnp.arctan(sqrt_term_gt)
    f_gt = bracket_gt / (x_safe**2 - 1.0)

    # For x = 1, f(1) = 1/3
    f_eq = jnp.ones_like(x_safe) / 3.0

    # Select appropriate case
    result = jnp.where(x_safe < 0.999, f_lt, jnp.where(x_safe > 1.001, f_gt, f_eq))

    return result


def nfw_convergence(
    grid: jnp.ndarray,
    centre: jnp.ndarray,
    kappa_s: float,
    scale_radius: float,
    axis_ratio: float = 1.0,
    angle: float = 0.0,
) -> jnp.ndarray:
    """
    Compute convergence for a spherical NFW profile.

    kappa(x) = 2 * kappa_s * f(x)

    where x = r / r_s and f(x) is the NFW dimensionless convergence function.

    Parameters
    ----------
    grid : jnp.ndarray
        Grid of (y, x) coordinates with shape (..., 2) in arcseconds
    centre : jnp.ndarray
        Centre of the mass profile (y, x) in arcseconds
    kappa_s : float
        Characteristic convergence
    scale_radius : float
        Scale radius r_s in arcseconds
    axis_ratio : float
        Ratio of semi-minor to semi-major axis (b/a), default 1.0
    angle : float
        Position angle in radians, default 0.0

    Returns
    -------
    jnp.ndarray
        Convergence at each grid point
    """
    shifted = grid - centre
    r = jnp.sqrt(shifted[..., 0] ** 2 + shifted[..., 1] ** 2)
    r = jnp.maximum(r, 1e-12)

    x = r / scale_radius

    return 2.0 * kappa_s * _nfw_convergence_func(x)


# =============================================================================
# Power Law Profiles
# =============================================================================


def spherical_power_law_deflections(
    grid: jnp.ndarray,
    centre: jnp.ndarray,
    einstein_radius: float,
    slope: float,
) -> jnp.ndarray:
    """
    Compute deflection angles for a SPHERICAL power-law mass profile.

    Note: This is a spherical approximation. For elliptical power-law profiles,
    complex numerical integration is required (not yet implemented).

    For slope = 2.0, this reduces to SIS (not SIE).

    Parameters
    ----------
    grid : jnp.ndarray
        Grid of (y, x) coordinates with shape (..., 2) in arcseconds
    centre : jnp.ndarray
        Centre of the mass profile (y, x) in arcseconds
    einstein_radius : float
        Einstein radius in arcseconds
    slope : float
        Power law slope (gamma), typical range 1.5-2.5

    Returns
    -------
    jnp.ndarray
        Deflection angles (alpha_y, alpha_x) at each grid point, shape (..., 2)
    """
    shifted = grid - centre
    r = jnp.sqrt(shifted[..., 0] ** 2 + shifted[..., 1] ** 2)
    r = jnp.maximum(r, 1e-12)

    # Deflection for spherical power law
    # alpha = theta_E * (theta_E / r)^(gamma - 2)
    alpha_mag = einstein_radius * (einstein_radius / r) ** (slope - 2.0)

    unit_y = shifted[..., 0] / r
    unit_x = shifted[..., 1] / r

    alpha_y = alpha_mag * unit_y
    alpha_x = alpha_mag * unit_x

    return jnp.stack([alpha_y, alpha_x], axis=-1)


# =============================================================================
# Dispatcher Function
# =============================================================================


def evaluate_deflections(
    profile_type: str,
    grid: jnp.ndarray,
    params: dict,
) -> jnp.ndarray:
    """
    Evaluate deflection angles for a mass profile given its type and parameters.

    Parameters
    ----------
    profile_type : str
        One of 'sis', 'sie', 'nfw', 'spherical_power_law', 'shear'
    grid : jnp.ndarray
        Grid of (y, x) coordinates
    params : dict
        Dictionary of profile parameters

    Returns
    -------
    jnp.ndarray
        Deflection angles at each grid point
    """
    if profile_type == "sis":
        return sis_deflections(
            grid=grid,
            centre=params["centre"],
            einstein_radius=params["einstein_radius"],
        )
    elif profile_type == "sie":
        return sie_deflections(
            grid=grid,
            centre=params["centre"],
            einstein_radius=params["einstein_radius"],
            axis_ratio=params.get("axis_ratio", 1.0),
            angle=params.get("angle", 0.0),
        )
    elif profile_type == "nfw":
        return nfw_deflections(
            grid=grid,
            centre=params["centre"],
            kappa_s=params["kappa_s"],
            scale_radius=params["scale_radius"],
            axis_ratio=params.get("axis_ratio", 1.0),
            angle=params.get("angle", 0.0),
        )
    elif profile_type == "spherical_power_law":
        return spherical_power_law_deflections(
            grid=grid,
            centre=params["centre"],
            einstein_radius=params["einstein_radius"],
            slope=params["slope"],
        )
    elif profile_type == "shear":
        return external_shear_deflections(
            grid=grid,
            gamma_1=params["gamma_1"],
            gamma_2=params["gamma_2"],
        )
    else:
        raise ValueError(f"Unknown mass profile type: {profile_type}")


def evaluate_convergence(
    profile_type: str,
    grid: jnp.ndarray,
    params: dict,
) -> jnp.ndarray:
    """
    Evaluate convergence for a mass profile given its type and parameters.

    Parameters
    ----------
    profile_type : str
        One of 'sis', 'sie', 'nfw', 'shear'
    grid : jnp.ndarray
        Grid of (y, x) coordinates
    params : dict
        Dictionary of profile parameters

    Returns
    -------
    jnp.ndarray
        Convergence at each grid point
    """
    if profile_type == "sis":
        return sis_convergence(
            grid=grid,
            centre=params["centre"],
            einstein_radius=params["einstein_radius"],
        )
    elif profile_type == "sie":
        return sie_convergence(
            grid=grid,
            centre=params["centre"],
            einstein_radius=params["einstein_radius"],
            axis_ratio=params.get("axis_ratio", 1.0),
            angle=params.get("angle", 0.0),
        )
    elif profile_type == "nfw":
        return nfw_convergence(
            grid=grid,
            centre=params["centre"],
            kappa_s=params["kappa_s"],
            scale_radius=params["scale_radius"],
            axis_ratio=params.get("axis_ratio", 1.0),
            angle=params.get("angle", 0.0),
        )
    elif profile_type == "shear":
        return external_shear_convergence(
            grid=grid,
            gamma_1=params["gamma_1"],
            gamma_2=params["gamma_2"],
        )
    else:
        raise ValueError(f"Unknown mass profile type: {profile_type}")
