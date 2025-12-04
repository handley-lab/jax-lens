"""
Cosmological distance calculations for gravitational lensing.

Implements flat ΛCDM cosmology with configurable parameters.
Default values are Planck 2015 cosmology.

All functions are pure JAX functions that can be differentiated and vmapped.

Reference: PyAutoLens/autogalaxy cosmology implementations
"""

import jax.numpy as jnp
from jax import lax
from jax.scipy.integrate import trapezoid
from typing import NamedTuple, Optional


class Cosmology(NamedTuple):
    """Cosmological parameters for a flat ΛCDM universe."""

    H0: float = 67.74  # Hubble constant in km/s/Mpc
    Om0: float = 0.3089  # Matter density parameter
    Ob0: float = 0.0486  # Baryon density parameter
    Tcmb0: float = 2.725  # CMB temperature in K


# Default Planck 2015 cosmology
PLANCK15 = Cosmology()


def _hubble_parameter(z: float, cosmo: Cosmology = PLANCK15) -> float:
    """
    Compute the Hubble parameter H(z) at redshift z.

    For flat ΛCDM: H(z) = H0 * sqrt(Om0 * (1+z)^3 + (1 - Om0))

    Parameters
    ----------
    z : float
        Redshift
    cosmo : Cosmology
        Cosmological parameters

    Returns
    -------
    float
        H(z) in km/s/Mpc
    """
    Ol0 = 1.0 - cosmo.Om0  # Dark energy density (flat universe)
    return cosmo.H0 * jnp.sqrt(cosmo.Om0 * (1 + z) ** 3 + Ol0)


def _comoving_distance_integrand(z: float, cosmo: Cosmology = PLANCK15) -> float:
    """Integrand for comoving distance: c / H(z)."""
    c_km_s = 299792.458  # Speed of light in km/s
    return c_km_s / _hubble_parameter(z, cosmo)


def comoving_distance(
    z: float,
    cosmo: Cosmology = PLANCK15,
    n_steps: int = 100,
) -> float:
    """
    Compute the comoving distance to redshift z.

    D_c(z) = c * integral_0^z dz' / H(z')

    Parameters
    ----------
    z : float
        Redshift
    cosmo : Cosmology
        Cosmological parameters
    n_steps : int
        Number of integration steps

    Returns
    -------
    float
        Comoving distance in Mpc
    """
    # Simple trapezoidal integration
    z_arr = jnp.linspace(0.0, z, n_steps)
    integrand = jnp.vectorize(lambda zi: _comoving_distance_integrand(zi, cosmo))(z_arr)

    return trapezoid(integrand, z_arr)


def angular_diameter_distance(
    z: float,
    cosmo: Cosmology = PLANCK15,
    n_steps: int = 100,
) -> float:
    """
    Compute the angular diameter distance to redshift z.

    D_A(z) = D_c(z) / (1 + z)

    Parameters
    ----------
    z : float
        Redshift
    cosmo : Cosmology
        Cosmological parameters
    n_steps : int
        Number of integration steps

    Returns
    -------
    float
        Angular diameter distance in Mpc
    """
    d_c = comoving_distance(z, cosmo, n_steps)
    return d_c / (1.0 + z)


def angular_diameter_distance_z1z2(
    z1: float,
    z2: float,
    cosmo: Cosmology = PLANCK15,
    n_steps: int = 100,
) -> float:
    """
    Compute the angular diameter distance between two redshifts z1 and z2 (z1 < z2).

    For a flat universe:
    D_A(z1, z2) = D_c(z2) / (1 + z2) - D_c(z1) / (1 + z2)
                = (D_c(z2) - D_c(z1)) / (1 + z2)

    Parameters
    ----------
    z1 : float
        Lower redshift
    z2 : float
        Higher redshift
    cosmo : Cosmology
        Cosmological parameters
    n_steps : int
        Number of integration steps

    Returns
    -------
    float
        Angular diameter distance between z1 and z2 in Mpc
    """
    d_c1 = comoving_distance(z1, cosmo, n_steps)
    d_c2 = comoving_distance(z2, cosmo, n_steps)

    return (d_c2 - d_c1) / (1.0 + z2)


def scaling_factor(
    z_lens: float,
    z_source: float,
    z_final: float,
    cosmo: Cosmology = PLANCK15,
    n_steps: int = 100,
) -> float:
    """
    Compute the multi-plane lensing scaling factor.

    This factor scales deflection angles when ray-tracing through multiple planes.

    scaling = D_ls / D_s * D_final / D_l_final

    where:
    - D_ls = angular diameter distance from lens to source
    - D_s = angular diameter distance to source
    - D_final = angular diameter distance to final plane
    - D_l_final = angular diameter distance from lens to final plane

    For single-plane lensing (z_source = z_final), this simplifies to D_ls / D_s.

    Parameters
    ----------
    z_lens : float
        Redshift of the lens plane
    z_source : float
        Redshift of the source plane being deflected to
    z_final : float
        Redshift of the final (most distant) plane
    cosmo : Cosmology
        Cosmological parameters
    n_steps : int
        Number of integration steps

    Returns
    -------
    float
        Scaling factor for deflection angles
    """
    d_ls = angular_diameter_distance_z1z2(z_lens, z_source, cosmo, n_steps)
    d_s = angular_diameter_distance(z_source, cosmo, n_steps)

    # Avoid division by zero
    d_s = jnp.maximum(d_s, 1e-12)

    # For single-plane, z_source = z_final
    is_single_plane = jnp.abs(z_source - z_final) < 1e-8

    # Multi-plane scaling
    d_final = angular_diameter_distance(z_final, cosmo, n_steps)
    d_l_final = angular_diameter_distance_z1z2(z_lens, z_final, cosmo, n_steps)
    d_l_final = jnp.maximum(d_l_final, 1e-12)

    multi_plane_factor = (d_ls / d_s) * (d_final / d_l_final)
    single_plane_factor = d_ls / d_s

    return jnp.where(is_single_plane, single_plane_factor, multi_plane_factor)


def critical_surface_density(
    z_lens: float,
    z_source: float,
    cosmo: Cosmology = PLANCK15,
    n_steps: int = 100,
) -> float:
    """
    Compute the critical surface density for lensing.

    Sigma_cr = c^2 / (4 * pi * G) * D_s / (D_l * D_ls)

    Parameters
    ----------
    z_lens : float
        Redshift of the lens
    z_source : float
        Redshift of the source
    cosmo : Cosmology
        Cosmological parameters
    n_steps : int
        Number of integration steps

    Returns
    -------
    float
        Critical surface density in M_sun / Mpc^2
    """
    # Constants
    c = 299792.458  # km/s
    G = 4.302e-9  # Mpc * (km/s)^2 / M_sun

    d_l = angular_diameter_distance(z_lens, cosmo, n_steps)
    d_s = angular_diameter_distance(z_source, cosmo, n_steps)
    d_ls = angular_diameter_distance_z1z2(z_lens, z_source, cosmo, n_steps)

    # Avoid division by zero
    d_l = jnp.maximum(d_l, 1e-12)
    d_ls = jnp.maximum(d_ls, 1e-12)

    return c**2 / (4.0 * jnp.pi * G) * d_s / (d_l * d_ls)


def time_delay_distance(
    z_lens: float,
    z_source: float,
    cosmo: Cosmology = PLANCK15,
    n_steps: int = 100,
) -> float:
    """
    Compute the time delay distance.

    D_dt = (1 + z_l) * D_l * D_s / D_ls

    Parameters
    ----------
    z_lens : float
        Redshift of the lens
    z_source : float
        Redshift of the source
    cosmo : Cosmology
        Cosmological parameters
    n_steps : int
        Number of integration steps

    Returns
    -------
    float
        Time delay distance in Mpc
    """
    d_l = angular_diameter_distance(z_lens, cosmo, n_steps)
    d_s = angular_diameter_distance(z_source, cosmo, n_steps)
    d_ls = angular_diameter_distance_z1z2(z_lens, z_source, cosmo, n_steps)

    d_ls = jnp.maximum(d_ls, 1e-12)

    return (1.0 + z_lens) * d_l * d_s / d_ls


# =============================================================================
# Pre-computed cosmology for efficiency
# =============================================================================


def precompute_distances(
    redshifts: jnp.ndarray,
    cosmo: Cosmology = PLANCK15,
    n_steps: int = 100,
) -> dict:
    """
    Pre-compute angular diameter distances for a set of redshifts.

    This is useful for efficiency when the same redshifts are used repeatedly.

    Parameters
    ----------
    redshifts : jnp.ndarray
        Array of redshifts
    cosmo : Cosmology
        Cosmological parameters
    n_steps : int
        Number of integration steps

    Returns
    -------
    dict
        Dictionary containing:
        - 'd_a': Angular diameter distances to each redshift
        - 'd_a_matrix': Matrix of D_A(z_i, z_j) for all pairs
        - 'redshifts': The input redshifts
    """
    n_z = len(redshifts)

    # Compute D_A to each redshift
    d_a = jnp.array(
        [angular_diameter_distance(z, cosmo, n_steps) for z in redshifts]
    )

    # Compute D_A(z_i, z_j) matrix
    d_a_matrix = jnp.zeros((n_z, n_z))
    for i in range(n_z):
        for j in range(i + 1, n_z):
            d_ij = angular_diameter_distance_z1z2(
                redshifts[i], redshifts[j], cosmo, n_steps
            )
            d_a_matrix = d_a_matrix.at[i, j].set(d_ij)

    return {
        "d_a": d_a,
        "d_a_matrix": d_a_matrix,
        "redshifts": redshifts,
    }


def scaling_factor_from_precomputed(
    lens_idx: int,
    source_idx: int,
    final_idx: int,
    precomputed: dict,
) -> float:
    """
    Compute scaling factor using pre-computed distances.

    Parameters
    ----------
    lens_idx : int
        Index of lens plane in precomputed redshifts
    source_idx : int
        Index of source plane in precomputed redshifts
    final_idx : int
        Index of final plane in precomputed redshifts
    precomputed : dict
        Pre-computed distances from precompute_distances()

    Returns
    -------
    float
        Scaling factor
    """
    d_ls = precomputed["d_a_matrix"][lens_idx, source_idx]
    d_s = precomputed["d_a"][source_idx]

    d_s = jnp.maximum(d_s, 1e-12)

    is_single_plane = source_idx == final_idx

    d_final = precomputed["d_a"][final_idx]
    d_l_final = precomputed["d_a_matrix"][lens_idx, final_idx]
    d_l_final = jnp.maximum(d_l_final, 1e-12)

    multi_plane_factor = (d_ls / d_s) * (d_final / d_l_final)
    single_plane_factor = d_ls / d_s

    return jnp.where(is_single_plane, single_plane_factor, multi_plane_factor)
