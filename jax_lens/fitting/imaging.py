"""
Functional fitting for imaging data.

All functions are pure JAX functions that can be differentiated and vmapped.

Reference: PyAutoLens FitImaging class
"""

import jax.numpy as jnp
from jax import lax
from jax.scipy.signal import convolve2d
from typing import NamedTuple, Optional, Tuple

from jax_lens.lens.tracer import (
    TracerConfig,
    tracer_image,
)


class FitResult(NamedTuple):
    """Results from fitting an image."""

    model_image: jnp.ndarray
    residuals: jnp.ndarray
    chi_squared_map: jnp.ndarray
    chi_squared: float
    log_likelihood: float
    noise_normalization: float


def residuals(
    data: jnp.ndarray,
    model: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute residuals between data and model.

    Parameters
    ----------
    data : jnp.ndarray
        Observed image data
    model : jnp.ndarray
        Model image

    Returns
    -------
    jnp.ndarray
        Residuals (data - model)
    """
    return data - model


def chi_squared_map(
    data: jnp.ndarray,
    model: jnp.ndarray,
    noise_map: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute chi-squared at each pixel.

    Parameters
    ----------
    data : jnp.ndarray
        Observed image data
    model : jnp.ndarray
        Model image
    noise_map : jnp.ndarray
        Noise (standard deviation) at each pixel

    Returns
    -------
    jnp.ndarray
        Chi-squared value at each pixel
    """
    res = residuals(data, model)
    return (res / noise_map) ** 2


def chi_squared(
    data: jnp.ndarray,
    model: jnp.ndarray,
    noise_map: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> float:
    """
    Compute total chi-squared.

    Parameters
    ----------
    data : jnp.ndarray
        Observed image data
    model : jnp.ndarray
        Model image
    noise_map : jnp.ndarray
        Noise (standard deviation) at each pixel
    mask : jnp.ndarray, optional
        Boolean mask (True = use pixel, False = ignore)

    Returns
    -------
    float
        Total chi-squared value
    """
    chi2_map = chi_squared_map(data, model, noise_map)

    if mask is not None:
        chi2_map = jnp.where(mask, chi2_map, 0.0)

    return jnp.sum(chi2_map)


def noise_normalization(
    noise_map: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> float:
    """
    Compute the noise normalization term for the log-likelihood.

    This is: sum(log(2 * pi * sigma^2))

    Parameters
    ----------
    noise_map : jnp.ndarray
        Noise (standard deviation) at each pixel
    mask : jnp.ndarray, optional
        Boolean mask

    Returns
    -------
    float
        Noise normalization term
    """
    log_term = jnp.log(2.0 * jnp.pi * noise_map**2)

    if mask is not None:
        log_term = jnp.where(mask, log_term, 0.0)

    return jnp.sum(log_term)


def log_likelihood(
    data: jnp.ndarray,
    model: jnp.ndarray,
    noise_map: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> float:
    """
    Compute the Gaussian log-likelihood.

    log L = -0.5 * (chi^2 + noise_normalization)

    Parameters
    ----------
    data : jnp.ndarray
        Observed image data
    model : jnp.ndarray
        Model image
    noise_map : jnp.ndarray
        Noise (standard deviation) at each pixel
    mask : jnp.ndarray, optional
        Boolean mask (True = use pixel)

    Returns
    -------
    float
        Log-likelihood value
    """
    chi2 = chi_squared(data, model, noise_map, mask)
    norm = noise_normalization(noise_map, mask)

    return -0.5 * (chi2 + norm)


def log_likelihood_chi2_only(
    data: jnp.ndarray,
    model: jnp.ndarray,
    noise_map: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> float:
    """
    Compute log-likelihood using only chi-squared (no noise normalization).

    This is useful when the noise normalization is constant and can be ignored
    for optimization purposes.

    log L = -0.5 * chi^2

    Parameters
    ----------
    data : jnp.ndarray
        Observed image data
    model : jnp.ndarray
        Model image
    noise_map : jnp.ndarray
        Noise (standard deviation) at each pixel
    mask : jnp.ndarray, optional
        Boolean mask

    Returns
    -------
    float
        Log-likelihood value (chi-squared term only)
    """
    chi2 = chi_squared(data, model, noise_map, mask)
    return -0.5 * chi2


# =============================================================================
# Convolution
# =============================================================================


def convolve_image(
    image: jnp.ndarray,
    psf: jnp.ndarray,
) -> jnp.ndarray:
    """
    Convolve an image with a PSF kernel.

    Parameters
    ----------
    image : jnp.ndarray
        Input image (2D array)
    psf : jnp.ndarray
        PSF kernel (2D array, should be normalized to sum to 1)

    Returns
    -------
    jnp.ndarray
        Convolved image (same size as input)
    """
    return convolve2d(image, psf, mode="same")


def convolve_image_fft(
    image: jnp.ndarray,
    psf: jnp.ndarray,
) -> jnp.ndarray:
    """
    Convolve an image with a PSF using FFT.

    More efficient for large PSF kernels.

    Parameters
    ----------
    image : jnp.ndarray
        Input image (2D array)
    psf : jnp.ndarray
        PSF kernel (2D array)

    Returns
    -------
    jnp.ndarray
        Convolved image
    """
    # Pad PSF to image size
    pad_y = (image.shape[0] - psf.shape[0]) // 2
    pad_x = (image.shape[1] - psf.shape[1]) // 2

    psf_padded = jnp.pad(
        psf,
        ((pad_y, image.shape[0] - psf.shape[0] - pad_y),
         (pad_x, image.shape[1] - psf.shape[1] - pad_x)),
        mode="constant",
    )

    # FFT convolution
    image_fft = jnp.fft.fft2(image)
    psf_fft = jnp.fft.fft2(jnp.fft.ifftshift(psf_padded))

    convolved_fft = image_fft * psf_fft
    convolved = jnp.fft.ifft2(convolved_fft).real

    return convolved


# =============================================================================
# Complete Fit Function
# =============================================================================


def fit_imaging(
    data: jnp.ndarray,
    noise_map: jnp.ndarray,
    model_image: jnp.ndarray,
    psf: Optional[jnp.ndarray] = None,
    mask: Optional[jnp.ndarray] = None,
) -> FitResult:
    """
    Perform a complete imaging fit.

    Parameters
    ----------
    data : jnp.ndarray
        Observed image data
    noise_map : jnp.ndarray
        Noise map
    model_image : jnp.ndarray
        Model image (before PSF convolution)
    psf : jnp.ndarray, optional
        PSF kernel for convolution
    mask : jnp.ndarray, optional
        Boolean mask

    Returns
    -------
    FitResult
        Named tuple containing fit results
    """
    # Convolve with PSF if provided
    if psf is not None:
        convolved_model = convolve_image(model_image, psf)
    else:
        convolved_model = model_image

    # Compute fit quantities
    res = residuals(data, convolved_model)
    chi2_map = chi_squared_map(data, convolved_model, noise_map)

    if mask is not None:
        chi2_map = jnp.where(mask, chi2_map, 0.0)

    chi2 = jnp.sum(chi2_map)
    norm = noise_normalization(noise_map, mask)
    log_like = -0.5 * (chi2 + norm)

    return FitResult(
        model_image=convolved_model,
        residuals=res,
        chi_squared_map=chi2_map,
        chi_squared=chi2,
        log_likelihood=log_like,
        noise_normalization=norm,
    )


# =============================================================================
# Tracer-based Fitting
# =============================================================================


def tracer_log_likelihood(
    grid: jnp.ndarray,
    data: jnp.ndarray,
    noise_map: jnp.ndarray,
    config: TracerConfig,
    params: dict,
    psf: Optional[jnp.ndarray] = None,
    mask: Optional[jnp.ndarray] = None,
    scaling_factors: Optional[jnp.ndarray] = None,
) -> float:
    """
    Compute log-likelihood for a tracer configuration.

    This is the main function for model fitting.

    Parameters
    ----------
    grid : jnp.ndarray
        Image-plane grid of (y, x) coordinates
    data : jnp.ndarray
        Observed image data (flattened to match grid)
    noise_map : jnp.ndarray
        Noise map (flattened)
    config : TracerConfig
        Static tracer configuration
    params : dict
        Model parameters
    psf : jnp.ndarray, optional
        PSF kernel (requires reshaping grid to 2D)
    mask : jnp.ndarray, optional
        Boolean mask
    scaling_factors : jnp.ndarray, optional
        Pre-computed cosmological scaling factors

    Returns
    -------
    float
        Log-likelihood value
    """
    # Compute model image
    model = tracer_image(grid, config, params, scaling_factors)

    # For now, skip PSF convolution in the tracer function
    # (requires 2D image handling)

    return log_likelihood(data, model, noise_map, mask)


def tracer_log_likelihood_2d(
    grid_2d: jnp.ndarray,
    data_2d: jnp.ndarray,
    noise_map_2d: jnp.ndarray,
    config: TracerConfig,
    params: dict,
    psf: Optional[jnp.ndarray] = None,
    mask_2d: Optional[jnp.ndarray] = None,
    scaling_factors: Optional[jnp.ndarray] = None,
) -> float:
    """
    Compute log-likelihood for a 2D image.

    Parameters
    ----------
    grid_2d : jnp.ndarray
        2D grid of (y, x) coordinates with shape (ny, nx, 2)
    data_2d : jnp.ndarray
        Observed 2D image
    noise_map_2d : jnp.ndarray
        2D noise map
    config : TracerConfig
        Static tracer configuration
    params : dict
        Model parameters
    psf : jnp.ndarray, optional
        PSF kernel
    mask_2d : jnp.ndarray, optional
        2D boolean mask
    scaling_factors : jnp.ndarray, optional
        Pre-computed cosmological scaling factors

    Returns
    -------
    float
        Log-likelihood value
    """
    # Flatten grid for tracer evaluation
    original_shape = grid_2d.shape[:-1]
    grid_flat = grid_2d.reshape(-1, 2)

    # Compute model
    model_flat = tracer_image(grid_flat, config, params, scaling_factors)
    model_2d = model_flat.reshape(original_shape)

    # Convolve with PSF
    if psf is not None:
        model_2d = convolve_image(model_2d, psf)

    return log_likelihood(data_2d, model_2d, noise_map_2d, mask_2d)
