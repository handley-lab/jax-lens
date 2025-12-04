"""
Demonstration of jax_lens gravitational lensing simulation.

This script generates a visualization showing:
1. The unlensed source galaxy
2. The lensed image (Einstein ring)
3. Mock observational data with PSF and noise

Run with: python examples/demo_lensing.py
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, ".")

from jax_lens.profiles import light, mass
from jax_lens.lens.tracer import simple_tracer_image
from jax_lens.fitting.imaging import convolve_image


def create_grid(n_pixels, pixel_scale):
    """Create a coordinate grid centred at (0, 0)."""
    extent = n_pixels * pixel_scale / 2
    coords = jnp.linspace(-extent, extent, n_pixels)
    yy, xx = jnp.meshgrid(coords, coords, indexing='ij')
    return jnp.stack([yy, xx], axis=-1).reshape(-1, 2)


def create_psf(size=21, sigma=0.1, pixel_scale=0.05):
    """Create a Gaussian PSF."""
    coords = jnp.linspace(-(size//2)*pixel_scale, (size//2)*pixel_scale, size)
    yy, xx = jnp.meshgrid(coords, coords, indexing='ij')
    r2 = yy**2 + xx**2
    psf = jnp.exp(-r2 / (2 * sigma**2))
    return psf / psf.sum()


def main():
    # Setup
    n_pixels = 100
    pixel_scale = 0.05  # arcsec/pixel

    # Create grid
    grid = create_grid(n_pixels, pixel_scale)

    # Lens parameters (SIE - Singular Isothermal Ellipsoid)
    lens_params = {
        "centre": jnp.array([0.0, 0.0]),
        "einstein_radius": 1.2,
        "axis_ratio": 0.8,
        "angle": 0.5,  # radians
    }

    # Source parameters (Sersic profile)
    source_params = {
        "centre": jnp.array([0.1, 0.05]),
        "intensity": 4.0,
        "effective_radius": 0.3,
        "sersic_index": 1.0,
        "axis_ratio": 0.7,
        "angle": 1.0,
    }

    # Generate unlensed source
    source_image = light.sersic(
        grid=grid,
        centre=source_params["centre"],
        intensity=source_params["intensity"],
        effective_radius=source_params["effective_radius"],
        sersic_index=source_params["sersic_index"],
        axis_ratio=source_params["axis_ratio"],
        angle=source_params["angle"],
    ).reshape(n_pixels, n_pixels)

    # Generate lensed image
    lensed_image = simple_tracer_image(
        grid=grid,
        lens_mass_params=lens_params,
        source_light_params=source_params,
        lens_mass_type="sie",
        source_light_type="sersic",
    ).reshape(n_pixels, n_pixels)

    # Create mock observation (PSF convolution + noise)
    psf = create_psf(size=21, sigma=0.1, pixel_scale=pixel_scale)
    convolved = convolve_image(lensed_image, psf, padding="edge")

    # Add Gaussian noise
    np.random.seed(42)
    noise_sigma = 0.05
    mock_data = np.array(convolved) + np.random.normal(0, noise_sigma, convolved.shape)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    extent_arcsec = n_pixels * pixel_scale / 2
    extent = [-extent_arcsec, extent_arcsec, -extent_arcsec, extent_arcsec]

    # Source
    im0 = axes[0].imshow(np.array(source_image), origin='lower', extent=extent, cmap='inferno')
    axes[0].set_title("Source (unlensed)", fontsize=12)
    axes[0].set_xlabel("x [arcsec]")
    axes[0].set_ylabel("y [arcsec]")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Lensed image
    im1 = axes[1].imshow(np.array(lensed_image), origin='lower', extent=extent, cmap='inferno')
    axes[1].set_title("Lensed Image (no PSF)", fontsize=12)
    axes[1].set_xlabel("x [arcsec]")
    axes[1].set_ylabel("y [arcsec]")
    # Add Einstein radius circle
    theta = np.linspace(0, 2*np.pi, 100)
    axes[1].plot(lens_params["einstein_radius"]*np.cos(theta),
                 lens_params["einstein_radius"]*np.sin(theta),
                 'w--', alpha=0.5, linewidth=1)
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # Mock data
    im2 = axes[2].imshow(mock_data, origin='lower', extent=extent, cmap='inferno')
    axes[2].set_title("Mock Data (PSF + noise)", fontsize=12)
    axes[2].set_xlabel("x [arcsec]")
    axes[2].set_ylabel("y [arcsec]")
    axes[2].plot(lens_params["einstein_radius"]*np.cos(theta),
                 lens_params["einstein_radius"]*np.sin(theta),
                 'w--', alpha=0.5, linewidth=1)
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    plt.savefig("lens_demo.png", dpi=150, bbox_inches='tight')
    print("Saved lens_demo.png")
    plt.show()


if __name__ == "__main__":
    main()
