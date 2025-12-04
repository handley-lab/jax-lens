"""
Basic example: Simulating and fitting a strong gravitational lens.

This example demonstrates:
1. Creating a simple lens + source system
2. Simulating mock data
3. Computing the log-likelihood
4. Using jax.vmap for batched parameter evaluation
5. Computing gradients with jax.grad
"""

import jax
import jax.numpy as jnp
import jax.random as jr

# Import jax_lens components
import sys
sys.path.insert(0, "/home/will/tmp/natalie/demo")

import jax_lens as jl
from jax_lens.profiles import light, mass
from jax_lens.lens.tracer import (
    TracerConfig,
    PlaneConfig,
    tracer_image,
    simple_tracer_image,
)
from jax_lens.fitting.imaging import log_likelihood, convolve_image
from jax_lens.pipeline import create_simple_lens_likelihood


def create_grid(n_pixels: int, pixel_scale: float) -> jnp.ndarray:
    """Create a 2D grid of coordinates centered on (0, 0)."""
    half_size = (n_pixels - 1) / 2 * pixel_scale
    y = jnp.linspace(half_size, -half_size, n_pixels)
    x = jnp.linspace(-half_size, half_size, n_pixels)
    yy, xx = jnp.meshgrid(y, x, indexing="ij")
    return jnp.stack([yy, xx], axis=-1)


def create_psf(size: int, sigma: float) -> jnp.ndarray:
    """Create a simple Gaussian PSF."""
    half = (size - 1) / 2
    y = jnp.linspace(-half, half, size)
    x = jnp.linspace(-half, half, size)
    yy, xx = jnp.meshgrid(y, x, indexing="ij")
    psf = jnp.exp(-0.5 * (yy**2 + xx**2) / sigma**2)
    return psf / jnp.sum(psf)  # Normalize


def main():
    print("=" * 60)
    print("JAX-Native Strong Lensing Example")
    print("=" * 60)

    # ==========================================================================
    # Setup
    # ==========================================================================
    key = jr.PRNGKey(42)

    # Grid parameters
    n_pixels = 100
    pixel_scale = 0.05  # arcsec/pixel

    # Create coordinate grid
    grid_2d = create_grid(n_pixels, pixel_scale)
    grid_flat = grid_2d.reshape(-1, 2)
    print(f"\nGrid shape: {grid_2d.shape}")
    print(f"Pixel scale: {pixel_scale} arcsec/pixel")
    print(f"Field of view: {n_pixels * pixel_scale:.1f} arcsec")

    # ==========================================================================
    # Define True Parameters
    # ==========================================================================
    true_params = {
        "lens_mass": {
            "centre": jnp.array([0.0, 0.0]),
            "einstein_radius": 1.0,
            "axis_ratio": 0.8,
            "angle": 0.5,  # radians
        },
        "source_light": {
            "centre": jnp.array([0.1, 0.05]),
            "intensity": 1.0,
            "effective_radius": 0.3,
            "sersic_index": 1.0,
            "axis_ratio": 0.7,
            "angle": 1.0,
        },
    }

    print("\nTrue Parameters:")
    print(f"  Lens Einstein radius: {true_params['lens_mass']['einstein_radius']}")
    print(f"  Lens axis ratio: {true_params['lens_mass']['axis_ratio']}")
    print(f"  Source centre: {true_params['source_light']['centre']}")
    print(f"  Source effective radius: {true_params['source_light']['effective_radius']}")

    # ==========================================================================
    # Simulate Data
    # ==========================================================================
    print("\n" + "-" * 60)
    print("Simulating Data")
    print("-" * 60)

    # Compute true model image
    true_image_flat = simple_tracer_image(
        grid=grid_flat,
        lens_mass_params=true_params["lens_mass"],
        source_light_params=true_params["source_light"],
        lens_mass_type="sie",
        source_light_type="sersic",
    )
    true_image_2d = true_image_flat.reshape(n_pixels, n_pixels)

    # Create and apply PSF
    psf = create_psf(11, 1.5)
    true_image_convolved = convolve_image(true_image_2d, psf)

    # Add noise
    noise_level = 0.01
    noise_map = jnp.ones_like(true_image_convolved) * noise_level
    key, subkey = jr.split(key)
    noise = jr.normal(subkey, true_image_convolved.shape) * noise_level
    data = true_image_convolved + noise

    print(f"Image shape: {data.shape}")
    print(f"Peak flux: {jnp.max(true_image_convolved):.3f}")
    print(f"Noise level: {noise_level}")
    print(f"Peak S/N: {jnp.max(true_image_convolved) / noise_level:.1f}")

    # ==========================================================================
    # Create Likelihood Function
    # ==========================================================================
    print("\n" + "-" * 60)
    print("Creating Likelihood Function")
    print("-" * 60)

    likelihood_fn = create_simple_lens_likelihood(
        grid=grid_flat,
        data=data.flatten(),
        noise_map=noise_map.flatten(),
        z_lens=0.5,
        z_source=1.0,
        lens_mass_type="sie",
        source_light_type="sersic",
        psf=psf,
        image_shape=(n_pixels, n_pixels),
    )

    # JIT compile for speed
    likelihood_fn_jit = jax.jit(likelihood_fn)

    # Evaluate at true parameters
    log_like_true = likelihood_fn_jit(true_params)
    print(f"Log-likelihood at true params: {log_like_true:.2f}")

    # ==========================================================================
    # Test with Perturbed Parameters
    # ==========================================================================
    print("\n" + "-" * 60)
    print("Testing Perturbed Parameters")
    print("-" * 60)

    # Create perturbed parameters
    perturbed_params = {
        "lens_mass": {
            "centre": jnp.array([0.02, -0.01]),
            "einstein_radius": 1.05,
            "axis_ratio": 0.75,
            "angle": 0.55,
        },
        "source_light": {
            "centre": jnp.array([0.12, 0.08]),
            "intensity": 0.95,
            "effective_radius": 0.32,
            "sersic_index": 1.1,
            "axis_ratio": 0.65,
            "angle": 1.1,
        },
    }

    log_like_perturbed = likelihood_fn_jit(perturbed_params)
    print(f"Log-likelihood at perturbed params: {log_like_perturbed:.2f}")
    print(f"Delta log-likelihood: {log_like_perturbed - log_like_true:.2f}")

    # ==========================================================================
    # Compute Gradients
    # ==========================================================================
    print("\n" + "-" * 60)
    print("Computing Gradients")
    print("-" * 60)

    grad_fn = jax.grad(likelihood_fn)
    grads = grad_fn(true_params)

    print("Gradients at true parameters (should be ~0):")
    print(f"  d/d(einstein_radius): {grads['lens_mass']['einstein_radius']:.4f}")
    print(f"  d/d(source_intensity): {grads['source_light']['intensity']:.4f}")

    grads_perturbed = grad_fn(perturbed_params)
    print("\nGradients at perturbed parameters:")
    print(f"  d/d(einstein_radius): {grads_perturbed['lens_mass']['einstein_radius']:.4f}")
    print(f"  d/d(source_intensity): {grads_perturbed['source_light']['intensity']:.4f}")

    # ==========================================================================
    # Batched Evaluation with vmap
    # ==========================================================================
    print("\n" + "-" * 60)
    print("Batched Evaluation with vmap")
    print("-" * 60)

    n_batch = 100

    # Create batched parameters by varying einstein_radius
    key, subkey = jr.split(key)
    einstein_radii = jr.uniform(subkey, (n_batch,), minval=0.8, maxval=1.2)

    # Build batched parameter structure
    batched_params = {
        "lens_mass": {
            "centre": jnp.tile(true_params["lens_mass"]["centre"], (n_batch, 1)),
            "einstein_radius": einstein_radii,
            "axis_ratio": jnp.full(n_batch, 0.8),
            "angle": jnp.full(n_batch, 0.5),
        },
        "source_light": {
            "centre": jnp.tile(true_params["source_light"]["centre"], (n_batch, 1)),
            "intensity": jnp.full(n_batch, 1.0),
            "effective_radius": jnp.full(n_batch, 0.3),
            "sersic_index": jnp.full(n_batch, 1.0),
            "axis_ratio": jnp.full(n_batch, 0.7),
            "angle": jnp.full(n_batch, 1.0),
        },
    }

    # vmap the likelihood function
    batched_likelihood_fn = jax.vmap(likelihood_fn)

    # Time the batched evaluation
    import time

    # Warmup
    _ = batched_likelihood_fn(batched_params)

    start = time.time()
    log_likes = batched_likelihood_fn(batched_params)
    elapsed = time.time() - start

    print(f"Evaluated {n_batch} likelihoods in {elapsed * 1000:.2f} ms")
    print(f"Time per evaluation: {elapsed / n_batch * 1000:.3f} ms")
    print(f"Log-likelihood range: [{jnp.min(log_likes):.1f}, {jnp.max(log_likes):.1f}]")

    # Find best fit in batch
    best_idx = jnp.argmax(log_likes)
    print(f"\nBest Einstein radius in batch: {einstein_radii[best_idx]:.3f}")
    print(f"True Einstein radius: {true_params['lens_mass']['einstein_radius']:.3f}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
This example demonstrated:
1. Creating coordinate grids for lensing calculations
2. Defining lens and source parameters as nested dictionaries
3. Simulating lensed images with PSF convolution and noise
4. Creating JIT-compiled likelihood functions
5. Computing gradients with jax.grad
6. Batched parameter evaluation with jax.vmap

The key advantage of this JAX-native implementation is that:
- All operations are differentiable (enables gradient-based sampling)
- vmap allows efficient batched evaluation (perfect for MCMC)
- JIT compilation provides significant speedups
- Same code works on CPU and GPU
""")


if __name__ == "__main__":
    main()
