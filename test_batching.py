"""Test that jax_lens actually works with batched inference via vmap."""

import jax
import jax.numpy as jnp
from jax import vmap, jit
import time

import sys
sys.path.insert(0, ".")

from jax_lens.profiles import light, mass
from jax_lens.lens.tracer import simple_tracer_image


def test_batched_sersic():
    """Test batched Sersic evaluation over multiple parameter sets."""
    print("\n=== Batched Sersic Profile ===")

    # Grid
    n = 50
    extent = 2.0
    coords = jnp.stack(jnp.meshgrid(
        jnp.linspace(-extent, extent, n),
        jnp.linspace(-extent, extent, n),
        indexing='ij'
    ), axis=-1).reshape(-1, 2)

    # Batch of parameters (100 different intensities)
    n_batch = 100
    intensities = jnp.linspace(0.5, 2.0, n_batch)

    # Define batched function
    def sersic_for_intensity(intensity):
        return light.sersic(
            grid=coords,
            centre=jnp.array([0.0, 0.0]),
            intensity=intensity,
            effective_radius=0.5,
            sersic_index=2.0,
        )

    # Compile and run batched version
    batched_sersic = jit(vmap(sersic_for_intensity))

    # Warmup
    _ = batched_sersic(intensities).block_until_ready()

    # Time it
    start = time.time()
    results = batched_sersic(intensities).block_until_ready()
    elapsed = time.time() - start

    print(f"Batch size: {n_batch}")
    print(f"Grid points: {len(coords)}")
    print(f"Output shape: {results.shape}")
    print(f"Time for {n_batch} evaluations: {elapsed*1000:.2f} ms")
    print(f"Time per evaluation: {elapsed/n_batch*1000:.3f} ms")

    # Verify results vary with intensity
    assert results.shape == (n_batch, len(coords))
    assert not jnp.allclose(results[0], results[-1])
    print("✓ Batched Sersic works correctly")


def test_batched_sie_deflections():
    """Test batched SIE deflection over multiple Einstein radii."""
    print("\n=== Batched SIE Deflections ===")

    # Grid
    n = 50
    extent = 2.0
    coords = jnp.stack(jnp.meshgrid(
        jnp.linspace(-extent, extent, n),
        jnp.linspace(-extent, extent, n),
        indexing='ij'
    ), axis=-1).reshape(-1, 2)

    # Batch of Einstein radii
    n_batch = 100
    einstein_radii = jnp.linspace(0.5, 2.0, n_batch)

    def sie_for_einstein_radius(einstein_radius):
        return mass.sie_deflections(
            grid=coords,
            centre=jnp.array([0.0, 0.0]),
            einstein_radius=einstein_radius,
            axis_ratio=0.8,
            angle=0.5,
        )

    batched_sie = jit(vmap(sie_for_einstein_radius))

    # Warmup
    _ = batched_sie(einstein_radii).block_until_ready()

    # Time it
    start = time.time()
    results = batched_sie(einstein_radii).block_until_ready()
    elapsed = time.time() - start

    print(f"Batch size: {n_batch}")
    print(f"Grid points: {len(coords)}")
    print(f"Output shape: {results.shape}")
    print(f"Time for {n_batch} evaluations: {elapsed*1000:.2f} ms")
    print(f"Time per evaluation: {elapsed/n_batch*1000:.3f} ms")

    assert results.shape == (n_batch, len(coords), 2)
    print("✓ Batched SIE deflections work correctly")


def test_batched_full_lens_model():
    """Test batched full lens model - the key use case for MCMC."""
    print("\n=== Batched Full Lens Model (MCMC use case) ===")

    # Grid
    n = 50
    pixel_scale = 0.1
    extent = n * pixel_scale / 2
    coords = jnp.stack(jnp.meshgrid(
        jnp.linspace(-extent, extent, n),
        jnp.linspace(-extent, extent, n),
        indexing='ij'
    ), axis=-1).reshape(-1, 2)

    # Batch of lens parameters (simulating MCMC walkers)
    n_walkers = 32

    # Random parameter variations
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)

    einstein_radii = 1.0 + 0.1 * jax.random.normal(keys[0], (n_walkers,))
    axis_ratios = 0.8 + 0.05 * jax.random.normal(keys[1], (n_walkers,))
    axis_ratios = jnp.clip(axis_ratios, 0.3, 0.99)
    source_x = 0.1 + 0.05 * jax.random.normal(keys[2], (n_walkers,))
    source_y = 0.05 + 0.05 * jax.random.normal(keys[3], (n_walkers,))

    def lens_model(einstein_radius, axis_ratio, src_x, src_y):
        lens_params = {
            "centre": jnp.array([0.0, 0.0]),
            "einstein_radius": einstein_radius,
            "axis_ratio": axis_ratio,
            "angle": 0.5,
        }
        source_params = {
            "centre": jnp.array([src_y, src_x]),
            "intensity": 1.0,
            "effective_radius": 0.3,
            "sersic_index": 1.0,
            "axis_ratio": 0.7,
            "angle": 1.0,
        }
        return simple_tracer_image(
            grid=coords,
            lens_mass_params=lens_params,
            source_light_params=source_params,
            lens_mass_type="sie",
            source_light_type="sersic",
        )

    # Vectorize over all parameters
    batched_lens = jit(vmap(lens_model))

    # Warmup
    _ = batched_lens(einstein_radii, axis_ratios, source_x, source_y).block_until_ready()

    # Time it
    start = time.time()
    results = batched_lens(einstein_radii, axis_ratios, source_x, source_y).block_until_ready()
    elapsed = time.time() - start

    print(f"Number of walkers: {n_walkers}")
    print(f"Grid points: {len(coords)}")
    print(f"Output shape: {results.shape}")
    print(f"Time for {n_walkers} lens models: {elapsed*1000:.2f} ms")
    print(f"Time per model: {elapsed/n_walkers*1000:.3f} ms")

    assert results.shape == (n_walkers, len(coords))

    # Verify different parameters give different results
    assert not jnp.allclose(results[0], results[1])
    print("✓ Batched full lens model works correctly")

    # Compare to sequential (with JIT - fair comparison)
    print("\n--- Sequential comparison (both JIT compiled) ---")
    jit_lens_model = jit(lens_model)

    # Warmup sequential
    _ = jit_lens_model(einstein_radii[0], axis_ratios[0], source_x[0], source_y[0]).block_until_ready()

    start = time.time()
    for i in range(n_walkers):
        _ = jit_lens_model(einstein_radii[i], axis_ratios[i], source_x[i], source_y[i]).block_until_ready()
    sequential_time = time.time() - start

    print(f"Sequential time (JIT): {sequential_time*1000:.2f} ms")
    print(f"Batched time: {elapsed*1000:.2f} ms")
    print(f"Speedup: {sequential_time/elapsed:.1f}x")


def test_gradient_through_batch():
    """Test that gradients work through batched operations."""
    print("\n=== Gradients Through Batched Operations ===")

    n = 30
    extent = 2.0
    coords = jnp.stack(jnp.meshgrid(
        jnp.linspace(-extent, extent, n),
        jnp.linspace(-extent, extent, n),
        indexing='ij'
    ), axis=-1).reshape(-1, 2)

    # Mock "data"
    key = jax.random.PRNGKey(0)
    mock_data = jax.random.normal(key, (n*n,))

    def loss_fn(einstein_radius):
        """Chi-squared loss for a single parameter."""
        lens_params = {
            "centre": jnp.array([0.0, 0.0]),
            "einstein_radius": einstein_radius,
            "axis_ratio": 0.8,
            "angle": 0.5,
        }
        source_params = {
            "centre": jnp.array([0.1, 0.05]),
            "intensity": 1.0,
            "effective_radius": 0.3,
            "sersic_index": 1.0,
        }
        model = simple_tracer_image(
            grid=coords,
            lens_mass_params=lens_params,
            source_light_params=source_params,
            lens_mass_type="sie",
            source_light_type="sersic",
        )
        return jnp.sum((model - mock_data)**2)

    # Gradient of loss
    grad_loss = jax.grad(loss_fn)

    # Batched gradient (useful for finite differencing or ensemble methods)
    einstein_radii = jnp.array([0.9, 1.0, 1.1])
    batched_grad = jit(vmap(grad_loss))

    grads = batched_grad(einstein_radii)

    print(f"Einstein radii: {einstein_radii}")
    print(f"Gradients: {grads}")
    print(f"Gradient shape: {grads.shape}")

    assert grads.shape == (3,)
    assert jnp.all(jnp.isfinite(grads))
    print("✓ Gradients through batched operations work correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("jax_lens Batching Tests")
    print("=" * 60)

    test_batched_sersic()
    test_batched_sie_deflections()
    test_batched_full_lens_model()
    test_gradient_through_batch()

    print("\n" + "=" * 60)
    print("All batching tests passed!")
    print("=" * 60)
