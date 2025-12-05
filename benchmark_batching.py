"""Benchmark batching speedup as a function of batch size."""

import jax
import jax.numpy as jnp
from jax import vmap, jit
import time
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, ".")

from jax_lens.lens.tracer import simple_tracer_image


def benchmark_batch_size(batch_sizes, n_pixels=50, n_repeats=10):
    """Benchmark sequential vs batched for various batch sizes."""

    # Grid
    pixel_scale = 0.1
    extent = n_pixels * pixel_scale / 2
    coords = jnp.stack(jnp.meshgrid(
        jnp.linspace(-extent, extent, n_pixels),
        jnp.linspace(-extent, extent, n_pixels),
        indexing='ij'
    ), axis=-1).reshape(-1, 2)

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

    # JIT compile single model
    jit_lens_model = jit(lens_model)

    # Warmup
    _ = jit_lens_model(1.0, 0.8, 0.1, 0.05).block_until_ready()

    results = []

    for batch_size in batch_sizes:
        print(f"Benchmarking batch_size={batch_size}...")

        # Generate random parameters
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        einstein_radii = 1.0 + 0.1 * jax.random.normal(keys[0], (batch_size,))
        axis_ratios = jnp.clip(0.8 + 0.05 * jax.random.normal(keys[1], (batch_size,)), 0.3, 0.99)
        source_x = 0.1 + 0.05 * jax.random.normal(keys[2], (batch_size,))
        source_y = 0.05 + 0.05 * jax.random.normal(keys[3], (batch_size,))

        # Batched version
        batched_lens = jit(vmap(lens_model))
        _ = batched_lens(einstein_radii, axis_ratios, source_x, source_y).block_until_ready()  # warmup

        batched_times = []
        for _ in range(n_repeats):
            start = time.perf_counter()
            _ = batched_lens(einstein_radii, axis_ratios, source_x, source_y).block_until_ready()
            batched_times.append(time.perf_counter() - start)
        batched_time = np.median(batched_times)

        # Sequential version
        sequential_times = []
        for _ in range(n_repeats):
            start = time.perf_counter()
            for i in range(batch_size):
                _ = jit_lens_model(einstein_radii[i], axis_ratios[i], source_x[i], source_y[i]).block_until_ready()
            sequential_times.append(time.perf_counter() - start)
        sequential_time = np.median(sequential_times)

        speedup = sequential_time / batched_time
        results.append({
            'batch_size': batch_size,
            'sequential_ms': sequential_time * 1000,
            'batched_ms': batched_time * 1000,
            'speedup': speedup,
        })

        print(f"  Sequential: {sequential_time*1000:.2f} ms, Batched: {batched_time*1000:.2f} ms, Speedup: {speedup:.1f}x")

    return results


def plot_results(results):
    """Plot speedup vs batch size."""
    batch_sizes = [r['batch_size'] for r in results]
    speedups = [r['speedup'] for r in results]
    sequential = [r['sequential_ms'] for r in results]
    batched = [r['batched_ms'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Speedup plot
    axes[0].plot(batch_sizes, speedups, 'o-', markersize=8, linewidth=2)
    axes[0].set_xlabel('Batch Size', fontsize=12)
    axes[0].set_ylabel('Speedup (sequential / batched)', fontsize=12)
    axes[0].set_title('vmap Speedup vs Batch Size', fontsize=14)
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    # Time comparison plot
    axes[1].plot(batch_sizes, sequential, 'o-', label='Sequential (JIT)', markersize=8, linewidth=2)
    axes[1].plot(batch_sizes, batched, 's-', label='Batched (vmap)', markersize=8, linewidth=2)
    axes[1].set_xlabel('Batch Size', fontsize=12)
    axes[1].set_ylabel('Time (ms)', fontsize=12)
    axes[1].set_title('Execution Time vs Batch Size', fontsize=14)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('batching_benchmark.png', dpi=150, bbox_inches='tight')
    print("\nSaved batching_benchmark.png")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Batching Speedup Benchmark")
    print("=" * 60)
    print("Grid: 50x50, varying batch size\n")

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    results = benchmark_batch_size(batch_sizes, n_pixels=50, n_repeats=5)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Batch':>8} {'Sequential':>12} {'Batched':>12} {'Speedup':>10}")
    print("-" * 44)
    for r in results:
        print(f"{r['batch_size']:>8} {r['sequential_ms']:>10.2f}ms {r['batched_ms']:>10.2f}ms {r['speedup']:>9.1f}x")

    plot_results(results)
