"""
Tests for light and mass profiles.

These tests verify that the JAX implementations produce correct results
and are compatible with JAX transformations (jit, vmap, grad).
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap, grad

import sys
sys.path.insert(0, "/home/will/tmp/natalie/demo")

from jax_lens.profiles import light, mass


class TestLightProfiles:
    """Tests for light profile implementations."""

    def test_sersic_at_center(self):
        """Sersic profile should peak at the centre."""
        grid = jnp.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
        centre = jnp.array([0.0, 0.0])

        image = light.sersic(
            grid=grid,
            centre=centre,
            intensity=1.0,
            effective_radius=0.5,
            sersic_index=1.0,
        )

        # Value at centre should be highest
        assert image[0] >= image[1]
        assert image[0] >= image[2]

    def test_sersic_symmetry(self):
        """Circular Sersic should be axially symmetric."""
        grid = jnp.array([[0.1, 0.0], [0.0, 0.1], [-0.1, 0.0], [0.0, -0.1]])
        centre = jnp.array([0.0, 0.0])

        image = light.sersic(
            grid=grid,
            centre=centre,
            intensity=1.0,
            effective_radius=0.5,
            sersic_index=1.0,
            axis_ratio=1.0,
        )

        # All points at same radius should have same value
        assert jnp.allclose(image[0], image[1], rtol=1e-5)
        assert jnp.allclose(image[0], image[2], rtol=1e-5)
        assert jnp.allclose(image[0], image[3], rtol=1e-5)

    def test_sersic_ellipticity(self):
        """Elliptical Sersic should be elongated along major axis."""
        grid = jnp.array([[0.2, 0.0], [0.0, 0.2]])  # Along y and x axes
        centre = jnp.array([0.0, 0.0])

        # axis_ratio < 1 means compressed along y
        image = light.sersic(
            grid=grid,
            centre=centre,
            intensity=1.0,
            effective_radius=0.5,
            sersic_index=1.0,
            axis_ratio=0.5,
            angle=0.0,  # Major axis along x
        )

        # Point along x (major axis) should be brighter
        assert image[1] > image[0]

    def test_exponential_equals_sersic_n1(self):
        """Exponential profile should equal Sersic with n=1."""
        grid = jnp.array([[0.0, 0.0], [0.1, 0.2], [0.5, -0.3]])
        centre = jnp.array([0.0, 0.0])

        exp_image = light.exponential(
            grid=grid,
            centre=centre,
            intensity=1.0,
            effective_radius=0.5,
        )

        sersic_image = light.sersic(
            grid=grid,
            centre=centre,
            intensity=1.0,
            effective_radius=0.5,
            sersic_index=1.0,
        )

        assert jnp.allclose(exp_image, sersic_image, rtol=1e-5)

    def test_gaussian_normalization(self):
        """Gaussian should have correct peak value."""
        grid = jnp.array([[0.0, 0.0]])
        centre = jnp.array([0.0, 0.0])

        image = light.gaussian(
            grid=grid,
            centre=centre,
            intensity=2.5,
            sigma=0.3,
        )

        # At centre, Gaussian should equal intensity
        assert jnp.isclose(image[0], 2.5, rtol=1e-5)

    def test_sersic_jit_compatible(self):
        """Sersic should work with JAX JIT."""
        @jit
        def compute_sersic(intensity):
            grid = jnp.array([[0.0, 0.0], [0.1, 0.1]])
            return light.sersic(
                grid=grid,
                centre=jnp.array([0.0, 0.0]),
                intensity=intensity,
                effective_radius=0.5,
                sersic_index=1.0,
            )

        result = compute_sersic(1.0)
        assert result.shape == (2,)

    def test_sersic_vmap_compatible(self):
        """Sersic should work with JAX vmap."""
        grid = jnp.array([[0.0, 0.0], [0.1, 0.1]])
        intensities = jnp.array([1.0, 2.0, 3.0])

        @vmap
        def compute_sersic(intensity):
            return light.sersic(
                grid=grid,
                centre=jnp.array([0.0, 0.0]),
                intensity=intensity,
                effective_radius=0.5,
                sersic_index=1.0,
            )

        result = compute_sersic(intensities)
        assert result.shape == (3, 2)

    def test_sersic_grad_compatible(self):
        """Sersic should be differentiable."""
        def compute_total_flux(intensity):
            grid = jnp.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
            return jnp.sum(light.sersic(
                grid=grid,
                centre=jnp.array([0.0, 0.0]),
                intensity=intensity,
                effective_radius=0.5,
                sersic_index=1.0,
            ))

        grad_fn = grad(compute_total_flux)
        gradient = grad_fn(1.0)

        # Gradient should be positive and finite
        assert jnp.isfinite(gradient)
        assert gradient > 0


class TestMassProfiles:
    """Tests for mass profile implementations."""

    def test_sis_deflection_direction(self):
        """SIS deflections should point radially outward."""
        grid = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        centre = jnp.array([0.0, 0.0])

        deflections = mass.sis_deflections(
            grid=grid,
            centre=centre,
            einstein_radius=1.0,
        )

        # Deflections should point in same direction as position
        for i in range(4):
            dot_product = jnp.dot(grid[i], deflections[i])
            assert dot_product > 0

    def test_sis_deflection_magnitude(self):
        """SIS should have constant deflection magnitude."""
        grid = jnp.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
        centre = jnp.array([0.0, 0.0])
        einstein_radius = 1.5

        deflections = mass.sis_deflections(
            grid=grid,
            centre=centre,
            einstein_radius=einstein_radius,
        )

        magnitudes = jnp.sqrt(deflections[:, 0]**2 + deflections[:, 1]**2)

        # All magnitudes should equal einstein_radius
        assert jnp.allclose(magnitudes, einstein_radius, rtol=1e-5)

    def test_sis_convergence_profile(self):
        """SIS convergence should fall off as 1/r."""
        grid = jnp.array([[0.5, 0.0], [1.0, 0.0], [2.0, 0.0]])
        centre = jnp.array([0.0, 0.0])

        kappa = mass.sis_convergence(
            grid=grid,
            centre=centre,
            einstein_radius=1.0,
        )

        # kappa should scale as 1/r
        # kappa(r=0.5) / kappa(r=1.0) should be 2
        assert jnp.isclose(kappa[0] / kappa[1], 2.0, rtol=1e-5)
        # kappa(r=1.0) / kappa(r=2.0) should be 2
        assert jnp.isclose(kappa[1] / kappa[2], 2.0, rtol=1e-5)

    def test_sie_reduces_to_sis(self):
        """SIE with axis_ratio=1 should equal SIS."""
        grid = jnp.array([[0.5, 0.3], [-0.2, 0.8], [1.0, -0.5]])
        centre = jnp.array([0.0, 0.0])
        einstein_radius = 1.2

        sis_deflections = mass.sis_deflections(
            grid=grid,
            centre=centre,
            einstein_radius=einstein_radius,
        )

        sie_deflections = mass.sie_deflections(
            grid=grid,
            centre=centre,
            einstein_radius=einstein_radius,
            axis_ratio=1.0,
            angle=0.0,
        )

        assert jnp.allclose(sis_deflections, sie_deflections, rtol=1e-3)

    def test_nfw_convergence_positive(self):
        """NFW convergence should be positive everywhere."""
        grid = jnp.array([[0.1, 0.0], [1.0, 0.0], [5.0, 0.0]])
        centre = jnp.array([0.0, 0.0])

        kappa = mass.nfw_convergence(
            grid=grid,
            centre=centre,
            kappa_s=0.1,
            scale_radius=1.0,
        )

        assert jnp.all(kappa > 0)

    def test_mass_profile_jit_compatible(self):
        """Mass profiles should work with JAX JIT."""
        @jit
        def compute_deflections(einstein_radius):
            grid = jnp.array([[0.5, 0.3], [1.0, 0.0]])
            return mass.sis_deflections(
                grid=grid,
                centre=jnp.array([0.0, 0.0]),
                einstein_radius=einstein_radius,
            )

        result = compute_deflections(1.0)
        assert result.shape == (2, 2)

    def test_mass_profile_vmap_compatible(self):
        """Mass profiles should work with JAX vmap."""
        grid = jnp.array([[0.5, 0.3], [1.0, 0.0]])
        einstein_radii = jnp.array([0.5, 1.0, 1.5])

        @vmap
        def compute_deflections(einstein_radius):
            return mass.sis_deflections(
                grid=grid,
                centre=jnp.array([0.0, 0.0]),
                einstein_radius=einstein_radius,
            )

        result = compute_deflections(einstein_radii)
        assert result.shape == (3, 2, 2)

    def test_mass_profile_grad_compatible(self):
        """Mass profiles should be differentiable."""
        def compute_total_deflection(einstein_radius):
            grid = jnp.array([[0.5, 0.3], [1.0, 0.0]])
            deflections = mass.sis_deflections(
                grid=grid,
                centre=jnp.array([0.0, 0.0]),
                einstein_radius=einstein_radius,
            )
            return jnp.sum(deflections**2)

        grad_fn = grad(compute_total_deflection)
        gradient = grad_fn(1.0)

        assert jnp.isfinite(gradient)
        assert gradient > 0

    def test_external_shear_divergence_zero(self):
        """External shear should have zero divergence (pure shear, no convergence)."""
        # Test on a grid
        grid = jnp.array([
            [0.5, 0.3], [-0.2, 0.8], [1.0, -0.5], [-0.7, -0.3]
        ])

        gamma_1, gamma_2 = 0.1, 0.05

        deflections = mass.external_shear_deflections(
            grid=grid,
            gamma_1=gamma_1,
            gamma_2=gamma_2,
        )

        # For pure shear: div(alpha) = d(alpha_x)/dx + d(alpha_y)/dy = 0
        # With alpha_x = gamma_1*x + gamma_2*y and alpha_y = -gamma_1*y + gamma_2*x
        # d(alpha_x)/dx = gamma_1, d(alpha_y)/dy = -gamma_1
        # divergence = gamma_1 + (-gamma_1) = 0

        # We verify this analytically: the implementation must give:
        # alpha_y = -gamma_1 * y + gamma_2 * x
        # alpha_x = gamma_1 * x + gamma_2 * y
        expected_alpha_y = -gamma_1 * grid[:, 0] + gamma_2 * grid[:, 1]
        expected_alpha_x = gamma_1 * grid[:, 1] + gamma_2 * grid[:, 0]

        assert jnp.allclose(deflections[:, 0], expected_alpha_y, rtol=1e-5)
        assert jnp.allclose(deflections[:, 1], expected_alpha_x, rtol=1e-5)

    def test_external_shear_convergence_zero(self):
        """External shear should have zero convergence."""
        grid = jnp.array([[0.5, 0.3], [1.0, 0.0], [-0.5, 0.5]])

        kappa = mass.external_shear_convergence(
            grid=grid,
            gamma_1=0.1,
            gamma_2=0.05,
        )

        assert jnp.allclose(kappa, 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
