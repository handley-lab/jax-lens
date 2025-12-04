This is a comprehensive architectural implementation plan for refactoring PyAutoLens to a JAX-native architecture. This plan is designed to be handed to a senior research software engineer or contractor.

# PyAutoLens JAX Refactoring Implementation Plan

**Objective:** Transform the core PyAutoLens modeling pipeline from a stateful Object-Oriented architecture to a functional, stateless JAX-native architecture.
**Primary Goal:** Enable `jax.vmap` across the model parameter dimension to support batched MCMC sampling (e.g., NUTS, Hamiltonian Monte Carlo).
**Constraint:** Maintain exact numerical equivalence with the existing physics implementation.

---

## 1. Architecture Overview

The current architecture is **Instance-Based**:
`ModelInstance` $\rightarrow$ `Tracer` Object $\rightarrow$ `Fit` Object $\rightarrow$ `float`

The target architecture is **Functional & Stateless**:
`Parameter PyTree` + `Static Data` $\rightarrow$ `Pure Function` $\rightarrow$ `float`

### Core Architectural Shifts

1.  **State/Behavior Separation**:
    *   **Old**: `Galaxy` objects hold parameters (redshift, intensity) and methods (`image_2d_from`).
    *   **New**: Parameters are held in nested Dictionaries/NamedTuples (PyTrees). Logic is in standalone pure functions that accept these PyTrees.

2.  **Static Graph Construction**:
    *   The structure of the lens system (number of planes, types of profiles) is determined at **compile time**.
    *   Python control flow (loops, ifs) is used to generate the computational graph for the specific model configuration.
    *   JAX traces the flow of data through this graph.

3.  **The "Parameter PyTree"**:
    *   A standardized nested dictionary structure mirroring the model hierarchy, containing only JAX arrays or scalars.

---

## 2. Module-by-Module Refactoring Plan

### A. Profiles (`autogalaxy/profiles/`)
*   **Current**: Class hierarchy (`LightProfile`, `MassProfile`) with `image_2d_from(grid)` methods.
*   **Target**: Functional modules (`profiles/light/functional.py`).
*   **Action**: Extract mathematical kernels from `image_2d_from` and `deflections_yx_2d_from` into pure functions.
*   **Dependencies**: None. Start here.

### B. The Tracer (`lens/`)
*   **Current**: `Tracer` class in `lens/tracer.py`. Handles sorting galaxies, grouping into planes, and iterative ray-tracing.
*   **Target**: `lens/functional_tracer.py`.
*   **Action**:
    *   Replace `Tracer.__init__` sorting logic with a pre-processing step (performed once before JIT).
    *   Rewrite `traced_grid_2d_list_from` in `lens/tracer_util.py` to accept lists of Mass Profile parameters instead of objects.
*   **Dependencies**: Profile Refactoring.

### C. Imaging & Interferometer (`imaging/`, `interferometer/`)
*   **Current**: `FitImaging`, `FitInterferometer`. Heavily stateful (cached properties).
*   **Target**: `imaging/functional_fit.py`, `interferometer/functional_fit.py`.
*   **Action**: Create pure functions that take `(model_image, data, noise)` and return `log_likelihood`.
*   **Dependencies**: Tracer Refactoring.

### D. Analysis (`analysis/`)
*   **Current**: `Analysis.log_likelihood_function` instantiates `Tracer`.
*   **Target**: Update `Analysis` to act as an Adapter. It converts `autofit.ModelInstance` into the `Parameter PyTree` and calls the compiled JAX function.

---

## 3. Core Data Structures (The PyTree)

We must define a standard data structure for passing parameters to JAX functions.

```python
from typing import Dict, List, TypedDict, Union
import jax.numpy as jnp

# Type definitions for the JAX Parameter Tree
class ProfileParams(TypedDict):
    # Flexible dict for profile parameters (e.g., center, intensity, effective_radius)
    # Keys match the attribute names in current LightProfile/MassProfile classes
    pass

class GalaxyParams(TypedDict):
    redshift: float
    light_profiles: List[ProfileParams]
    mass_profiles: List[ProfileParams]

class TracerParams(TypedDict):
    galaxies: List[GalaxyParams]
    cosmology_params: Dict[str, float] # e.g. Omega_m, h
```

**Handling Redshift Sorting:**
In standard strong lensing, the number of planes and the ordering of galaxies are fixed for a given model.
*   **Strategy**: The "Tracer Structure" (which profiles belong to which galaxy, and which plane that galaxy belongs to) is defined **statically** in Python before JIT.
*   The JAX function will receive a list of lists of parameters, pre-grouped by plane.

---

## 4. Functional Likelihood Pipeline

This is the target "end-state" for the refactor.

**File:** `autolens/functional/pipeline.py` (Proposed)

```python
import jax.numpy as jnp
from autolens.lens import functional_tracer
from autolens.imaging import functional_fit

def log_likelihood(
    params: TracerParams, 
    data: jnp.ndarray, 
    noise_map: jnp.ndarray, 
    grid: jnp.ndarray,
    convolver_matrix: jnp.ndarray
) -> float:
    """
    Pure JAX function for likelihood evaluation.
    """
    # 1. Ray Tracing
    # Returns the image plane image derived from params
    model_image = functional_tracer.tracer_image_from_params(
        params, grid
    )
    
    # 2. Instrument Effects (Convolving)
    # Assuming simple matrix multiplication for convolution for JAX efficiency
    convolved_image = jnp.dot(convolver_matrix, model_image.flatten()).reshape(data.shape)
    
    # 3. Fitting
    log_like = functional_fit.compute_log_likelihood(
        convolved_image, data, noise_map
    )
    
    return log_like
```

---

## 5. Profile System Refactoring

Refactor `autogalaxy/profiles/` to separate parameters from evaluation.

**Task:** Create `autogalaxy/profiles/light/functional.py`

```python
import jax.numpy as jnp

def elliptical_sersic(grid, intensity, effective_radius, sersic_index, axis_ratio, angle, centre):
    """
    Stateless evaluation of a Sersic profile.
    """
    # ... Implementation of Sersic math using jnp ...
    # This logic currently exists in EllSersic.image_2d_from
    return image
```

**Task:** Update `Tracer` to call these based on a static configuration list.

---

## 6. The Tracer Refactoring (Critical Path)

The `Tracer` logic in `lens/tracer.py` and `lens/tracer_util.py` must be unrolled.

**Handling Multi-plane Lensing in JAX:**
We cannot iterate over a list of `Galaxy` objects. We must iterate over a list of parameter dictionaries.

**File:** `lens/functional_tracer.py`

```python
def trace_grids(mass_profile_params_per_plane, grid, cosmology_constants):
    """
    Iterates through planes to deflect the grid.
    
    mass_profile_params_per_plane: List[List[Dict]] 
        Outer list: Planes (sorted by redshift)
        Inner list: Mass profiles in that plane
    """
    current_grid = grid
    
    # We use a Python loop here because the number of planes is static (compile-time constant)
    # JAX will unroll this into the computational graph.
    for plane_params in mass_profile_params_per_plane:
        
        # Calculate Deflections for this plane
        deflections = jnp.zeros_like(current_grid)
        for profile_params in plane_params:
             # Dispatch to specific profile function based on type (handled by closure/partial setup)
             deflections += evaluate_deflections(profile_params, current_grid)
        
        # Apply Cosmology Scaling and subtract
        scaling_factor = ... # Calculation based on static redshifts
        current_grid -= deflections * scaling_factor
        
    return current_grid
```

---

## 7. Fit Classes Refactoring

Refactor `imaging/fit_imaging.py`.

**Current:** OOP handles masking, convolution, and residual calculation properties.
**New:** `imaging/functional_fit.py`.

```python
import jax.numpy as jnp

def compute_log_likelihood(model_image, data, noise_map):
    residuals = data - model_image
    chi_squared_map = (residuals / noise_map) ** 2
    chi_squared = jnp.sum(chi_squared_map)
    noise_normalization = jnp.sum(jnp.log(2 * jnp.pi * noise_map ** 2))
    return -0.5 * (chi_squared + noise_normalization)
```

**Note on Inversions:**
Refactoring `Inversion` (pixelizations) to JAX is complex due to dynamic matrix sizes in some mappers (e.g., Voronoi).
*   **Initial Scope:** Support only Light Profile fitting (no Inversions).
*   **Secondary Scope:** Support `Rectangular` and `Delaunay` pixelizations if grid sizes are static.

---

## 8. Integration with AutoFit

We need an adapter in `analysis/analysis/dataset.py` to bridge the gap.

```python
# analysis/analysis/dataset.py

class AnalysisDataset(..., use_jax=True):
    
    def __init__(self, ...):
        # ... setup ...
        if self.use_jax:
            self.jax_likelihood = self._compile_jax_likelihood()

    def _compile_jax_likelihood(self):
        # 1. Build the functional topology (which profiles are where)
        # 2. Partial out static data (grid, image data)
        # 3. Apply JIT
        return jax.jit(functional_pipeline.log_likelihood)

    def log_likelihood_function(self, instance):
        if self.use_jax:
            # 1. Convert instance (OOP) to PyTree (Dicts)
            params_pytree = self._instance_to_pytree(instance)
            # 2. Call JAX
            return self.jax_likelihood(params_pytree)
        else:
            # Fallback to original OOP
            return super().log_likelihood_function(instance)
```

---

## 9. Testing Strategy

Verification is vital. We must ensure the JAX implementation produces identical results to the NumPy OOP implementation.

1.  **Unit Tests (Kernels):**
    *   Compare `EllSersic.image_2d_from` (NumPy) vs `functional.elliptical_sersic` (JAX).
    *   Tolerance: `1e-6`.

2.  **Integration Tests (Likelihood):**
    *   Instantiate a `FitImaging` object using the old method.
    *   Extract parameters and run the new `functional_log_likelihood`.
    *   Assert `old_log_like == new_log_like`.

3.  **Gradient Tests:**
    *   Use `jax.grad` on the new likelihood.
    *   Compare against finite differences on the old likelihood. This validates differentiability.

---

## 10. Migration Path

**Phase 1: "Dual Core" (Low Risk)**
*   Implement the `functional` modules alongside existing code.
*   Update `Analysis` to support `use_jax=True` flag.
*   The `Tracer` object remains the source of truth for *model structure*, but the calculation is offloaded.

**Phase 2: JAX-Native Profiling**
*   Enable `vmap` support in the `Analysis` class.
*   Connect to `BlackJAX` or `NumPyro` for sampling.

**Phase 3: Deprecation (Long Term)**
*   Once JAX coverage is 100% (including Inversions), replace internal logic of `Tracer` classes to wrap the JAX functions, removing code duplication.

---

## 11. Code Example: vmap Usage

This is how the user will eventually run the batched analysis.

```python
import jax
import jax.numpy as jnp
from autolens.functional import pipeline

# 1. Prepare Static Data (do this once)
static_data = (image_data, noise_map, grid, convolver)

# 2. Prepare Batched Parameters (N walkers)
# params_batch is a Dict of Arrays, where every leaf has shape (N_walkers, ...)
params_batch = {
    "galaxies": [
        {
            "light_profiles": [{"intensity": jnp.array([...]), ...}],
            "mass_profiles": [...]
        }
    ]
}

# 3. Create Vectorized Function
# in_axes=(0, None, None, None, None) maps over the first argument (params)
batched_likelihood = jax.vmap(
    pipeline.log_likelihood, 
    in_axes=(0, None, None, None, None)
)

# 4. Execute
# Returns shape (N_walkers,)
log_likelihoods = batched_likelihood(params_batch, *static_data)
```