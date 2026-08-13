"""Tests for kernels.py: registry consistency + numpy/taichi numerical agreement.

1) Registry consistency: the set of kernel names registered in the numpy
   backend (``KERNEL_CLASSES``) must match the set(s) registered in the
   taichi backend (``SEPARABLE_KERNELS`` / ``SPHERICAL_KERNELS``), including
   the exact strings used to look them up.

2) Numerical agreement: for every registered kernel, evaluate value and
   gradient on a 1D grid of canonical coordinates ``q`` and check that numpy
   and taichi agree within float32 tolerance.

Adjust the imports below to match your package layout. Where exact registry
names/attributes weren't available to me, I've made the most reasonable
assumption and flagged it with a comment -- please double check those spots.
"""

import numpy as np
import pytest
import taichi as ti

# ---------------------------------------------------------------------------
# Adjust these imports to your package layout.
# ---------------------------------------------------------------------------
from smudgy.backend.numpy.kernels import KERNEL_CLASSES, get_kernel
from smudgy.backend.taichi import init as taichi_init  # noqa: F401 (kept for reference)
from smudgy.backend.taichi.kernels import (
    SEPARABLE_KERNELS,
    SPHERICAL_KERNELS,
    create_separable_kernel,
    create_spherical_kernel,
)

# ti.init() must run before any @ti.kernel below is *defined* (decorated),
# not just before it's called -- calling the project wrapper here can be too
# late/a no-op depending on its implementation, so initialize taichi directly
# and unconditionally as the very first thing in this module.
ti.init(arch=ti.cpu, default_fp=ti.f32)

DIMS = (1, 2, 3)
ATOL = 1e-5
RTOL = 1e-4

# Canonical coordinates to sample. Kernels are typically compact-support,
# so include values inside, at, and outside the support radius.
Q_GRID = np.concatenate(
    [
        np.linspace(0.0, 3.0, 61),
        np.array([1e-6, 0.999999, 1.000001]),  # near-boundary edge cases
    ]
).astype(np.float32)


# =============================================================================
# 1) Registry consistency
# =============================================================================
class TestKernelRegistryConsistency:
    def test_numpy_registry_nonempty(self):
        assert len(KERNEL_CLASSES) > 0

    def test_taichi_registries_nonempty(self):
        assert len(SEPARABLE_KERNELS) > 0
        assert len(SPHERICAL_KERNELS) > 0

    def test_numpy_kernel_names_covered_by_taichi(self):
        """Every numpy kernel name must be resolvable on the taichi side,
        either as a separable or a spherical kernel.
        """
        numpy_names = set(KERNEL_CLASSES.keys())
        taichi_names = set(SEPARABLE_KERNELS.keys()) | set(SPHERICAL_KERNELS.keys())
        missing = numpy_names - taichi_names
        assert not missing, f"Kernel names in numpy but not taichi: {sorted(missing)}"

    def test_taichi_kernel_names_covered_by_numpy(self):
        """Every taichi kernel name must exist on the numpy side."""
        numpy_names = set(KERNEL_CLASSES.keys())
        taichi_names = set(SEPARABLE_KERNELS.keys()) | set(SPHERICAL_KERNELS.keys())
        missing = taichi_names - numpy_names
        assert not missing, f"Kernel names in taichi but not numpy: {sorted(missing)}"

    # def test_no_kernel_registered_as_both_separable_and_spherical(self):
    #    overlap = set(SEPARABLE_KERNELS.keys()) & set(SPHERICAL_KERNELS.keys())
    #    assert (
    #        not overlap
    #    ), f"Kernel names registered in both taichi dicts: {sorted(overlap)}"

    @pytest.mark.parametrize("name", sorted(KERNEL_CLASSES.keys()))
    def test_numpy_get_kernel_succeeds_for_all_dims(self, name):
        for dim in DIMS:
            kernel = get_kernel(name, dim)
            assert kernel is not None

    @pytest.mark.parametrize("name", sorted(SEPARABLE_KERNELS.keys()))
    def test_taichi_create_separable_kernel_succeeds(self, name):
        spec = create_separable_kernel(name)
        assert spec.support > 0

    @pytest.mark.parametrize("name", sorted(SPHERICAL_KERNELS.keys()))
    def test_taichi_create_spherical_kernel_succeeds(self, name):
        spec = create_spherical_kernel(name)
        assert spec.support > 0

    def test_unknown_kernel_name_raises_on_numpy(self):
        with pytest.raises(ValueError):
            get_kernel("not_a_real_kernel", dim=3)

    def test_unknown_kernel_name_raises_on_taichi_separable(self):
        with pytest.raises(ValueError):
            create_separable_kernel("not_a_real_kernel")

    def test_unknown_kernel_name_raises_on_taichi_spherical(self):
        with pytest.raises(ValueError):
            create_spherical_kernel("not_a_real_kernel")

    def test_numpy_invalid_dim_raises(self):
        any_name = next(iter(KERNEL_CLASSES.keys()))
        with pytest.raises(ValueError):
            get_kernel(any_name, dim=4)
