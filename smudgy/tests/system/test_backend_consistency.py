"""Tests cross-backend consistency between numpy and taichi deposition."""

import numpy as np
import pytest

from smudgy import PointCloud

GRIDNUM = 64
KERNEL_NAMES = ["ngp", "cic", "tsc", "pcs", "pqs"]
STRUCTURE = "separable"


def _generate_dataset(dim: int):
    """Generate a random dataset for testing."""
    np.random.seed(42)
    N = 1000
    positions = np.random.uniform(0, 1, size=(N, dim)).astype(np.float32)
    weights = np.ones(N, dtype=np.float32)
    boxsize = np.ones(dim, dtype=np.float32)
    return positions, weights, boxsize


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("kernel_name", KERNEL_NAMES)
def test_numpy_taichi_deposition_consistency(dim, kernel_name):
    """Ensure numpy and taichi produce the same deposition result for static kernels."""
    positions, weights, boxsize = _generate_dataset(dim)
    fields = weights

    sim = PointCloud(
        positions=positions,
        weights=weights,
        boxsize=boxsize,
        verbose=False,
    )

    fields_numpy = sim.deposit_to_grid(
        fields=fields,
        averaged=False,
        gridnums=GRIDNUM,
        kernel_name=kernel_name,
        structure=STRUCTURE,
        adaptive=False,
        backend="numpy",
    )

    fields_taichi = sim.deposit_to_grid(
        fields=fields,
        averaged=False,
        gridnums=GRIDNUM,
        kernel_name=kernel_name,
        structure=STRUCTURE,
        adaptive=False,
        backend="taichi",
        accelerator="cpu",
        omp_threads=1,
    )

    assert fields_numpy.shape == fields_taichi.shape
    np.testing.assert_allclose(fields_numpy, fields_taichi, rtol=1e-4, atol=1e-6)
