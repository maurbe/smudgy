"""Tests backend consistency for deposit() across Numpy CPU and Taichi CPU."""

import numpy as np
import pytest

from smudgy import PointCloud

DIMS = [1, 2, 3]
GRIDNUM = 128
KERNEL_NAMES = ["ngp", "cic", "tsc", "pcs", "pqs"]


def _generate_dataset(dim: int):
    """Generate a random dataset for testing."""
    np.random.seed(42)
    N = 10_000
    positions = np.random.uniform(0, 1, size=(N, dim))
    weights = np.random.uniform(0, 1, size=(N,))
    boxsize = np.ones(dim)
    return {"positions": positions, "weights": weights, "boxsize": boxsize}


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("kernel_name", KERNEL_NAMES)
def test_backend_consistency(dim, kernel_name):
    """Test consistency between backends."""
    data = _generate_dataset(dim)

    positions = np.asarray(data["positions"], dtype=np.float32)
    weights = np.asarray(data["weights"], dtype=np.float32)
    boxsize = np.asarray(data["boxsize"], dtype=np.float32)
    fields = weights

    kwargs = {
        "fields": fields,
        "averaged": False,
        "adaptive": False,
        "gridnums": GRIDNUM,
        "kernel_name": kernel_name,
        "return_weights": True,
    }

    pc = PointCloud(
        positions=positions,
        weights=weights,
        boxsize=boxsize,
        verbose=False,
    )

    pc.set_backend(backend="numpy")
    f_numpy, w_numpy = pc.deposit(**kwargs)

    pc.set_backend(backend="taichi")
    f_taichi, w_taichi = pc.deposit(**kwargs)

    assert f_numpy.shape == f_taichi.shape
    np.testing.assert_allclose(f_numpy, f_taichi, rtol=1e-5, atol=1e-6)

    assert w_numpy.shape == w_taichi.shape
    np.testing.assert_allclose(w_numpy.sum(), weights.sum(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(w_taichi.sum(), weights.sum(), rtol=1e-5, atol=1e-6)
