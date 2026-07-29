"""Tests backend consistency for deposit() across Numpy CPU and Taichi CPU."""

import numpy as np
import pytest

from smudgy import PointCloud

DIMS = [1, 2, 3]
GRIDNUM = 64
KERNEL_NAME = "cic"


def _generate_dataset(dim: int):
    """Generate a random dataset for testing."""
    np.random.seed(42)
    N = 1000
    positions = np.random.uniform(0, 1, size=(N, dim))
    weights = np.ones(N, dtype=np.float32)
    boxsize = np.ones(dim, dtype=np.float32)
    return {"positions": positions, "weights": weights, "boxsize": boxsize}


@pytest.mark.parametrize("dim", DIMS)
def test_backend_consistency(dim):
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
        "kernel_name": KERNEL_NAME,
    }

    pc = PointCloud(
        positions=positions,
        weights=weights,
        boxsize=boxsize,
        verbose=False,
    )

    pc.set_backend(backend="numpy")
    f_numpy = pc.deposit(**kwargs)

    pc.set_backend(backend="taichi")
    f_taichi = pc.deposit(**kwargs)

    assert f_numpy.shape == f_taichi.shape
    np.testing.assert_allclose(f_numpy, f_taichi, rtol=1e-4, atol=1e-6)
