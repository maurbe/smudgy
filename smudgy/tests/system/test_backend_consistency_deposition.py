"""Tests backend consistency for deposit() across Numpy CPU and Quadrants CPU."""

import numpy as np
import pytest

from smudgy import PointCloud

DIMS = [1, 2, 3]
GRIDNUM = 128
KERNEL_NAMES = ["ngp", "cic", "tsc", "bspline4", "bspline5", "bspline6"]


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
        arch="cpu",
    )

    pc.set_backend(backend="numpy")
    f_numpy, w_numpy = pc.deposit(**kwargs)

    pc.set_backend(backend="quadrants", arch="cpu")
    f_qd, w_qd = pc.deposit(**kwargs)

    assert f_numpy.shape == f_qd.shape
    np.testing.assert_allclose(f_numpy, f_qd, rtol=1e-5, atol=1e-6)

    assert w_numpy.shape == w_qd.shape
    np.testing.assert_allclose(w_numpy.sum(), weights.sum(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(w_qd.sum(), weights.sum(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("kernel_name", KERNEL_NAMES)
def test_backend_consistency_non_periodic_edges(dim, kernel_name):
    """Non-periodic grids drop stencil points outside the grid, in both backends.

    Particles straddle and leave the grid on every side, so the edge cells,
    and the mass lost off the grid, are compared directly at the backend level.
    """
    from smudgy.backend.numpy.deposition import deposit as deposit_np
    from smudgy.backend.quadrants.deposition import deposit as deposit_qd

    rng = np.random.default_rng(7)
    n = 5_000
    positions = rng.uniform(-0.1, 1.1, size=(n, dim)).astype(np.float32)
    fields = rng.uniform(0, 1, size=(n, 2)).astype(np.float32)
    weights = rng.uniform(0.5, 1, size=n).astype(np.float32)
    args = (positions, fields, weights, np.ones(dim, np.float32), [32] * dim)

    f_np, w_np = deposit_np(*args, periodic=False, kernel_name=kernel_name)
    f_qd, w_qd = deposit_qd(
        *args, periodic=False, kernel_name=kernel_name, adaptive=False
    )

    np.testing.assert_allclose(f_np, f_qd, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(w_np, w_qd, rtol=1e-5, atol=1e-5)
    # mass outside the grid is dropped, not piled into the edge cells
    assert w_qd.sum() < 0.95 * weights.sum()


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("kernel_name", KERNEL_NAMES)
def test_backend_consistency_periodic_wrap(dim, kernel_name):
    """Periodic grids wrap positions outside [0, L) the same way in both backends."""
    from smudgy.backend.numpy.deposition import deposit as deposit_np
    from smudgy.backend.quadrants.deposition import deposit as deposit_qd

    rng = np.random.default_rng(8)
    n = 5_000
    positions = rng.uniform(-0.3, 1.3, size=(n, dim)).astype(np.float32)
    fields = rng.uniform(0, 1, size=(n, 2)).astype(np.float32)
    weights = rng.uniform(0.5, 1, size=n).astype(np.float32)
    args = (positions, fields, weights, np.ones(dim, np.float32), [32] * dim)

    f_np, w_np = deposit_np(*args, periodic=True, kernel_name=kernel_name)
    f_qd, w_qd = deposit_qd(*args, periodic=True, kernel_name=kernel_name, adaptive=False)

    np.testing.assert_allclose(f_np, f_qd, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(w_np, w_qd, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(w_qd.sum(), weights.sum(), rtol=1e-5)
