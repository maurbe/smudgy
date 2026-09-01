"""Tests backend consistency for density computation across Numpy CPU and Taichi CPU."""

import numpy as np

import smudgy as sm
from smudgy.backend.numpy.tensor_utils import project_2d as project_2d_numpy
from smudgy.backend.taichi.tensor_utils import project_2d as project_2d_taichi


def _generate_dataset(dim: int):
    """Generate a random dataset for testing."""
    np.random.seed(42)
    N = 300
    positions = np.random.uniform(0, 1, size=(N, dim))
    weights = np.ones(N, dtype=np.float32)
    boxsize = np.ones(dim, dtype=np.float32)
    return {"positions": positions, "weights": weights, "boxsize": boxsize}


def test_backend_consistency():
    """Test consistency between backends."""
    data = _generate_dataset(dim=3)

    pc = sm.PointCloud(
        positions=data["positions"],
        weights=data["weights"],
        boxsize=data["boxsize"],
        verbose=False,
        arch="cpu",
    ).global_setup(num_neighbors=8, structure="covariant", kernel_name="lucy")
    pc.find_neighbors()
    pc.compute_smoothing()

    H_tensor = pc.smoothing.smoothing_tensors
    H_2d_np, evals_np, evecs_np = project_2d_numpy(H_tensor)
    H_2d_ti, evals_ti, evecs_ti = project_2d_taichi(H_tensor)

    assert H_2d_np.shape == H_2d_ti.shape
    np.testing.assert_allclose(H_2d_np, H_2d_ti, rtol=1e-3, atol=1e-4)

    assert evals_np.shape == evals_ti.shape
    np.testing.assert_allclose(evals_np, evals_ti, rtol=1e-3, atol=1e-4)

    assert evecs_np.shape == evecs_ti.shape
    print(evecs_np[:5], evecs_ti[:5])
    # Eigenvector *direction* is inherently more sensitive to tiny numerical
    # differences than eigenvalues/H_2d, especially near-degenerate
    # eigenvalues -- looser tolerance than the checks above is expected.
    np.testing.assert_allclose(evecs_np, evecs_ti, rtol=1e-2, atol=1e-3)
