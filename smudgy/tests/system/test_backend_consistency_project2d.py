"""Tests backend consistency for project_2d() across Numpy CPU and Taichi CPU."""

import numpy as np
import pytest

import smudgy as sm
from smudgy.backend.numpy.tensor_utils import project_2d as project_2d_numpy
from smudgy.backend.taichi.tensor_utils import project_2d as project_2d_taichi

PLANES = [(0, 1), (0, 2), (1, 2)]


def _generate_dataset(dim: int):
    """Generate a random dataset for testing."""
    np.random.seed(42)
    N = 300
    positions = np.random.uniform(0, 1, size=(N, dim))
    weights = np.ones(N, dtype=np.float32)
    boxsize = np.ones(dim, dtype=np.float32)
    return {"positions": positions, "weights": weights, "boxsize": boxsize}


def _random_tensors(n, gap, scale, seed=0):
    """Randomly oriented SPD tensors with two axes differing by a relative `gap`.

    Small gaps make many 2D projections nearly isotropic, the regime where
    ti.sym_eig used to return wrong eigenvectors.
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, 3, 3)))
    axes = scale * np.stack(
        [np.ones(n), 1 + gap * rng.uniform(0.5, 1, n), rng.uniform(0.3, 3, n)], axis=1
    )
    return np.einsum("nij,nj,nkj->nik", Q, axes, Q).astype(np.float32)


def _assert_projections_agree(H_tensor, plane=(0, 1)):
    H_2d_np, evals_np, evecs_np = project_2d_numpy(H_tensor, plane)
    H_2d_ti, evals_ti, evecs_ti = project_2d_taichi(H_tensor, plane)

    assert H_2d_np.shape == H_2d_ti.shape == (len(H_tensor), 2, 2)
    assert evals_np.shape == evals_ti.shape
    assert evecs_np.shape == evecs_ti.shape
    assert np.all(np.isfinite(evals_ti)) and np.all(np.isfinite(evecs_ti))

    # entries relative to each tensor's largest eigenvalue
    scale = evals_np[:, -1:, np.newaxis]
    np.testing.assert_array_less(
        np.abs(H_2d_np - H_2d_ti), np.broadcast_to(1e-4 * scale, H_2d_np.shape)
    )
    np.testing.assert_allclose(evals_np, evals_ti, rtol=1e-4)

    # the taichi eigensystem is orthonormal and reproduces its own tensor
    eye = np.broadcast_to(np.eye(2), evecs_ti.shape)
    np.testing.assert_allclose(np.swapaxes(evecs_ti, 1, 2) @ evecs_ti, eye, atol=1e-5)
    recon = np.einsum("nij,nj,nkj->nik", evecs_ti, evals_ti, evecs_ti)
    np.testing.assert_array_less(
        np.abs(recon - H_2d_ti), np.broadcast_to(1e-5 * scale, H_2d_ti.shape)
    )

    # eigenvectors are only unique where the two eigenvalues are well separated
    gap = (evals_np[:, 1] - evals_np[:, 0]) / evals_np[:, 1]
    separated = gap > 1e-2
    np.testing.assert_array_less(
        np.abs(evecs_np - evecs_ti).max(axis=(1, 2))[separated], 1e-3
    )


def test_backend_consistency():
    """Test consistency between backends on covariant smoothing tensors."""
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

    for plane in PLANES:
        _assert_projections_agree(pc.smoothing.smoothing_tensors, plane)


@pytest.mark.parametrize("plane", PLANES)
@pytest.mark.parametrize("gap", [1e-1, 1e-3, 0.0])
@pytest.mark.parametrize("scale", [1e-2, 1.0])
def test_backend_consistency_near_degenerate(plane, gap, scale):
    """Near-isotropic projections, at unit and at typical smoothing-length scale."""
    _assert_projections_agree(_random_tensors(5000, gap, scale), plane)
