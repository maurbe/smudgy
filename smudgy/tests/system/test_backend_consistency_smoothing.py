"""Tests backend consistency for smoothing() across Numpy CPU and Taichi CPU."""

import numpy as np
import pytest

from smudgy import PointCloud

DIMS = [1, 2, 3]
STRUCTURES = ["isotropic", "covariant"]
# particles uniform in x and clustered around a few filament axes in the
# other coordinates: None is a uniform cloud; 3e-3 gives strongly anisotropic
# neighborhoods, 1e-4 ones that sit on the default axis-ratio clamp
SHAPES = {"uniform": None, "filament": 3e-3, "thin_filament": 1e-4}
TENSOR_OPTIONS = {
    "default": {},
    "volume_preserving": {"volume_preserving": True, "ratio_max": 4.0},
}


def _generate_dataset(dim: int, shape: str):
    """Generate a random dataset for testing."""
    np.random.seed(42)
    N = 900
    positions = np.random.uniform(0, 1, size=(N, dim))
    thickness = SHAPES[shape]
    if thickness is not None and dim > 1:
        centers = np.random.uniform(0, 1, size=(4, dim - 1))
        members = np.random.randint(4, size=N)
        positions[:, 1:] = centers[members] + np.random.normal(
            0, thickness, size=(N, dim - 1)
        )
        positions[:, 1:] %= 1.0
    weights = np.ones(N, dtype=np.float32)
    boxsize = np.ones(dim, dtype=np.float32)
    return {"positions": positions, "weights": weights, "boxsize": boxsize}


def _smoothing(backend, data, structure, **tensor_options):
    pc = PointCloud(
        positions=data["positions"],
        weights=data["weights"],
        boxsize=data["boxsize"],
        verbose=False,
        backend=backend,
        arch="cpu",
    ).global_setup(num_neighbors=8, structure=structure)
    pc.find_neighbors()
    pc.compute_smoothing(**tensor_options)
    return pc.smoothing


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("shape", list(SHAPES))
def test_backend_consistency_isotropic(dim, shape):
    """Test consistency between backends for smoothing lengths."""
    data = _generate_dataset(dim, shape)
    s_numpy, s_taichi = (
        _smoothing(b, data, "isotropic").smoothing_lengths for b in ("numpy", "taichi")
    )
    assert s_numpy.shape == s_taichi.shape
    np.testing.assert_allclose(s_numpy, s_taichi, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("options", list(TENSOR_OPTIONS))
def test_backend_consistency_covariant(dim, shape, options):
    """Test consistency between backends for smoothing tensors and their eigensystem."""
    data = _generate_dataset(dim, shape)
    s_numpy, s_taichi = (
        _smoothing(b, data, "covariant", **TENSOR_OPTIONS[options])
        for b in ("numpy", "taichi")
    )

    h_numpy, h_taichi = s_numpy.smoothing_tensors, s_taichi.smoothing_tensors
    l_numpy, l_taichi = (
        s_numpy.smoothing_tensors_eigvals,
        s_taichi.smoothing_tensors_eigvals,
    )
    v_numpy, v_taichi = (
        s_numpy.smoothing_tensors_eigvecs,
        s_taichi.smoothing_tensors_eigvecs,
    )
    assert h_numpy.shape == h_taichi.shape == (len(data["positions"]), dim, dim)
    assert l_numpy.shape == l_taichi.shape
    assert v_numpy.shape == v_taichi.shape

    # rtol 5e-4: the backends' float32 relative coordinates already differ by
    # an ulp (~1e-5 relative at thin-filament scales), which zeta_max amplifies
    # through the clamped Mahalanobis metric to ~1e-4.
    np.testing.assert_allclose(l_numpy, l_taichi, rtol=5e-4, atol=1e-7)
    # H entries relative to each tensor's largest axis: off-diagonal entries
    # are near zero, so a plain rtol would compare float32 noise
    scale = np.broadcast_to(l_numpy[:, :1, np.newaxis], h_numpy.shape)
    np.testing.assert_array_less(np.abs(h_numpy - h_taichi), 5e-4 * scale + 1e-12)
    assert np.all(np.diff(l_numpy, axis=1) <= 0), "eigvals must be descending"

    # eigenvectors are only unique where their eigenvalue is well separated
    gaps = np.abs(np.diff(l_numpy, axis=1)) / l_numpy[:, :1]
    separated = np.ones_like(l_numpy, dtype=bool)
    separated[:, :-1] &= gaps > 1e-2
    separated[:, 1:] &= gaps > 1e-2
    v_diff = np.abs(v_numpy - v_taichi).max(axis=1)
    np.testing.assert_array_less(v_diff[separated], 1e-3)
