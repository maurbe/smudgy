"""Tests backend consistency for interpolation() (smoothing + smoothing length indirectly too) across Numpy CPU and Taichi CPU."""

import numpy as np
import pytest

from smudgy import PointCloud

DIMS = [2, 3]
GRIDNUM = 64
KERNEL_NAME = "gaussian"
STRUCTURES = ["isotropic", "covariant"]


def _generate_dataset(dim: int):
    """Generate a random dataset for testing."""
    np.random.seed(42)
    N = 1000
    positions = np.random.uniform(0, 1, size=(N, dim))
    weights = np.ones(N, dtype=np.float32)
    boxsize = np.ones(dim, dtype=np.float32)
    return {"positions": positions, "weights": weights, "boxsize": boxsize}


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("strucutre", STRUCTURES)
def test_backend_consistency(dim, strucutre):
    """Test consistency between backends."""
    data = _generate_dataset(dim)

    positions = data["positions"]
    weights = data["weights"]
    boxsize = data["boxsize"]
    vector_field = np.ones((positions.shape[0], dim))

    data_2 = _generate_dataset(dim)
    query_positions = data_2["positions"]

    f_interpolated = []
    for backend in ["numpy", "taichi"]:
        pc = PointCloud(
            positions=positions,
            weights=weights,
            boxsize=boxsize,
            verbose=False,
            backend=backend,
        ).global_setup(kernel_name=KERNEL_NAME, num_neighbors=8, structure=strucutre)
        pc.compute_smoothing()
        pc.compute_density()
        pc.add_fields("vf", vector_field)

        f = pc.interpolate(fields="vf", query_positions=query_positions)
        f_interpolated.append(f)
    f_numpy, f_taichi = f_interpolated

    assert f_numpy.shape == f_taichi.shape
    np.testing.assert_allclose(f_numpy, f_taichi, rtol=1e-4, atol=1e-6)
