"""Tests backend consistency for density computation across Numpy CPU and Taichi CPU."""

import numpy as np
import pytest

from smudgy import PointCloud

DIMS = [2, 3]
STRUCTURES = ["isotropic", "covariant"]


def _generate_dataset(dim: int):
    """Generate a random dataset for testing."""
    np.random.seed(42)
    N = 100
    positions = np.random.uniform(0, 1, size=(N, dim))
    weights = np.ones(N, dtype=np.float32)
    boxsize = np.ones(dim, dtype=np.float32)
    return {"positions": positions, "weights": weights, "boxsize": boxsize}


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("structure", STRUCTURES)
def test_backend_consistency(dim, structure):
    """Test consistency between backends."""
    data = _generate_dataset(dim)

    positions = data["positions"]
    weights = data["weights"]
    boxsize = data["boxsize"]

    density_objects = []
    for backend in ["numpy", "taichi"]:
        pc = PointCloud(
            positions=positions,
            weights=weights,
            boxsize=boxsize,
            verbose=False,
            backend=backend,
        ).global_setup(num_neighbors=8, structure=structure, kernel_name="lucy")
        pc.compute_smoothing()
        pc.compute_density()

        if structure == "isotropic":
            dens = pc.smoothing.density_isotropic
            density_objects.append(dens)

        else:
            dens = pc.smoothing.density_covariant
            density_objects.append(dens)

    s_numpy, s_taichi = density_objects
    assert s_numpy.shape == s_taichi.shape
    np.testing.assert_allclose(s_numpy, s_taichi, rtol=1e-4, atol=1e-6)
