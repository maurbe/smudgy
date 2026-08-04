"""Tests backend consistency for smoothing() across Numpy CPU and Taichi CPU."""

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

    smoothing_objects = []
    for backend in ["numpy", "taichi"]:
        pc = PointCloud(
            positions=positions,
            weights=weights,
            boxsize=boxsize,
            verbose=False,
            backend=backend,
        ).global_setup(num_neighbors=8, structure=structure)
        pc.compute_smoothing()

        if structure == "isotropic":
            s = pc.smoothing.smoothing_lengths
            smoothing_objects.append(s)

        else:
            s = pc.smoothing.smoothing_tensors
            v = pc.smoothing.smoothing_tensors_eigvecs
            l = pc.smoothing.smoothing_tensors_eigvals
            smoothing_objects.append([s, v, l])

    if structure == "isotropic":
        s_numpy, s_taichi = smoothing_objects
        assert s_numpy.shape == s_taichi.shape
        np.testing.assert_allclose(s_numpy, s_taichi, rtol=1e-4, atol=1e-6)

    else:
        # h_numpy, h_taichi = smoothing_objects[0][0], smoothing_objects[1][0]
        # assert h_numpy.shape == h_taichi.shape
        # np.testing.assert_allclose(h_numpy, h_taichi, rtol=1e-3, atol=1e-6)

        # v_numpy, v_taichi = smoothing_objects[0][1], smoothing_objects[1][1]
        # assert v_numpy.shape == v_taichi.shape
        # np.testing.assert_allclose(v_numpy[0], v_taichi[0], rtol=1e-3, atol=1e-6)

        l_numpy, l_taichi = smoothing_objects[0][2], smoothing_objects[1][2]
        # print('HERE', l_numpy.shape, l_taichi.shape)
        assert l_numpy.shape == l_taichi.shape
        np.testing.assert_allclose(l_numpy, l_taichi, rtol=1e-3, atol=1e-6)
