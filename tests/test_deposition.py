import numpy as np
import pytest
from sph_lib import collection

@pytest.mark.parametrize("method", ["cic", "tsc"])
@pytest.mark.parametrize("mode", ["isotropic", "anisotropic"])
def test_deposition_grid_adaptive(method, mode):
    np.random.seed(42)
    N = 100
    D = 3
    positions = np.random.uniform(0, 1, size=(N, D))
    values = np.random.uniform(-1, 1, size=(N, 1))
    boxsize = np.ones(D)
    gridnums = np.array([16, 16, 16])
    periodic = [True, True, True]
    pcellsizesHalf = np.full((N, D), 0.05)
    if mode == "isotropic":
        hsm = np.full(N, 0.1)
        hmat_eigvecs = None
        hmat_eigvals = None
    else:
        hsm = None
        hmat_eigvecs = np.array([np.eye(D) for _ in range(N)])
        hmat_eigvals = np.full((N, D), 0.1)
    if method == "cic":
        if mode == "isotropic":
            fields, weights = collection.cic_adaptive_3d(positions, values, boxsize, gridnums, periodic, pcellsizesHalf)
        else:
            # For anisotropic, assume function exists or skip
            pytest.skip("Anisotropic adaptive CIC not implemented in this example.")
    else:  # tsc
        if mode == "isotropic":
            fields, weights = collection.tsc_adaptive_3d(positions, values, boxsize, gridnums, periodic, pcellsizesHalf)
        else:
            pytest.skip("Anisotropic adaptive TSC not implemented in this example.")
    assert fields.shape[:3] == tuple(gridnums)
