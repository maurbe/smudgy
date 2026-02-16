
import numpy as np
import pytest
from sph_lib import collection
from sph_lib.ops import PointCloud


@pytest.mark.parametrize("mode", ["isotropic", "anisotropic"])
def test_deposition_grid_adaptive(mode):
    np.random.seed(42)
    N = 100
    D = 3
    positions = np.random.uniform(0, 1, size=(N, D))
    values = np.random.uniform(-1, 1, size=(N, 1))
    masses = np.ones(N)
    boxsize = np.ones(D)
    gridnums = np.array([16, 16, 16])

    pc = PointCloud(positions, masses, boxsize=boxsize, verbose=False)
    pc.set_sph_parameters(mode=mode)
    pc.compute_smoothing_lengths()

    averaged = [False] * values.shape[1]
    # Use PointCloud.deposit_to_grid for both modes
    result = pc.deposit_to_grid(
        fields=values,
        averaged=averaged,
        gridnums=gridnums,
        method=mode,
        return_weights=True,
        kernel='quintic',
        integration='midpoint',
        min_kernel_evaluations=128,
    )
    fields, weights = result
    assert fields.shape[:3] == tuple(gridnums)
