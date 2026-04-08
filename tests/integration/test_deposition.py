"""Test suite for deposition operations in SPH library."""

import pytest
import numpy as np
from smudgy import PointCloud


@pytest.mark.parametrize("method", ["isotropic", "anisotropic"])
def test_deposition_grid_adaptive(method):
    """Test grid deposition for adaptive modes."""
    np.random.seed(42)
    N = 100
    D = 3
    positions = np.random.uniform(0, 1, size=(N, D))
    values = np.random.uniform(-1, 1, size=(N, 1))
    weights = np.ones(N)
    boxsize = 1.0
    gridnums = 16

    pc = PointCloud(positions, weights, boxsize=boxsize, verbose=False)
    pc.setup(method=method)
    pc.compute_smoothing_lengths()

    averaged = [False] * values.shape[1]
    # Use PointCloud.deposit_to_grid for both modes
    result = pc.deposit_to_grid(
        fields=values,
        averaged=averaged,
        gridnums=gridnums,
        method=method,
        return_weights=True,
        kernel_name="quintic_spline",
        integration="midpoint",
        min_kernel_evaluations_per_axis=4,
    )
    fields, weights = result
    assert fields.shape[:3] == tuple([gridnums] * 3)
