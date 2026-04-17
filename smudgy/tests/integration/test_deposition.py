"""Test suite for deposition operations in SPH library."""

import pytest
import numpy as np
from smudgy import PointCloud

STRUCTURES = ["separable", "isotropic", "anisotropic"]

@pytest.mark.parametrize("structure", STRUCTURES)
def test_deposition_grid_adaptive(structure):
    """Test grid deposition for adaptive modes."""
    np.random.seed(42)
    N = 10
    D = 3
    positions = np.random.uniform(0, 1, size=(N, D))
    values = np.random.uniform(-1, 1, size=(N, 1))
    weights = np.ones(N)
    boxsize = 1.0
    gridnums = 16

    pc = PointCloud(positions, weights, boxsize=boxsize, verbose=True)
    pc.global_setup(structure=structure, num_neighbors=8)
    pc.compute_smoothing()

    averaged = [False] * values.shape[1]
    # Use PointCloud.deposit_to_grid for both modes
    result = pc.deposit_to_grid(
        fields=values,
        averaged=averaged,
        gridnums=gridnums,
        
        kernel_name="gaussian",

        integration="midpoint",
        min_kernel_evaluations_per_axis=4,
        return_weights=True,
    )
    fields, weights = result

    print("Deposited fields shape:", fields.shape)
    assert fields.shape[:3] == tuple([gridnums] * 3)
