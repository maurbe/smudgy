"""Test interpolation functionality for different modes and quantities."""

import pytest
import numpy as np
from smudgy import PointCloud

PBCS = [False, True]
METHODS = ["isotropic", "anisotropic"]
QUANTITIES = ["field", "gradient"]


@pytest.mark.parametrize("pbc", PBCS)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("quantity", QUANTITIES)
def test_interpolation_modes(pbc, method, quantity):
    """Test interpolation workflow for different PBC, methods, and quantities."""
    np.random.seed(42)
    N = 1000
    D = 3
    kernel_name = "cubic_spline"

    positions = np.random.uniform(0, 1, size=(N, D))
    values = np.random.uniform(-1, 1, size=N)
    boxsize = 1.0 if pbc else None
    weights = np.ones(N)

    pc = PointCloud(positions, weights, boxsize=boxsize, verbose=False)
    pc.setup(num_neighbors=8, method=method, kernel_name=kernel_name)
    pc.compute_smoothing_lengths()
    pc.compute_density()

    # Interpolation
    if quantity == "field":
        result = pc.interpolate_fields(
            values, positions, kernel_name=kernel_name, compute_gradients=False
        )
        assert result.shape[0] == N
        assert np.all(np.isfinite(result))
    else:  # gradient
        result = pc.interpolate_fields(
            values, positions, kernel_name=kernel_name, compute_gradients=True
        )
        assert result.shape[0] == N
        assert result.shape[-1] == D
        assert np.all(np.isfinite(result))
