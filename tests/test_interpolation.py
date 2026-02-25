"""Test interpolation functionality for different modes and quantities."""

import numpy as np
import pytest

from smudgy import PointCloud


@pytest.mark.parametrize("pbc", [False, True])
@pytest.mark.parametrize("mode", ["isotropic", "anisotropic"])
@pytest.mark.parametrize("quantity", ["field", "gradient"])
def test_interpolation_modes(pbc, mode, quantity):
    """Test interpolation for different PBC, modes, and quantities."""
    np.random.seed(42)
    N = 1000
    D = 3
    positions = np.random.uniform(0, 1, size=(N, D))
    values = np.random.uniform(-1, 1, size=N)
    boxsize = 1.0 if pbc else None
    masses = np.ones(N)

    pc = PointCloud(positions, masses, boxsize=boxsize, verbose=False)
    pc.set_sph_parameters(kernel_name="cubic_spline", mode=mode, num_neighbors=32)
    pc.compute_smoothing_lengths()
    pc.compute_density()

    # Interpolation
    if quantity == "field":
        result = pc.interpolate_fields(values, positions, compute_gradients=False)
        assert result.shape[0] == N
    else:  # gradient
        result = pc.interpolate_fields(values, positions, compute_gradients=True)
        assert result.shape[0] == N
        assert result.shape[-1] == D
