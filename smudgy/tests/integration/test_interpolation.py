"""Test interpolation functionality for different modes and quantities."""

import numpy as np
import pytest

from smudgy import PointCloud

PBCS = [False, True]
STRUCTURES = ["isotropic"]  # , "anisotropic"]
QUANTITIES = ["field", "gradient"]


@pytest.mark.parametrize("pbc", PBCS)
@pytest.mark.parametrize("structure", STRUCTURES)
@pytest.mark.parametrize("quantity", QUANTITIES)
def test_interpolation_modes(pbc, structure, quantity):
    """Test interpolation workflow for different PBC, methods, and quantities."""
    np.random.seed(42)
    N = 1000
    M = 10
    D = 3
    num_fields = 5
    kernel_name = "cubic_spline"

    positions = np.random.uniform(0, 1, size=(N, D))
    fields = np.random.uniform(-1, 1, size=(N, num_fields))
    boxsize = 1.0 if pbc else None
    weights = np.ones(N)

    query_positions = np.random.uniform(0, 1, size=(M, D))

    pc = PointCloud(positions, weights, boxsize=boxsize, verbose=False)
    pc.global_setup(
        kernel_name=kernel_name,
        structure=structure,
        num_neighbors=8,
    )

    pc.compute_smoothing()
    pc.compute_density()

    # Interpolation
    if quantity == "field":
        result = pc.interpolate_fields(
            fields=fields, query_positions=query_positions, compute_gradients=False
        )
        assert result.shape[0] == M
        assert result.shape[1] == num_fields
        assert np.all(np.isfinite(result))

    else:  # gradient
        result = pc.interpolate_gradient_fields(
            fields=fields, query_positions=query_positions
        )
        assert result.shape[0] == M
        assert result.shape[1] == num_fields
        assert result.shape[-1] == D
        assert np.all(np.isfinite(result))
