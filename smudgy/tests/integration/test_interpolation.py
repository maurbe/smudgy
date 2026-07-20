"""Test interpolation functionality for different modes and quantities."""

import warnings

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

    # Interpolation with new API
    if quantity == "field":
        result = pc.interpolate_fields(
            fields=fields, query_positions=query_positions, mode="field"
        )
        assert result.shape[0] == M
        assert result.shape[1] == num_fields
        assert np.all(np.isfinite(result))

    else:  # gradient
        result = pc.interpolate_fields(
            fields=fields, query_positions=query_positions, mode="gradient"
        )
        assert result.shape[0] == M
        assert result.shape[1] == num_fields
        assert result.shape[-1] == D
        assert np.all(np.isfinite(result))


def test_backward_compatibility_compute_gradients():
    """Test that deprecated compute_gradients parameter still works with warning."""
    np.random.seed(42)
    N = 100
    M = 10
    D = 3

    positions = np.random.uniform(0, 1, size=(N, D))
    fields = np.random.uniform(-1, 1, size=(N, 3))
    weights = np.ones(N)
    query_positions = np.random.uniform(0, 1, size=(M, D))

    pc = PointCloud(positions, weights, verbose=False)
    pc.global_setup(kernel_name="cubic_spline", structure="isotropic", num_neighbors=32)
    pc.compute_smoothing()
    pc.compute_density()

    # New API
    result_new = pc.interpolate_fields(
        fields=fields, query_positions=query_positions, mode="gradient"
    )

    # Old API (should warn)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result_old = pc.interpolate_fields(
            fields=fields, query_positions=query_positions, compute_gradients=True
        )
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "deprecated" in str(w[-1].message).lower()

    # Results should match
    assert np.allclose(result_new, result_old)


def test_interpolate_gradient_fields_compatibility():
    """Test that interpolate_gradient_fields wrapper still works."""
    np.random.seed(42)
    N = 100
    M = 10
    D = 3

    positions = np.random.uniform(0, 1, size=(N, D))
    fields = np.random.uniform(-1, 1, size=(N, 3))
    weights = np.ones(N)
    query_positions = np.random.uniform(0, 1, size=(M, D))

    pc = PointCloud(positions, weights, verbose=False)
    pc.global_setup(kernel_name="cubic_spline", structure="isotropic", num_neighbors=32)
    pc.compute_smoothing()
    pc.compute_density()

    # New API
    result_new = pc.interpolate_fields(
        fields=fields, query_positions=query_positions, mode="gradient"
    )

    # Wrapper method (should match)
    result_wrapper = pc.interpolate_gradient_fields(fields, query_positions)

    assert np.allclose(result_new, result_wrapper)
