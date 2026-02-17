"""Tests input validation and error handling for main class APIs."""

import numpy as np
import pytest

from sph_lib import PointCloud


def test_invalid_positions_shape():
    """Test invalid positions shape raises AssertionError."""
    pos = np.zeros((10, 4), dtype=np.float32)
    mass = np.ones(10, dtype=np.float32)
    with pytest.raises(AssertionError):
        PointCloud(positions=pos, masses=mass)


def test_fields_length_mismatch():
    """Test fields length mismatch raises ValueError."""
    pos = np.zeros((10, 2), dtype=np.float32)
    mass = np.ones(10, dtype=np.float32)
    sim = PointCloud(positions=pos, masses=mass, boxsize=1.0, verbose=False)
    fields = np.ones((9, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        sim.deposit_to_grid(fields=fields, averaged=[False], gridnums=8, method="ngp")


def test_gridnums_dim_mismatch():
    """Test gridnums dimension mismatch raises ValueError."""
    pos = np.zeros((10, 2), dtype=np.float32)
    mass = np.ones(10, dtype=np.float32)
    sim = PointCloud(positions=pos, masses=mass, boxsize=1.0, verbose=False)
    with pytest.raises(ValueError):
        sim.deposit_to_grid(
            fields=mass[:, None], averaged=[False], gridnums=[8, 8, 8], method="ngp"
        )


def test_omp_threads_validation():
    """Test omp_threads=0 raises ValueError in deposit_to_grid."""
    pos = np.zeros((10, 2), dtype=np.float32)
    mass = np.ones(10, dtype=np.float32)
    sim = PointCloud(positions=pos, masses=mass, boxsize=1.0, verbose=False)
    with pytest.raises(ValueError):
        sim.deposit_to_grid(
            fields=mass[:, None],
            averaged=[False],
            gridnums=8,
            method="ngp",
            omp_threads=0,
        )
