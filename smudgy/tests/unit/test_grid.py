"""Unit tests for the grid creation functions in smudgy.grid."""

import numpy as np

from smudgy.grid import (
    create_grid_1d,
    create_grid_2d,
    create_grid_3d,
)


def test_create_grid_1d():
    """Test create_grid_1d with various input types and shapes."""
    # int inputs
    arr = create_grid_1d(4, 8)
    assert arr.shape == (4, 1)
    # float boxsize
    arr = create_grid_1d(3, 6.0)
    assert arr.shape == (3, 1)
    # numpy int
    arr = create_grid_1d(np.int32(5), np.float32(10))
    assert arr.shape == (5, 1)


def test_create_grid_2d():
    """Test create_grid_2d with various input types and shapes."""
    # scalar n_cells, scalar boxsize
    arr = create_grid_2d(3, 6)
    assert arr.shape == (3 * 3, 2)
    # tuple n_cells, tuple boxsize
    arr = create_grid_2d((2, 4), (5, 7))
    assert arr.shape == (2 * 4, 2)
    # list n_cells, list boxsize
    arr = create_grid_2d([3, 2], [6.0, 4.0])
    assert arr.shape == (3 * 2, 2)
    # numpy array n_cells, numpy array boxsize
    arr = create_grid_2d(np.array([2, 3]), np.array([8, 9]))
    assert arr.shape == (2 * 3, 2)
    # mixed types
    arr = create_grid_2d([2, 3], (8.0, 9))
    assert arr.shape == (2 * 3, 2)


def test_create_grid_3d():
    """Test create_grid_3d with various input types and shapes."""
    # scalar n_cells, scalar boxsize
    arr = create_grid_3d(2, 6)
    assert arr.shape == (2 * 2 * 2, 3)
    # tuple n_cells, tuple boxsize
    arr = create_grid_3d((2, 3, 4), (5, 7, 9))
    assert arr.shape == (2 * 3 * 4, 3)
    # list n_cells, list boxsize
    arr = create_grid_3d([3, 2, 1], [6.0, 4.0, 2.0])
    assert arr.shape == (3 * 2 * 1, 3)
    # numpy array n_cells, numpy array boxsize
    arr = create_grid_3d(np.array([2, 3, 2]), np.array([8, 9, 10]))
    assert arr.shape == (2 * 3 * 2, 3)
    # mixed types
    arr = create_grid_3d([2, 3, 2], (8.0, 9, 10.0))
    assert arr.shape == (2 * 3 * 2, 3)
