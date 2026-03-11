import numpy as np
import pytest
from smudgy.utils import shift_coordinates, coordinate_difference_with_pbc

def test_shift_coordinates_basic():
    coords = np.array([[2, 3], [5, 7], [1, 9]])
    shifted = shift_coordinates(coords)
    assert np.allclose(shifted.min(axis=0), 0)
    assert shifted.shape == coords.shape


def test_shift_coordinates_error():
    coords = np.array([1, 2, 3])
    with pytest.raises(AssertionError):
        shift_coordinates(coords)


def test_coordinate_difference_with_pbc_none():
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[0, 1], [1, 2]])
    diff = coordinate_difference_with_pbc(x, y, boxsize=None)
    assert np.allclose(diff, x - y)


def test_coordinate_difference_with_pbc_scalar():
    x = np.array([[0.9, 0.1], [0.2, 0.8]])
    y = np.array([[0.1, 0.9], [0.8, 0.2]])
    diff = coordinate_difference_with_pbc(x, y, boxsize=1.0)
    assert diff.shape == x.shape


def test_coordinate_difference_with_pbc_array():
    x = np.array([[0.9, 0.1], [0.2, 0.8]])
    y = np.array([[0.1, 0.9], [0.8, 0.2]])
    diff = coordinate_difference_with_pbc(x, y, boxsize=[1.0, 1.0])
    assert diff.shape == x.shape


def test_coordinate_difference_with_pbc_dim_error():
    x = np.array([[1, 2, 3]])
    y = np.array([[4, 5]])
    with pytest.raises(AssertionError):
        coordinate_difference_with_pbc(x, y, boxsize=1.0)


def test_coordinate_difference_with_pbc_boxsize_error():
    x = np.array([[1, 2]])
    y = np.array([[3, 4]])
    with pytest.raises(AssertionError):
        coordinate_difference_with_pbc(x, y, boxsize=[1.0, 2.0, 3.0])
