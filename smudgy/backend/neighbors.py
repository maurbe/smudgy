"""Utility functions for SPH operations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy import spatial

FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.int_]
BoxInput = float | Sequence[float] | npt.ArrayLike


def build_kdtree(
    points: npt.ArrayLike, boxsize: BoxInput | None = None
) -> spatial.cKDTree:
    """Construct a cKDTree with optional periodic box sizes.

    If ``boxsize`` is provided, the tree will use periodic boundary conditions.

    Parameters
    ----------
    points
            Array of shape ``(N, D)`` with particle coordinates, where ``N`` is the
            number of particles and ``D`` the dimension.
    boxsize
            Scalar or ``(D,)`` array defining periodic box lengths, or ``None``.

    Returns
    -------
    scipy.spatial.cKDTree
            Tree built from ``points``.

    """
    return spatial.cKDTree(points, boxsize=boxsize)


def query_kdtree(
    tree: spatial.cKDTree, points: npt.ArrayLike, k: int
) -> tuple[FloatArray, IntArray]:
    """Query a cKDTree for the k nearest neighbors of given points.

    Parameters
    ----------
    tree
            cKDTree instance to query.
    points
            Array of shape ``(M, D)`` with query coordinates, where ``M`` is the
            number of query positions.
    k
            Number of nearest neighbors to return.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
            Tuple of ``(distances, indices)`` from ``cKDTree.query``.

    """
    return tree.query(points, k=k, workers=-1)


def shift_coordinates(coordinates: npt.ArrayLike) -> FloatArray:
    """Shift coordinates so the minimum per-axis value is at zero.

    Parameters
    ----------
    coordinates
            Array of shape ``(N, D)`` with particle coordinates.

    Returns
    -------
    numpy.ndarray
            Shifted coordinates with the same shape as ``coordinates``.

    Raises
    ------
    AssertionError
            If the input array is not 2D.

    """
    coordinates_array = np.asarray(coordinates)
    assert (
        coordinates_array.ndim == 2
    ), "Input coordinates must be a 2D array of shape (N, D)"
    return coordinates_array - coordinates_array.min(axis=0)


def coordinate_difference_with_pbc(
    x: npt.ArrayLike, y: npt.ArrayLike, boxsize: BoxInput
) -> FloatArray:
    """Compute coordinate differences with periodic boundary conditions.

    Parameters
    ----------
    x
            Array-like coordinates.
    y
            Array-like coordinates to subtract from ``x``.
    boxsize
            Scalar or ``(D,)`` array defining periodic box lengths.

    Returns
    -------
    numpy.ndarray
            Coordinate differences wrapped into the range
            $[-0.5 * boxsize, 0.5 * boxsize)$ per dimension.

    Raises
    ------
    AssertionError
            If the input coordinate arrays do not have the same spatial dimension.
            If ``boxsize`` is an array, if it is not 1D or if its length does not match the coordinate dimension.

    """
    shape_x = x.shape
    shape_y = y.shape
    assert (
        shape_x[-1] == shape_y[-1]
    ), "Input coordinate arrays must have the same spatial dimension"

    # compute differences
    coordinate_differences = np.asarray(x) - np.asarray(y)

    # early exit if boxsize is None (non-periodic case)
    if boxsize is None:
        return coordinate_differences

    # prepare boxsize array for broadcasting
    if np.isscalar(boxsize):
        box_arr = np.array([boxsize] * shape_x[-1])
    else:
        assert np.asarray(boxsize).ndim == 1, "'boxsize' must be a scalar or 1D array"
        assert (
            len(boxsize) == shape_x[-1]
        ), "Length of 'boxsize' must match coordinate dimension"
        box_arr = np.asarray(boxsize)

    half_box = 0.5 * box_arr
    return (coordinate_differences + half_box) % box_arr - half_box
