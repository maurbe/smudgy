"""k-d tree neighbour search and periodic coordinate differences."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy import spatial

from ._node import kdtree_workers

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
    # Unbalanced, non-compacted nodes: ~2x faster to build and no slower (here
    # slightly faster) to query than scipy's median-split defaults.
    return spatial.cKDTree(
        points, boxsize=boxsize, balanced_tree=False, compact_nodes=False
    )


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
    return tree.query(points, k=k, workers=kdtree_workers())


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
    ValueError
            If the coordinate arrays have different spatial dimensions, or
            ``boxsize`` is neither a scalar nor of shape ``(D,)``.

    """
    shape_x = np.shape(x)
    if shape_x[-1] != np.shape(y)[-1]:
        raise ValueError("Input coordinate arrays must have the same spatial dimension")

    # compute differences
    coordinate_differences = np.asarray(x) - np.asarray(y)

    # early exit if boxsize is None (non-periodic case)
    if boxsize is None:
        return coordinate_differences

    # prepare boxsize array for broadcasting
    box_arr = np.asarray(boxsize)
    if box_arr.shape not in ((), (shape_x[-1],)):
        raise ValueError(
            f"'boxsize' must be a scalar or have shape ({shape_x[-1]},), got {box_arr.shape}"
        )

    half_box = 0.5 * box_arr
    return (coordinate_differences + half_box) % box_arr - half_box
