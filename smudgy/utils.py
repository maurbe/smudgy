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
        box_arr = np.array([boxsize] * shape_x.shape[-1])
    else:
        assert np.asarray(boxsize).ndim == 1, "'boxsize' must be a scalar or 1D array"
        assert (
            len(boxsize) == shape_x[-1]
        ), "Length of 'boxsize' must match coordinate dimension"
        box_arr = np.asarray(boxsize)

    half_box = 0.5 * box_arr
    return (coordinate_differences + half_box) % box_arr - half_box


def compute_hsm(
    tree: spatial.cKDTree,
    num_neighbors: int,
    query_positions: npt.ArrayLike | None = None,
) -> tuple[FloatArray, IntArray, FloatArray]:
    """Estimate smoothing length as half the distance to the Nth neighbor.

    Parameters
    ----------
    tree
            cKDTree built from particle positions.
    num_neighbors
            Number of neighbors used for the estimate.
    query_positions
            Array of shape ``(M, D)`` with positions where smoothing length is evaluated.
            If ``None``, uses particle positions from the tree.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Tuple of ``(hsm, nn_inds, nn_dists)`` where ``hsm`` has shape ``(M,)``,
            ``nn_dists`` has shape ``(M, num_neighbors)``, and ``nn_inds`` has shape
            ``(M, num_neighbors)``.

    """
    if query_positions is None:
        query_positions = tree.data

    nn_dists, nn_inds = query_kdtree(tree, query_positions, k=num_neighbors)
    hsm = nn_dists[:, -1] * 0.5
    return hsm, nn_inds, nn_dists


def compute_hsm_tensor(
    tree: spatial.cKDTree,
    weights: npt.ArrayLike,
    num_neighbors: int,
    query_positions: npt.ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, IntArray, FloatArray]:
    """Compute anisotropic smoothing tensor using a covariance-based method.

    Implements the method from Marinho (2021), generalized for 2D and 3D.

    Parameters
    ----------
    tree
            cKDTree built from particle positions.
    weights
            Array of shape ``(N,)`` with particle weights.
    num_neighbors
            Number of neighbors used for covariance estimation.
    query_positions
            Array of shape ``(M, D)`` where the tensor is evaluated.
            If ``None``, uses particle positions from the tree.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Tuple ``(H, eigvals, eigvecs, nn_inds, nn_dists, rel_coords)`` where
            ``H`` has shape ``(M, D, D)``, ``eigvals`` has shape ``(M, D)``,
            ``eigvecs`` has shape ``(M, D, D)``, ``nn_inds`` has shape
            ``(M, num_neighbors)``, ``nn_dists`` has shape ``(M, num_neighbors)``,
            and ``rel_coords`` has shape ``(M, num_neighbors, D)``.

    """
    pos = tree.data
    boxsize = tree.boxsize
    spatial_dim = pos.shape[-1]

    if spatial_dim not in (2, 3):
        raise ValueError(
            "Only 2D and 3D positions are supported for anisotropic smoothing tensors."
        )

    if query_positions is None:
        query_positions = pos
    weights = weights.flatten() if weights.ndim > 1 else weights

    # Find nearest neighbors
    nn_dists, nn_inds = query_kdtree(tree, query_positions, k=num_neighbors)
    neighbor_coords = pos[nn_inds]
    neighbor_weights = weights[nn_inds]

    # Account for periodic boundary conditions
    rel_coords = coordinate_difference_with_pbc(
        neighbor_coords, query_positions[:, np.newaxis, :], boxsize
    )

    # Compute mass-weighted covariance matrix
    outer = np.einsum("...i, ...j -> ...ij", rel_coords, rel_coords)
    outer = outer * neighbor_weights[..., np.newaxis, np.newaxis]
    Sigma = (
        np.sum(outer, axis=1)
        / np.sum(neighbor_weights, axis=1)[..., np.newaxis, np.newaxis]
    )

    # Compute eigendecomposition and construct smoothing tensor H = VΛV^T
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.sqrt(eigvals)
    Λ = eigvals[..., np.newaxis] * np.eye(spatial_dim)
    H = np.matmul(np.matmul(eigvecs, Λ), np.transpose(eigvecs, axes=(0, 2, 1)))

    return H, eigvals, eigvecs, nn_inds, nn_dists, rel_coords


def project_hsm_tensor_to_2d(
    h_tensor: npt.ArrayLike,
    plane: str | None = None,
    basis: tuple[Sequence[float], Sequence[float]] | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Project 3D smoothing tensors onto a 2D plane.

    Parameters
    ----------
    h_tensor
            Array of shape ``(N, 3, 3)`` with 3D smoothing tensors.
    plane
            Projection plane: ``"xy"``, ``"xz"``, or ``"yz"``.
            Mutually exclusive with ``basis``.
    basis
            2-tuple of basis vectors ``(e1, e2)`` spanning the projection plane.
            Each vector should be array-like of length 3.
            Mutually exclusive with ``plane``.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Tuple ``(h_tensor_2d, eigvals, eigvecs)`` where ``h_tensor_2d`` has
            shape ``(N, 2, 2)``, ``eigvals`` has shape ``(N, 2)``, and ``eigvecs``
            has shape ``(N, 2, 2)``.

    Raises
    ------
    ValueError
            If neither or both of ``plane`` and ``basis`` are provided.
            If ``plane`` is not one of the allowed values.
            If ``basis`` is not a 2-tuple of 3D vectors.
    """
    # Validate inputs
    if plane is None and basis is None:
        raise ValueError("Either 'plane' or 'basis' must be provided")
    if plane is not None and basis is not None:
        raise ValueError(
            "'plane' and 'basis' are mutually exclusive, only provide one of the two"
        )

    # Define projection basis vectors
    if basis is not None:
        if len(basis) != 2:
            raise ValueError("'basis' must be a 2-tuple of vectors")
        e1, e2 = basis
    else:
        if plane == "xy":
            e1, e2 = [1, 0, 0], [0, 1, 0]
        elif plane == "xz":
            e1, e2 = [1, 0, 0], [0, 0, 1]
        elif plane == "yz":
            e1, e2 = [0, 1, 0], [0, 0, 1]
        else:
            raise ValueError("'plane' must be one of 'xy', 'xz', or 'yz'")

    # Compute projected tensors: (P @ H^-1 @ P^T)^-1
    projection_matrix = np.array([e1, e2], dtype="float32")  # (2, 3)
    h_tensor_inv = np.linalg.inv(h_tensor)  # (N, 3, 3)

    # Vectorized computation: P @ H_inv @ P^T for all particles
    temp = np.einsum(
        "ij,njk,lk->nil", projection_matrix, h_tensor_inv, projection_matrix
    )
    h_tensor_2d = np.linalg.inv(temp)  # (N, 2, 2)

    # Compute eigendecomposition of 2D tensors
    eigvals, eigvecs = np.linalg.eigh(h_tensor_2d)
    return h_tensor_2d, eigvals, eigvecs
