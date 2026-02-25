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
    """Construct a cKDTree with optional per-dimension periodic box sizes.

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


def shift_particle_positions(pos: npt.ArrayLike) -> FloatArray:
    """Shift particle positions so the minimum per-axis value is at zero.

    Parameters
    ----------
    pos
            Array of shape ``(N, D)`` with particle coordinates.

    Returns
    -------
    numpy.ndarray
            Shifted coordinates with the same shape as ``pos``.

    """
    pos_array = np.asarray(pos)
    return pos_array - pos_array.min(axis=0)


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

    """
    diff = np.asarray(x) - np.asarray(y)

    # early exit if boxsize is None (non-periodic case)
    if boxsize is None:
        return diff

    box_arr = np.asarray(boxsize)
    if box_arr.ndim == 0:
        half_box = 0.5 * box_arr
        return (diff + half_box) % box_arr - half_box

    if box_arr.ndim != 1:
        raise ValueError("'boxsize' must be a scalar or 1D array")

    if diff.ndim == 0:
        raise ValueError("Vector 'boxsize' requires vector inputs for x and y")

    if diff.shape[-1] != box_arr.shape[0]:
        raise ValueError("Dimension mismatch between coordinates and 'boxsize'")

    half_box = 0.5 * box_arr
    return (diff + half_box) % box_arr - half_box


def compute_pcellsize_half(
    tree: spatial.cKDTree,
    num_neighbors: int,
    query_pos: npt.ArrayLike | None = None,
) -> tuple[FloatArray, IntArray]:
    """Estimate rectangular particle extent as half-distance to the Nth neighbor.

    Parameters
    ----------
    tree
            cKDTree built from particle positions.
    num_neighbors
            Number of neighbors used for the estimate.
    query_pos
            Array of shape ``(M, D)`` with query coordinates.
            If ``None``, uses particle positions from the tree.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
            Tuple of ``(h_cellsize, nn_inds)`` where ``h_cellsize`` has shape ``(M,)``
            and contains half the distance to the Nth neighbor, and ``nn_inds`` has
            shape ``(M, num_neighbors)``.

    """
    if query_pos is None:
        query_pos = tree.data

    # smoothing length as 0.5 * distance to Nth nearest neighbor
    nn_dists, nn_inds = query_kdtree(tree, query_pos, k=num_neighbors)
    h_cellsize = nn_dists[:, -1] * 0.5
    return h_cellsize, nn_inds


def compute_hsm(
    tree: spatial.cKDTree,
    num_neighbors: int,
    query_pos: npt.ArrayLike | None = None,
) -> tuple[FloatArray, IntArray, FloatArray]:
    """Estimate smoothing length as half the distance to the Nth neighbor.

    Parameters
    ----------
    tree
            cKDTree built from particle positions.
    num_neighbors
            Number of neighbors used for the estimate.
    query_pos
            Array of shape ``(M, D)`` with positions where smoothing length is evaluated.
            If ``None``, uses particle positions from the tree.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Tuple of ``(hsm, nn_inds, nn_dists)`` where ``hsm`` has shape ``(M,)``,
            ``nn_dists`` has shape ``(M, num_neighbors)``, and ``nn_inds`` has shape
            ``(M, num_neighbors)``.

    """
    if query_pos is None:
        query_pos = tree.data

    nn_dists, nn_inds = query_kdtree(tree, query_pos, k=num_neighbors)
    hsm = nn_dists[:, -1] * 0.5
    return hsm, nn_inds, nn_dists


def compute_hsm_tensor(
    tree: spatial.cKDTree,
    weights: npt.ArrayLike,
    num_neighbors: int,
    query_pos: npt.ArrayLike | None = None,
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
    query_pos
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
    # Get particle positions and boxsize from the tree object
    pos = tree.data
    boxsize = tree.boxsize
    D = pos.shape[-1]

    if D not in (2, 3):
        raise ValueError(
            "Only 2D and 3D positions are supported for anisotropic smoothing tensors."
        )

    # Use particle positions if query_pos not provided
    if query_pos is None:
        query_pos = pos

    # Ensure weights is 1D
    weights = weights.flatten() if weights.ndim > 1 else weights

    # Find nearest neighbors
    nn_dists, nn_inds = query_kdtree(tree, query_pos, k=num_neighbors)
    neighbor_coords = pos[nn_inds]
    neighbor_weights = weights[nn_inds]

    # Account for periodic boundary conditions
    rel_coords = coordinate_difference_with_pbc(
        neighbor_coords, query_pos[:, np.newaxis, :], boxsize
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
    Λ = eigvals[..., np.newaxis] * np.eye(D)
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

    """
    # Validate inputs
    if plane is None and basis is None:
        raise ValueError("Either 'plane' or 'basis' must be provided")
    if plane is not None and basis is not None:
        raise ValueError("'plane' and 'basis' are mutually exclusive")

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
