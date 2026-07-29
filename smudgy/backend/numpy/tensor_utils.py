# backend/numpy/tensor_utils.py
import numpy as np

E1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
E2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def project_2d(h_tensor: np.ndarray, e1: np.ndarray = E1, e2: np.ndarray = E2):
    P = np.stack([e1, e2]).astype(np.float32)  # (2, D)
    h_tensor_inv = np.linalg.inv(h_tensor)  # (N, D, D)
    temp = P @ h_tensor_inv @ P.T  # (N, 2, 2), batched matmul
    h_tensor_2d = np.linalg.inv(temp)
    eigvals, eigvecs = np.linalg.eigh(h_tensor_2d)
    return (
        h_tensor_2d.astype(np.float32),
        eigvals.astype(np.float32),
        eigvecs.astype(np.float32),
    )


"""Utility functions for SPH operations."""
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy import spatial

FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.int_]
BoxInput = float | Sequence[float] | npt.ArrayLike
"""
"""
def project_smoTens_to_2d(
    h_tensor: npt.ArrayLike,
    plane: list[int] | None = None,
    basis: list[Sequence[float], Sequence[float]] | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
"""
"""Project 3D smoothing tensors onto a 2D plane.

    Parameters
    ----------
    h_tensor
            Array of shape ``(N, 3, 3)`` with 3D smoothing tensors.
    plane
            Indices of the axes to project onto for 3D to 2D deposition.
            Mutually exclusive with ``basis``.
    basis
            List of basis vectors ``(e1, e2)`` spanning the projection plane.
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
            If ``plane`` is not one of the allowed values or types or does not have length 2.
            If ``basis`` is not a 2-tuple of 3D vectors.

"""
"""
    # Validate inputs
    if plane is None and basis is None:
        raise ValueError("Either 'plane' or 'basis' must be provided")
    if plane is not None and basis is not None:
        raise ValueError(
            "'plane' and 'basis' are mutually exclusive, only provide one of the two"
        )

    # Validate that plane is either list or array of length 2 with valid indices
    if plane is not None:
        if not (isinstance(plane, (list, np.ndarray)) and len(plane) == 2):
            raise ValueError("'plane' must be a list or array of length 2")
        if any(p not in [0, 1, 2] for p in plane):
            raise ValueError("'plane' indices must be in the range [0, 2]")

    # Define projection basis vectors
    if basis is not None:
        if len(basis) != 2:
            raise ValueError("'basis' must be a 2-tuple of vectors")
        e1, e2 = basis
    else:
        unit_vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        e1 = unit_vectors[plane[0]]
        e2 = unit_vectors[plane[1]]

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
"""
