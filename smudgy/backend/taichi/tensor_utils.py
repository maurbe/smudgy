"""Utilities for tensor operations in Taichi backend."""

from typing import Tuple
import numpy as np
import taichi as ti

E1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
E2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
E3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
BASIS_VECTORS = [E1, E2, E3]


@ti.kernel
def _project_2d_kernel(
    h_tensor: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (N, 3, 3)
    e1: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (3,)
    e2: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (3,)
    h2d_out: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (N, 2, 2)
    eigvals_out: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 2)
    eigvecs_out: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (N, 2, 2)
):
    P = ti.Matrix.zero(ti.f32, 2, 3)
    for j in ti.static(range(3)):
        P[0, j] = e1[j]
        P[1, j] = e2[j]

    for n in range(h_tensor.shape[0]):
        H = ti.Matrix.zero(ti.f32, 3, 3)
        for a in ti.static(range(3)):
            for b in ti.static(range(3)):
                H[a, b] = h_tensor[n, a, b]

        h2d = (P @ H.inverse() @ P.transpose()).inverse()
        eigvals, eigvecs = ti.sym_eig(h2d, ti.f32)

        # ==========================================================================================
        # Fix for annyoing taichi bug in 2D:
        # eigvals/vecs are returned in descending order, 3D is fine and follows numpy backend
        if eigvals[0] > eigvals[1]:
            eigvals[0], eigvals[1] = eigvals[1], eigvals[0]
            for a in ti.static(range(2)):
                eigvecs[a, 0], eigvecs[a, 1] = eigvecs[a, 1], eigvecs[a, 0]

        # Eigenvector orientation convention:
        # Numpy and Taichi have no convention about the sign of the eigenvectors,
        # impose a consistent sign convention for comparison with Taichi backend
        # (largest-magnitude component of each eigenvector is positive).
        for col in ti.static(range(2)):
            max_abs = 0.0
            sign = 1.0
            for row in ti.static(range(2)):
                v = eigvecs[row, col]
                if ti.abs(v) > max_abs:
                    max_abs = ti.abs(v)
                    sign = 1.0 if v >= 0.0 else -1.0
            for row in ti.static(range(2)):
                eigvecs[row, col] *= sign
        # ==========================================================================================

        for a in ti.static(range(2)):
            eigvals_out[n, a] = eigvals[a]
            for b in ti.static(range(2)):
                h2d_out[n, a, b] = h2d[a, b]
                eigvecs_out[n, a, b] = eigvecs[a, b]


def project_2d(h_tensor: np.ndarray, 
               plane: Tuple[int, int] = (0, 1)):
    """Project a 3D smoothing tensor to 2D and compute its eigenvalues and eigenvectors.

    Parameters
    ----------
    h_tensor : np.ndarray
        Array of 3D smoothing tensors of shape (N, 3, 3).
    plane : Tuple[int, int], optional
        Indices of the basis vectors for the 2D plane, default is (0, 1).

    Returns
    -------
    h2d : np.ndarray
        Array of projected 2D smoothing tensors of shape (N, 2, 2).
    eigvals : np.ndarray
        Array of eigenvalues of the projected 2D tensors of shape (N, 2).
    eigvecs : np.ndarray
        Array of eigenvectors of the projected 2D tensors of shape (N, 2, 2).

    """
    h_tensor = np.ascontiguousarray(h_tensor, dtype=np.float32)
    ea = np.ascontiguousarray(BASIS_VECTORS[plane[0]], dtype=np.float32)
    eb = np.ascontiguousarray(BASIS_VECTORS[plane[1]], dtype=np.float32)

    N = h_tensor.shape[0]
    h2d = np.zeros((N, 2, 2), dtype=np.float32)
    eigvals = np.zeros((N, 2), dtype=np.float32)
    eigvecs = np.zeros((N, 2, 2), dtype=np.float32)

    _project_2d_kernel(h_tensor, ea, eb, h2d, eigvals, eigvecs)
    return h2d, eigvals, eigvecs
