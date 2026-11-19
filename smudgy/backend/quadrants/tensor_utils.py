"""Utilities for tensor operations in Quadrants backend."""

import numpy as np
import quadrants as qd

from .linalg import sym_eig_descending

E1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
E2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
E3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
BASIS_VECTORS = [E1, E2, E3]


@qd.kernel
def _project_2d_kernel(
    h_tensor: qd.types.ndarray(dtype=qd.f32, ndim=3),  # (N, 3, 3)
    e1: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (3,)
    e2: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (3,)
    h2d_out: qd.types.ndarray(dtype=qd.f32, ndim=3),  # (N, 2, 2)
    eigvals_out: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, 2)
    eigvecs_out: qd.types.ndarray(dtype=qd.f32, ndim=3),  # (N, 2, 2)
):
    P = qd.Matrix.zero(qd.f32, 2, 3)
    for j in qd.static(range(3)):
        P[0, j] = e1[j]
        P[1, j] = e2[j]

    for n in range(h_tensor.shape[0]):
        H = qd.Matrix.zero(qd.f32, 3, 3)
        for a in qd.static(range(3)):
            for b in qd.static(range(3)):
                H[a, b] = h_tensor[n, a, b]

        h2d = (P @ H.inverse() @ P.transpose()).inverse()
        h2d = 0.5 * (h2d + h2d.transpose())  # symmetric up to f32 round-off

        # Descending, same convention as compute_hmat (and the numpy backend).
        eigvals, eigvecs = sym_eig_descending(h2d, 2)

        for a in qd.static(range(2)):
            eigvals_out[n, a] = eigvals[a]
            for b in qd.static(range(2)):
                h2d_out[n, a, b] = h2d[a, b]
                eigvecs_out[n, a, b] = eigvecs[a, b]


def project_2d(h_tensor: np.ndarray, plane: tuple[int, int] = (0, 1)):
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
