"""Utilities for tensor operations in Quadrants backend."""

import numpy as np
import quadrants as qd

from .._common import BASIS_VECTORS, as_f32
from .linalg import sym_eig_descending


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

        # Shadow of the ellipsoid: covariance Sigma = H H^T projected onto the
        # plane. Integrating the 3D kernel along the line of sight leaves a
        # kernel with this shape (Schur complement of Sigma^-1).
        PH = P @ H
        sigma2d = PH @ PH.transpose()
        sigma2d = 0.5 * (sigma2d + sigma2d.transpose())  # symmetric up to f32 round-off

        # Descending, same convention as compute_hmat (and the numpy backend).
        lam, eigvecs = sym_eig_descending(sigma2d, 2)

        # H_2 = Sigma_2^(1/2): same eigenvectors, square-rooted eigenvalues
        eigvals = qd.Vector.zero(qd.f32, 2)
        for a in qd.static(range(2)):
            eigvals[a] = qd.sqrt(qd.max(lam[a], 0.0))
        for a in qd.static(range(2)):
            eigvals_out[n, a] = eigvals[a]
            for b in qd.static(range(2)):
                h2d = 0.0
                for c in qd.static(range(2)):
                    h2d += eigvecs[a, c] * eigvals[c] * eigvecs[b, c]
                h2d_out[n, a, b] = h2d
                eigvecs_out[n, a, b] = eigvecs[a, b]


def project_2d(h_tensor: np.ndarray, plane: tuple[int, int] = (0, 1)):
    """Shadow of 3D smoothing tensors on a coordinate plane, with its eigensystem.

    The projected tensor is ``H_2 = (P H H^T P^T)^(1/2)``, with ``P`` the
    (2, 3) matrix of the plane's basis vectors: the outline of the smoothing
    ellipsoid seen along the line of sight. The 3D kernel integrated along
    that line has this shape (its radial profile is the projected 3D
    profile, not the 2D kernel).

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
        Array of eigenvalues (principal semi-axes) of the projected 2D
        tensors of shape (N, 2), descending.
    eigvecs : np.ndarray
        Array of eigenvectors of the projected 2D tensors of shape (N, 2, 2).

    """
    h_tensor = as_f32(h_tensor)
    ea = as_f32(BASIS_VECTORS[plane[0]])
    eb = as_f32(BASIS_VECTORS[plane[1]])

    N = h_tensor.shape[0]
    h2d = np.zeros((N, 2, 2), dtype=np.float32)
    eigvals = np.zeros((N, 2), dtype=np.float32)
    eigvecs = np.zeros((N, 2, 2), dtype=np.float32)

    _project_2d_kernel(h_tensor, ea, eb, h2d, eigvals, eigvecs)
    return h2d, eigvals, eigvecs
