"""Eigenvector conventions and plane projection of smoothing tensors (NumPy)."""

import numpy as np

from .._common import BASIS_VECTORS


def _fix_eigvec_sign(eigvecs: np.ndarray) -> np.ndarray:
    """Force the largest-magnitude component of each eigenvector to be positive."""
    # eigvecs shape (..., D, D), columns are eigenvectors
    idx = np.argmax(np.abs(eigvecs), axis=-2, keepdims=True)  # (..., 1, D)
    sign = np.take_along_axis(eigvecs, idx, axis=-2)  # (..., 1, D)
    sign = np.where(sign >= 0, 1.0, -1.0)
    return eigvecs * sign


def project_2d(h_tensor: np.ndarray, plane: tuple[int, int] = (0, 1)):
    """Shadow of 3D smoothing tensors on a coordinate plane.

    ``H_2 = (P H H^T P^T)^(1/2)``, with ``P`` the (2, 3) matrix of the plane's
    basis vectors: the outline of the smoothing ellipsoid seen along the line
    of sight, i.e. the shape of the 3D kernel integrated along that line.

    Parameters
    ----------
    h_tensor : np.ndarray
        Smoothing tensors, shape (N, 3, 3).
    plane : tuple of int, default (0, 1)
        Axes spanning the projection plane.

    Returns
    -------
    h2d : np.ndarray
        Projected tensors, shape (N, 2, 2).
    eigvals : np.ndarray
        Eigenvalues (principal semi-axes) of the projected tensors, shape
        (N, 2), descending.
    eigvecs : np.ndarray
        Eigenvectors of the projected tensors, shape (N, 2, 2).
    """
    P = BASIS_VECTORS[list(plane)].astype(np.float64)  # (2, 3)
    PH = P @ np.asarray(h_tensor, dtype=np.float64)  # (N, 2, 3)
    # Shadow of the ellipsoid: Sigma_2 = P H H^T P^T, H_2 = Sigma_2^(1/2)
    sigma_2d = PH @ np.swapaxes(PH, -1, -2)
    lam, eigvecs = np.linalg.eigh(sigma_2d)  # ascending
    # Descending, matching compute_hmat; columns follow the values.
    eigvals = np.sqrt(np.maximum(lam[:, ::-1], 0.0))
    # Eigenvector sign convention shared with the Quadrants backend
    # (largest-magnitude component of each eigenvector is positive).
    eigvecs = _fix_eigvec_sign(eigvecs[:, :, ::-1])
    h_tensor_2d = np.einsum("nij,nj,nkj->nik", eigvecs, eigvals, eigvecs)

    return (
        h_tensor_2d.astype(np.float32),
        eigvals.astype(np.float32),
        eigvecs.astype(np.float32),
    )
