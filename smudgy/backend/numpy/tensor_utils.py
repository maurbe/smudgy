# backend/numpy/tensor_utils.py
import numpy as np

E1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
E2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

def _fix_eigvec_sign(eigvecs: np.ndarray) -> np.ndarray:
    """Force the largest-magnitude component of each eigenvector to be positive."""
    # eigvecs shape (..., D, D), columns are eigenvectors
    idx = np.argmax(np.abs(eigvecs), axis=-2, keepdims=True)  # (..., 1, D)
    sign = np.take_along_axis(eigvecs, idx, axis=-2)          # (..., 1, D)
    sign = np.where(sign >= 0, 1.0, -1.0)
    return eigvecs * sign


def project_2d(h_tensor: np.ndarray, e1: np.ndarray = E1, e2: np.ndarray = E2):
    P = np.stack([e1, e2]).astype(np.float32)  # (2, D)
    h_tensor_inv = np.linalg.inv(h_tensor)  # (N, D, D)
    temp = P @ h_tensor_inv @ P.T  # (N, 2, 2), batched matmul
    h_tensor_2d = np.linalg.inv(temp)
    eigvals, eigvecs = np.linalg.eigh(h_tensor_2d)

    # Eigenvector orientation convention:
    # Numpy and Taichi have no convention about the sign of the eigenvectors,
    # impose a consistent sign convention for comparison with Taichi backend 
    # (largest-magnitude component of each eigenvector is positive).
    eigvecs = _fix_eigvec_sign(eigvecs)

    return (
        h_tensor_2d.astype(np.float32),
        eigvals.astype(np.float32),
        eigvecs.astype(np.float32),
    )
