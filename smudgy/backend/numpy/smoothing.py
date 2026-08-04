# backend/numpy/smoothing.py
import numpy as np
import numpy.typing as npt

from ..neighbors import coordinate_difference_with_pbc

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int32]

_EPS = 1e-7


def _as_float32(array):
    """Return a float32 view of ``array`` without copying when possible.

    Parameters
    ----------
    array
        Input array-like object.

    Returns
    -------
    numpy.ndarray
        ``float32`` view or copy of ``array``.

    """
    return np.asarray(array, dtype=np.float32)


def compute_hsml(nn_dists: FloatArray) -> FloatArray:
    nn_dists = _as_float32(nn_dists)
    return nn_dists[:, -1]


def compute_hmat(
    query_positions: FloatArray,
    neighbor_positions: FloatArray,  # (N, K, D) -- pre-gathered
    neighbor_weights: FloatArray,  # (N, K)    -- pre-gathered
    boxsize: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:

    dim = query_positions.shape[-1]
    if dim not in (2, 3):
        raise ValueError(
            "[smudgy] Only 2D and 3D positions are supported for anisotropic smoothing tensors."
        )

    query_positions = _as_float32(query_positions)
    neighbor_positions = _as_float32(neighbor_positions)
    neighbor_weights = _as_float32(neighbor_weights)
    boxsize = _as_float32(boxsize)

    rel_coords = coordinate_difference_with_pbc(
        neighbor_positions, query_positions[:, np.newaxis, :], boxsize
    )

    w = neighbor_weights / neighbor_weights.sum(axis=1, keepdims=True)
    Sigma = np.einsum("nk,nki,nkj->nij", w, rel_coords, rel_coords, optimize=True)
    Sigma += (
        _EPS
        * np.trace(Sigma, axis1=-2, axis2=-1)[..., None, None]
        * np.eye(dim, dtype=np.float32)
    )

    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.sqrt(np.clip(eigvals, 0, None))
    # eigvals, eigvecs should be in descending order
    # numpy: already ascending, but be explicit
    # order = np.argsort(eigvals, axis=-1, descending=True)
    # eigvals = np.take_along_axis(eigvals, order, axis=-1)
    # eigvecs = np.take_along_axis(eigvecs, order[..., None, :], axis=-2)

    H = np.einsum("nij,nj,nkj->nik", eigvecs, eigvals, eigvecs, optimize=True)

    return (
        H.astype(np.float32),
        eigvals.astype(np.float32),
        eigvecs.astype(np.float32),
        rel_coords.astype(np.float32),
    )
