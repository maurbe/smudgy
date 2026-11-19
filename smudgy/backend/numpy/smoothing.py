# backend/numpy/smoothing.py
import numpy as np
import numpy.typing as npt

from .._common import EPS, as_f32, check_hmat_args
from ..neighbors import coordinate_difference_with_pbc
from .tensor_utils import _fix_eigvec_sign

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int32]


def compute_hsml(nn_dists: FloatArray) -> FloatArray:
    """Return the distance to the farthest (k-th) neighbour as smoothing length."""
    return as_f32(nn_dists[:, -1])


def compute_hmat(
    query_positions: FloatArray,  # (N, D)
    positions: FloatArray,  # (Nc, D), indexed by nn_inds
    weights: FloatArray,  # (Nc,),  indexed by nn_inds
    nn_inds: IntArray,  # (N, K)
    boxsize: FloatArray | None = None,
    ratio_max: float = 100.0,  # max axis ratio h_max / h_min
    min_axis: float | None = None,  # physical floor (softening / cell size)
    volume_preserving: bool = False,  # det H = hsml^D instead of the hull scaling
    hsml: FloatArray | None = None,  # required if volume_preserving
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute covariant smoothing tensors (Marinho 2021); NumPy reference.

    ``H = zeta_max * Sigma^(1/2)``, the smallest ellipsoid of the neighbours'
    covariance shape that encloses all neighbours.

    Parameters
    ----------
    query_positions : np.ndarray
        Query positions, shape (N, D).
    positions : np.ndarray
        Particle positions, shape (Nc, D), indexed by `nn_inds`.
    weights : np.ndarray
        Particle weights, shape (Nc,), indexed by `nn_inds`.
    nn_inds : np.ndarray
        Neighbour indices of each query, shape (N, K).
    boxsize : np.ndarray, optional
        Box sizes for periodic boundary conditions.
    ratio_max : float, default 100.0
        Maximum axis ratio h_max / h_min (>= 1).
    min_axis : float, optional
        Lower bound on every axis.
    volume_preserving : bool, default False
        Rescale the axes so that det H = hsml^D. Requires `hsml`.
    hsml : np.ndarray, optional
        Isotropic smoothing lengths, shape (N,).

    Returns
    -------
    H : np.ndarray
        Smoothing tensors, shape (N, D, D).
    eigvals : np.ndarray
        Principal smoothing lengths in descending order, shape (N, D).
    eigvecs : np.ndarray
        Principal axes (columns), shape (N, D, D).
    """
    dim = query_positions.shape[-1]
    min_axis = check_hmat_args(
        dim, ratio_max, min_axis, volume_preserving, hsml, query_positions.shape[0]
    )
    if volume_preserving:
        hsml = np.asarray(hsml, dtype=np.float64)

    query_positions = as_f32(query_positions)
    neighbor_positions = as_f32(positions)[nn_inds]
    neighbor_weights = as_f32(weights)[nn_inds]
    boxsize = as_f32(boxsize) if boxsize is not None else None

    rel_coords = coordinate_difference_with_pbc(
        neighbor_positions, query_positions[:, np.newaxis, :], boxsize
    )

    # --- second-moment tensor about the query (Marinho eq. 37) ----------------
    rel64 = rel_coords.astype(np.float64)
    w = neighbor_weights.astype(np.float64)
    w /= w.sum(axis=1, keepdims=True)
    Sigma = np.einsum("nk,nki,nkj->nij", w, rel64, rel64, optimize=True)
    Sigma += (
        EPS
        * np.trace(Sigma, axis1=-2, axis2=-1)[..., None, None]
        * np.eye(dim, dtype=np.float64)
    )

    # eigh in float64: Sigma is near-singular exactly in the cases that matter
    eigvals, eigvecs = np.linalg.eigh(Sigma)  # ascending
    eigvals = eigvals[:, ::-1]  # -> descending
    eigvecs = _fix_eigvec_sign(eigvecs[:, :, ::-1])  # columns follow the values

    # --- shape: sqrt of the eigenvalues, with an axis-ratio clamp ------------
    sigma = np.sqrt(np.clip(eigvals, 0.0, None))  # principal components
    sigma = np.maximum(sigma, sigma[:, :1] / ratio_max)  # sigma[:, 0] is the largest
    lam_clamped = sigma**2  # used for the Mahalanobis metric

    # --- scale: the enclosing-hull condition (eq. 39) ------------------------
    # lam_clamped is only zero when Sigma is (all neighbours at the query);
    # zeta_max = 0 then, i.e. H = 0, matching the Quadrants backend.
    inv_lam = np.divide(
        1.0, lam_clamped, out=np.zeros_like(lam_clamped), where=lam_clamped > 0
    )
    proj = np.einsum("nkd,ndj->nkj", rel64, eigvecs, optimize=True)
    maha2 = np.einsum("nkj,nj->nk", proj**2, inv_lam, optimize=True)
    zeta_max = np.sqrt(maha2.max(axis=1))  # eq. 39

    axes = zeta_max[:, None] * sigma  # eq. 44: h_1, h_2, h_3

    if volume_preserving:  # det H = hsml^D
        det = np.maximum(np.prod(axes, axis=1), 1e-300)
        axes *= (hsml / det ** (1.0 / dim))[:, None]
    if min_axis > 0.0:  # only ever enlarges axes
        axes = np.maximum(axes, min_axis)

    H = np.einsum("nij,nj,nkj->nik", eigvecs, axes, eigvecs, optimize=True)

    return (
        H.astype(np.float32),
        axes.astype(np.float32),  # eigenvalues of H, descending
        eigvecs.astype(np.float32),  # eigenvectors of H (columns)
    )
