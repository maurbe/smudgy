# backend/numpy/smoothing.py
import numpy as np
import numpy.typing as npt

from ..neighbors import coordinate_difference_with_pbc
from .tensor_utils import _fix_eigvec_sign

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
    """Covariance-based smoothing tensor, following Marinho (2021), eqs. 39-44.

    H = zeta_max * Sigma^(1/2), with zeta_max the largest Mahalanobis distance
    among the K neighbours, so the support ellipsoid is the smallest one of the
    covariance's shape that still contains every neighbour. In the isotropic
    limit each axis reduces to the distance to the farthest neighbour, i.e. to
    the usual smoothing length, for any K and any weighting.

    Returns (H, axes, eigvecs), where `axes` and `eigvecs` are the
    eigenvalues and eigenvectors of H itself: H = V diag(axes) V^T. Axes are
    sorted in descending order; eigenvectors follow the sign convention of
    `_fix_eigvec_sign` (largest-magnitude component positive).

    If every neighbour coincides with the query, Sigma vanishes and H = 0; pass
    `min_axis` to keep H invertible in that case.
    """
    dim = query_positions.shape[-1]
    if dim not in (1, 2, 3):
        raise ValueError(
            "[smudgy] Only 1D, 2D and 3D positions are supported for anisotropic smoothing tensors."
        )
    if not ratio_max >= 1.0:
        raise ValueError(f"[smudgy] ratio_max must be >= 1, got {ratio_max}.")
    if volume_preserving:
        if hsml is None:
            raise ValueError("[smudgy] volume_preserving=True requires hsml.")
        hsml = np.asarray(hsml, dtype=np.float64)
        if hsml.shape != (query_positions.shape[0],):
            raise ValueError(
                f"[smudgy] hsml must have shape ({query_positions.shape[0]},), got {hsml.shape}."
            )

    query_positions = _as_float32(query_positions)
    neighbor_positions = _as_float32(positions)[nn_inds]
    neighbor_weights = _as_float32(weights)[nn_inds]
    boxsize = _as_float32(boxsize) if boxsize is not None else None

    rel_coords = coordinate_difference_with_pbc(
        neighbor_positions, query_positions[:, np.newaxis, :], boxsize
    )

    # --- second-moment tensor about the query (Marinho eq. 37) ----------------
    rel64 = rel_coords.astype(np.float64)
    w = neighbor_weights.astype(np.float64)
    w /= w.sum(axis=1, keepdims=True)
    Sigma = np.einsum("nk,nki,nkj->nij", w, rel64, rel64, optimize=True)
    Sigma += (
        _EPS
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
    # zeta_max = 0 then, i.e. H = 0, matching the Taichi backend.
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
    if min_axis is not None:  # only ever enlarges axes
        axes = np.maximum(axes, min_axis)

    H = np.einsum("nij,nj,nkj->nik", eigvecs, axes, eigvecs, optimize=True)

    return (
        H.astype(np.float32),
        axes.astype(np.float32),  # eigenvalues of H, descending
        eigvecs.astype(np.float32),  # eigenvectors of H (columns)
    )
