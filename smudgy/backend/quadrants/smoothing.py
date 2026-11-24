"""Smoothing module for smudgy, providing functions to compute smoothing lengths and anisotropic smoothing tensors using Quadrants."""

import numpy as np
import quadrants as qd

from .._common import EPS, as_f32, as_i32, check_hmat_args, normalize_boxsize
from . import check_index_limits
from .linalg import sym_eig_descending


def compute_hsml(nn_dists: np.ndarray) -> np.ndarray:
    """Compute smoothing lengths from nearest-neighbor distances.

    Parameters
    ----------
    nn_dists : np.ndarray
        Array of nearest-neighbor distances.

    Returns
    -------
    np.ndarray
        Array of smoothing lengths.

    """
    return as_f32(nn_dists[:, -1])


@qd.func
def _coordinate_difference_with_pbc(
    d: qd.f32, box: qd.f32, periodic: qd.template()
) -> qd.f32:
    r = d
    if qd.static(periodic):
        half = 0.5 * box
        shifted = r + half
        wrapped = shifted - box * qd.floor(shifted / box)  # folds into [0, box)
        r = wrapped - half
    return r


@qd.func
def _neighbor_offset(
    query_positions: qd.template(),
    q,
    positions: qd.template(),
    j,
    boxsize: qd.template(),
    dim: qd.template(),
    periodic: qd.template(),
):
    """x_j - x_q, wrapped to the minimum image if `periodic`."""
    d = qd.Vector.zero(qd.f32, dim)
    for a in qd.static(range(dim)):
        d[a] = _coordinate_difference_with_pbc(
            positions[j, a] - query_positions[q, a], boxsize[a], periodic
        )
    return d


@qd.kernel
def _compute_hmat_kernel(
    query_positions: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, D)
    positions: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (Nc, D)
    weights: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (Nc,)
    nn_inds: qd.types.ndarray(dtype=qd.i32, ndim=2),  # (N, K), indexes positions
    boxsize: qd.types.ndarray(dtype=qd.f32, ndim=1),
    hsml: qd.types.ndarray(dtype=qd.f32, ndim=1),  # only read if volume_preserving
    periodic: qd.template(),
    dim: qd.template(),
    volume_preserving: qd.template(),
    eps: qd.f32,
    ratio_max: qd.f32,
    min_axis: qd.f32,  # 0.0 disables
    H_out: qd.types.ndarray(dtype=qd.f32, ndim=3),
    eigvals_out: qd.types.ndarray(dtype=qd.f32, ndim=2),
    eigvecs_out: qd.types.ndarray(dtype=qd.f32, ndim=3),
):
    N, K = nn_inds.shape
    for q in range(N):
        # --- second-moment tensor about the query (Marinho eq. 37) ------------
        Sigma = qd.Matrix.zero(qd.f32, dim, dim)
        wsum = 0.0
        for k in range(K):
            j = nn_inds[q, k]
            w = weights[j]
            d = _neighbor_offset(
                query_positions, q, positions, j, boxsize, dim, periodic
            )
            Sigma += w * d.outer_product(d)
            wsum += w
        Sigma /= wsum
        trace = qd.max(Sigma.trace(), 1e-30)
        Sigma += eps * trace * qd.Matrix.identity(qd.f32, dim)

        # Descending (index 0 = largest principal component, which the
        # axis-ratio clamp below relies on), sign convention as in numpy.
        eigvals, eigvecs = sym_eig_descending(Sigma, dim)

        # --- shape: principal components, with an axis-ratio clamp ---------------
        sigma = qd.Vector.zero(qd.f32, dim)
        for a in qd.static(range(dim)):
            sigma[a] = qd.sqrt(qd.max(eigvals[a], 0.0))
        for a in qd.static(range(dim)):
            sigma[a] = qd.max(sigma[a], sigma[0] / ratio_max)

        # --- scale: enclosing-hull condition, Marinho (2021) eq. 39 -------------
        # zeta_max = largest Mahalanobis distance among the K neighbours, using
        # the clamped metric so the hull property holds for the returned tensor.
        maha2_max = 0.0
        for k in range(K):
            d = _neighbor_offset(
                query_positions, q, positions, nn_inds[q, k], boxsize, dim, periodic
            )
            maha2 = 0.0
            for j in qd.static(range(dim)):
                proj = 0.0
                for a in qd.static(range(dim)):
                    proj += d[a] * eigvecs[a, j]
                maha2 += proj * proj / qd.max(sigma[j] * sigma[j], 1e-37)
            maha2_max = qd.max(maha2_max, maha2)
        zeta_max = qd.sqrt(maha2_max)

        axes = qd.Vector.zero(qd.f32, dim)
        for a in qd.static(range(dim)):
            axes[a] = zeta_max * sigma[a]  # eq. 44: h_1, h_2, h_3

        if qd.static(volume_preserving):  # det H = hsml^dim
            det = 1.0
            for a in qd.static(range(dim)):
                det *= axes[a]
            scale = hsml[q] / qd.pow(qd.max(det, 1e-30), 1.0 / dim)
            for a in qd.static(range(dim)):
                axes[a] *= scale

        if min_axis > 0.0:  # only ever enlarges axes
            for a in qd.static(range(dim)):
                axes[a] = qd.max(axes[a], min_axis)

        for a in qd.static(range(dim)):
            eigvals_out[q, a] = axes[a]

        Lam = qd.Matrix.zero(qd.f32, dim, dim)
        for a in qd.static(range(dim)):
            Lam[a, a] = axes[a]
        H = eigvecs @ Lam @ eigvecs.transpose()

        for a in qd.static(range(dim)):
            for b in qd.static(range(dim)):
                H_out[q, a, b] = H[a, b]
                eigvecs_out[q, a, b] = eigvecs[a, b]


def compute_hmat(
    query_positions: np.ndarray,
    positions: np.ndarray,
    weights: np.ndarray,
    nn_inds: np.ndarray,
    boxsize: np.ndarray | None = None,
    ratio_max: float = 100.0,
    min_axis: float | None = None,
    volume_preserving: bool = False,
    hsml: np.ndarray | None = None,
):
    """Compute covariant smoothing tensors (Marinho 2021).

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
    Nq = nn_inds.shape[0]
    min_axis = check_hmat_args(dim, ratio_max, min_axis, volume_preserving, hsml, Nq)

    query_positions = as_f32(query_positions)
    positions = as_f32(positions)
    weights = as_f32(weights)
    nn_inds = as_i32(nn_inds)
    check_index_limits(nn_inds=nn_inds)
    box, periodic = normalize_boxsize(boxsize, dim)
    hsml_arr = as_f32(hsml) if volume_preserving else np.zeros(Nq, dtype=np.float32)

    H = np.zeros((Nq, dim, dim), dtype=np.float32)
    eigvals = np.zeros((Nq, dim), dtype=np.float32)
    eigvecs = np.zeros((Nq, dim, dim), dtype=np.float32)

    _compute_hmat_kernel(
        query_positions,
        positions,
        weights,
        nn_inds,
        box,
        hsml_arr,
        periodic,
        dim,
        bool(volume_preserving),
        float(EPS),
        float(ratio_max),
        min_axis,
        H,
        eigvals,
        eigvecs,
    )
    return H, eigvals, eigvecs
