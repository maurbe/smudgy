"""Smoothing module for smudgy, providing functions to compute smoothing lengths and anisotropic smoothing tensors using Taichi."""

import numpy as np
import quadrants as qd

_EPS = 1e-7


def _as_float32(array):
    """Return a C-contiguous float32 array. No copy if `array` is already float32 and C-contiguous; copies otherwise."""
    return np.ascontiguousarray(array, dtype=np.float32)


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
    return np.ascontiguousarray(nn_dists[:, -1], dtype=np.float32)


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


@qd.func
def _sym_eig_jacobi(A, dim: qd.template(), sweeps: qd.template()):
    """Eigendecomposition of a small symmetric matrix by cyclic Jacobi rotations.

    Used instead of qd.sym_eig, which returns non-orthogonal eigenvectors that
    can be off by tens of degrees for nearly degenerate matrices, i.e. for
    near-isotropic neighbourhoods. Jacobi's V is orthonormal by construction,
    one rotation diagonalises a 2x2 exactly, and a 3x3 converges to f32
    precision in a few sweeps. Returns (eigvals, V), unordered, with
    A = V diag(eigvals) V^T.
    """
    V = qd.Matrix.identity(qd.f32, dim)
    for _ in range(sweeps):
        for p in qd.static(range(dim)):
            for r in qd.static(range(p + 1, dim)):
                apr = A[p, r]
                if apr != 0.0:
                    # rotation angle that zeroes A[p, r] (Golub & Van Loan, sym.schur2)
                    tau = (A[r, r] - A[p, p]) / (2.0 * apr)
                    t = 1.0 / (qd.abs(tau) + qd.sqrt(1.0 + tau * tau))
                    if tau < 0.0:
                        t = -t
                    c = 1.0 / qd.sqrt(1.0 + t * t)
                    s = t * c
                    J = qd.Matrix.identity(qd.f32, dim)
                    J[p, p] = c
                    J[r, r] = c
                    J[p, r] = s
                    J[r, p] = -s
                    A = J.transpose() @ A @ J
                    A[p, r] = 0.0
                    A[r, p] = 0.0
                    V = V @ J
    eigvals = qd.Vector.zero(qd.f32, dim)
    for a in qd.static(range(dim)):
        eigvals[a] = A[a, a]
    return eigvals, V


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
            d = _neighbor_offset(query_positions, q, positions, j, boxsize, dim, periodic)
            Sigma += w * d.outer_product(d)
            wsum += w
        Sigma /= wsum
        trace = qd.max(Sigma.trace(), 1e-30)
        Sigma += eps * trace * qd.Matrix.identity(qd.f32, dim)

        # Rescale to a unit-order magnitude band before the eigendecomposition
        # (Sigma's entries are tiny for unit-scale point clouds), then undo the
        # scale on the eigenvalues. H itself is invariant under an overall
        # rescaling of Sigma. 1x1 needs no rotations: zero sweeps.
        alpha = 10.0 / trace
        eigvals, eigvecs = _sym_eig_jacobi(
            alpha * Sigma, dim, qd.static(0 if dim == 1 else 6)
        )
        eigvals = eigvals / alpha

        # ==========================================================================================
        # Jacobi gives no ordering guarantee. Sort descending explicitly, matching the numpy
        # backend, so that index 0 is the largest principal component -- the
        # axis-ratio clamp below depends on it.
        for i in qd.static(range(dim)):
            for j in qd.static(range(i + 1, dim)):
                if eigvals[i] < eigvals[j]:
                    eigvals[i], eigvals[j] = eigvals[j], eigvals[i]
                    for a in qd.static(range(dim)):
                        eigvecs[a, i], eigvecs[a, j] = eigvecs[a, j], eigvecs[a, i]

        # Eigenvector orientation convention:
        # Numpy and Taichi have no convention about the sign of the eigenvectors,
        # impose a consistent sign convention for comparison with the numpy backend
        # (largest-magnitude component of each eigenvector is positive). H itself is
        # invariant under column sign flips; this only affects eigvecs_out.
        for col in qd.static(range(dim)):
            max_abs = 0.0
            sign = 1.0
            for row in qd.static(range(dim)):
                v = eigvecs[row, col]
                if qd.abs(v) > max_abs:
                    max_abs = qd.abs(v)
                    sign = 1.0 if v >= 0.0 else -1.0
            for row in qd.static(range(dim)):
                eigvecs[row, col] *= sign
        # ==========================================================================================

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
    if dim not in (1, 2, 3):
        raise ValueError(
            "[smudgy] Only 1D, 2D and 3D positions are supported for anisotropic smoothing tensors."
        )
    if not ratio_max >= 1.0:
        raise ValueError(f"[smudgy] ratio_max must be >= 1, got {ratio_max}.")

    query_positions = _as_float32(query_positions)
    positions = _as_float32(positions)
    weights = _as_float32(weights)
    nn_inds = np.ascontiguousarray(nn_inds, dtype=np.int32)

    periodic = boxsize is not None
    box = (
        np.ascontiguousarray(np.broadcast_to(boxsize, (dim,)), dtype=np.float32)
        if periodic
        else np.zeros(dim, dtype=np.float32)
    )

    Nq = nn_inds.shape[0]
    if volume_preserving:
        if hsml is None:
            raise ValueError("[smudgy] volume_preserving=True requires hsml.")
        hsml_arr = _as_float32(hsml)
        if hsml_arr.shape != (Nq,):
            raise ValueError(
                f"[smudgy] hsml must have shape ({Nq},), got {hsml_arr.shape}."
            )
    else:
        hsml_arr = np.zeros(Nq, dtype=np.float32)

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
        float(_EPS),
        float(ratio_max),
        float(min_axis) if min_axis is not None else 0.0,
        H,
        eigvals,
        eigvecs,
    )
    return H, eigvals, eigvecs
