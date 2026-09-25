"""Smoothing module for smudgy, providing functions to compute smoothing lengths and anisotropic smoothing tensors using Taichi."""

import numpy as np
import taichi as ti

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


@ti.func
def _coordinate_difference_with_pbc(
    d: ti.f32, box: ti.f32, periodic: ti.template()
) -> ti.f32:
    r = d
    if ti.static(periodic):
        half = 0.5 * box
        shifted = r + half
        wrapped = shifted - box * ti.floor(shifted / box)  # folds into [0, box)
        r = wrapped - half
    return r


@ti.func
def _sym_eig_jacobi(A, dim: ti.template(), sweeps: ti.template()):
    """Eigendecomposition of a small symmetric matrix by cyclic Jacobi rotations.

    Used instead of ti.sym_eig, which returns non-orthogonal eigenvectors that
    can be off by tens of degrees for nearly degenerate matrices, i.e. for
    near-isotropic neighbourhoods. Jacobi's V is orthonormal by construction,
    one rotation diagonalises a 2x2 exactly, and a 3x3 converges to f32
    precision in a few sweeps. Returns (eigvals, V), unordered, with
    A = V diag(eigvals) V^T.
    """
    V = ti.Matrix.identity(ti.f32, dim)
    for _ in range(sweeps):
        for p in ti.static(range(dim)):
            for r in ti.static(range(p + 1, dim)):
                apr = A[p, r]
                if apr != 0.0:
                    # rotation angle that zeroes A[p, r] (Golub & Van Loan, sym.schur2)
                    tau = (A[r, r] - A[p, p]) / (2.0 * apr)
                    t = 1.0 / (ti.abs(tau) + ti.sqrt(1.0 + tau * tau))
                    if tau < 0.0:
                        t = -t
                    c = 1.0 / ti.sqrt(1.0 + t * t)
                    s = t * c
                    J = ti.Matrix.identity(ti.f32, dim)
                    J[p, p] = c
                    J[r, r] = c
                    J[p, r] = s
                    J[r, p] = -s
                    A = J.transpose() @ A @ J
                    A[p, r] = 0.0
                    A[r, p] = 0.0
                    V = V @ J
    eigvals = ti.Vector.zero(ti.f32, dim)
    for a in ti.static(range(dim)):
        eigvals[a] = A[a, a]
    return eigvals, V


@ti.kernel
def _compute_hmat_kernel(
    query_positions: ti.types.ndarray(dtype=ti.f32, ndim=2),
    neighbor_positions: ti.types.ndarray(
        dtype=ti.f32, ndim=3
    ),  # (N, K, D) -- pre-gathered
    neighbor_weights: ti.types.ndarray(
        dtype=ti.f32, ndim=2
    ),  # (N, K)    -- pre-gathered
    boxsize: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hsml: ti.types.ndarray(dtype=ti.f32, ndim=1),  # only read if volume_preserving
    periodic: ti.template(),
    dim: ti.template(),
    volume_preserving: ti.template(),
    eps: ti.f32,
    ratio_max: ti.f32,
    min_axis: ti.f32,  # 0.0 disables
    H_out: ti.types.ndarray(dtype=ti.f32, ndim=3),
    eigvals_out: ti.types.ndarray(dtype=ti.f32, ndim=2),
    eigvecs_out: ti.types.ndarray(dtype=ti.f32, ndim=3),
    rel_coords_out: ti.types.ndarray(dtype=ti.f32, ndim=3),
):
    K = neighbor_positions.shape[1]
    for q in range(query_positions.shape[0]):
        # --- second-moment tensor about the query (Marinho eq. 37) ------------
        Sigma = ti.Matrix.zero(ti.f32, dim, dim)
        wsum = 0.0
        for k in range(K):
            w = neighbor_weights[q, k]
            d = ti.Vector.zero(ti.f32, dim)
            for a in ti.static(range(dim)):
                diff = _coordinate_difference_with_pbc(
                    neighbor_positions[q, k, a] - query_positions[q, a],
                    boxsize[a],
                    periodic,
                )
                d[a] = diff
                rel_coords_out[q, k, a] = diff
            Sigma += w * d.outer_product(d)
            wsum += w
        Sigma /= wsum
        trace = ti.max(Sigma.trace(), 1e-30)
        Sigma += eps * trace * ti.Matrix.identity(ti.f32, dim)

        # Rescale to a unit-order magnitude band before the eigendecomposition
        # (Sigma's entries are tiny for unit-scale point clouds), then undo the
        # scale on the eigenvalues. H itself is invariant under an overall
        # rescaling of Sigma. 1x1 needs no rotations: zero sweeps.
        alpha = 10.0 / trace
        eigvals, eigvecs = _sym_eig_jacobi(
            alpha * Sigma, dim, ti.static(0 if dim == 1 else 6)
        )
        eigvals = eigvals / alpha

        # ==========================================================================================
        # Jacobi gives no ordering guarantee. Sort descending explicitly, matching the numpy
        # backend, so that index 0 is the largest principal component -- the
        # axis-ratio clamp below depends on it.
        for i in ti.static(range(dim)):
            for j in ti.static(range(i + 1, dim)):
                if eigvals[i] < eigvals[j]:
                    eigvals[i], eigvals[j] = eigvals[j], eigvals[i]
                    for a in ti.static(range(dim)):
                        eigvecs[a, i], eigvecs[a, j] = eigvecs[a, j], eigvecs[a, i]

        # Eigenvector orientation convention:
        # Numpy and Taichi have no convention about the sign of the eigenvectors,
        # impose a consistent sign convention for comparison with the numpy backend
        # (largest-magnitude component of each eigenvector is positive). H itself is
        # invariant under column sign flips; this only affects eigvecs_out.
        for col in ti.static(range(dim)):
            max_abs = 0.0
            sign = 1.0
            for row in ti.static(range(dim)):
                v = eigvecs[row, col]
                if ti.abs(v) > max_abs:
                    max_abs = ti.abs(v)
                    sign = 1.0 if v >= 0.0 else -1.0
            for row in ti.static(range(dim)):
                eigvecs[row, col] *= sign
        # ==========================================================================================

        # --- shape: principal components, with an axis-ratio clamp ---------------
        sigma = ti.Vector.zero(ti.f32, dim)
        for a in ti.static(range(dim)):
            sigma[a] = ti.sqrt(ti.max(eigvals[a], 0.0))
        for a in ti.static(range(dim)):
            sigma[a] = ti.max(sigma[a], sigma[0] / ratio_max)

        # --- scale: enclosing-hull condition, Marinho (2021) eq. 39 -------------
        # zeta_max = largest Mahalanobis distance among the K neighbours, using
        # the clamped metric so the hull property holds for the returned tensor.
        maha2_max = 0.0
        for k in range(K):
            maha2 = 0.0
            for j in ti.static(range(dim)):
                proj = 0.0
                for a in ti.static(range(dim)):
                    proj += rel_coords_out[q, k, a] * eigvecs[a, j]
                maha2 += proj * proj / ti.max(sigma[j] * sigma[j], 1e-37)
            maha2_max = ti.max(maha2_max, maha2)
        zeta_max = ti.sqrt(maha2_max)

        axes = ti.Vector.zero(ti.f32, dim)
        for a in ti.static(range(dim)):
            axes[a] = zeta_max * sigma[a]  # eq. 44: h_1, h_2, h_3

        if ti.static(volume_preserving):  # det H = hsml^dim
            det = 1.0
            for a in ti.static(range(dim)):
                det *= axes[a]
            scale = hsml[q] / ti.pow(ti.max(det, 1e-30), 1.0 / dim)
            for a in ti.static(range(dim)):
                axes[a] *= scale

        if min_axis > 0.0:  # only ever enlarges axes
            for a in ti.static(range(dim)):
                axes[a] = ti.max(axes[a], min_axis)

        for a in ti.static(range(dim)):
            eigvals_out[q, a] = axes[a]

        Lam = ti.Matrix.zero(ti.f32, dim, dim)
        for a in ti.static(range(dim)):
            Lam[a, a] = axes[a]
        H = eigvecs @ Lam @ eigvecs.transpose()

        for a in ti.static(range(dim)):
            for b in ti.static(range(dim)):
                H_out[q, a, b] = H[a, b]
                eigvecs_out[q, a, b] = eigvecs[a, b]


def compute_hmat(
    query_positions: np.ndarray,
    neighbor_positions: np.ndarray,
    neighbor_weights: np.ndarray,
    boxsize: np.ndarray | None = None,
    ratio_max: float = 10.0,
    min_axis: float | None = None,
    volume_preserving: bool = False,
    hsml: np.ndarray | None = None,
):
    """Compute anisotropic smoothing tensors from neighbor positions and weights.

    H = zeta_max * Sigma^(1/2), following Marinho (2021), eqs. 39-44: the support
    ellipsoid is the smallest one of the covariance's shape that still contains
    every neighbour. In the isotropic limit each axis reduces to the distance to
    the farthest neighbour, i.e. to the usual smoothing length, for any K.

    Parameters
    ----------
    query_positions : np.ndarray
        Array of query positions.
    neighbor_positions : np.ndarray
        Array of neighbor positions for each query position.
    neighbor_weights : np.ndarray
        Array of neighbor weights for each query position.
    boxsize : np.ndarray, optional
        Array of box sizes for periodic boundary conditions. If None, no periodicity is applied.
    ratio_max : float, optional
        Maximum axis ratio h_max / h_min (>= 1). Guards against degenerate
        (coplanar or collinear) neighbor sets.
    min_axis : float, optional
        Physical floor on the shortest axis, e.g. the softening length or cell
        size. Only ever enlarges axes, so the hull property is preserved. Also
        keeps H invertible if every neighbor coincides with the query (H = 0
        otherwise).
    volume_preserving : bool, optional
        If True, rescale the axes so that det H = hsml^dim, i.e. anisotropy in
        shape only. Requires `hsml`. Note that this can shrink the longest axis
        and so may break the enclosing-hull property.
    hsml : np.ndarray, optional
        Isotropic smoothing lengths, required if `volume_preserving` is True.

    Returns
    -------
    H : np.ndarray
        Array of smoothing tensors for each query position.
    eigvals : np.ndarray
        Eigenvalues of H, i.e. the principal smoothing lengths, descending.
    eigvecs : np.ndarray
        Eigenvectors of H (columns), matching the order of `eigvals`.
    rel_coords : np.ndarray
        Array of relative coordinates of neighbors with respect to query positions.

    """
    dim = query_positions.shape[-1]
    if dim not in (1, 2, 3):
        raise ValueError(
            "[smudgy] Only 1D, 2D and 3D positions are supported for anisotropic smoothing tensors."
        )
    if not ratio_max >= 1.0:
        raise ValueError(f"[smudgy] ratio_max must be >= 1, got {ratio_max}.")

    query_positions = _as_float32(query_positions)
    neighbor_positions = _as_float32(neighbor_positions)
    neighbor_weights = _as_float32(neighbor_weights)

    periodic = boxsize is not None
    box = _as_float32(boxsize) if periodic else np.zeros(dim, dtype=np.float32)

    Nq, K = neighbor_weights.shape
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
    rel_coords = np.zeros((Nq, K, dim), dtype=np.float32)

    _compute_hmat_kernel(
        query_positions,
        neighbor_positions,
        neighbor_weights,
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
        rel_coords,
    )
    return H, eigvals, eigvecs, rel_coords
