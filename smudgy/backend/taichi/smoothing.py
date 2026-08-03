# backend/taichi/smoothing.py
import numpy as np
import taichi as ti

_EPS = 1e-7

def _as_float32(array):
    """Return a C-contiguous float32 array. No copy if `array` is
    already float32 and C-contiguous; copies otherwise.
    """
    return np.ascontiguousarray(array, dtype=np.float32)

def compute_hsml(nn_dists: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(nn_dists[:, -1], dtype=np.float32)


@ti.func
def _coordinate_difference_with_pbc(d: ti.f32, box: ti.f32, periodic: ti.template()) -> ti.f32:
    r = d
    if ti.static(periodic):
        half = 0.5 * box
        shifted = r + half
        wrapped = shifted - box * ti.floor(shifted / box)  # folds into [0, box)
        r = wrapped - half
    return r


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
    periodic: ti.template(),
    dim: ti.template(),
    eps: ti.f32,
    H_out: ti.types.ndarray(dtype=ti.f32, ndim=3),
    eigvals_out: ti.types.ndarray(dtype=ti.f32, ndim=2),
    eigvecs_out: ti.types.ndarray(dtype=ti.f32, ndim=3),
    rel_coords_out: ti.types.ndarray(dtype=ti.f32, ndim=3),
):
    K = neighbor_positions.shape[1]
    for q in range(query_positions.shape[0]):
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
        Sigma += eps * Sigma.trace() * ti.Matrix.identity(ti.f32, dim)

        eigvals, eigvecs = ti.sym_eig(Sigma, ti.f32)

        # ==========================================================================================
        # Fix for annyoing taichi bug in 2D: 
        # eigvals/vecs are returned in descending order, 3D is fine and follows numpy backend
        if ti.static(dim == 2):
            if eigvals[0] > eigvals[1]:
                eigvals[0], eigvals[1] = eigvals[1], eigvals[0]
                for a in ti.static(range(2)):
                    eigvecs[a, 0], eigvecs[a, 1] = eigvecs[a, 1], eigvecs[a, 0]

            # Eigenvector orientation convention:
            # Numpy and Taichi have no convention about the sign of the eigenvectors,
            # impose a consistent sign convention for comparison with Taichi backend 
            # (largest-magnitude component of each eigenvector is positive).
            for col in ti.static(range(2)):
                max_abs = 0.0
                sign = 1.0
                for row in ti.static(range(2)):
                    v = eigvecs[row, col]
                    if ti.abs(v) > max_abs:
                        max_abs = ti.abs(v)
                        sign = 1.0 if v >= 0.0 else -1.0
                for row in ti.static(range(2)):
                    eigvecs[row, col] *= sign
        # ==========================================================================================

        for a in ti.static(range(dim)):
            eigvals_out[q, a] = ti.sqrt(ti.max(eigvals[a], 0.0))

        Lam = ti.Matrix.zero(ti.f32, dim, dim)
        for a in ti.static(range(dim)):
            Lam[a, a] = eigvals_out[q, a]
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
):
    dim = query_positions.shape[-1]
    if dim not in (2, 3):
        raise ValueError(
            "[smudgy] Only 2D and 3D positions are supported for anisotropic smoothing tensors."
        )

    query_positions = _as_float32(query_positions)
    neighbor_positions = _as_float32(neighbor_positions)
    neighbor_weights = _as_float32(neighbor_weights)

    periodic = boxsize is not None
    box = (
        _as_float32(boxsize)
        if periodic
        else np.zeros(dim, dtype=np.float32)
    )

    Nq, K = neighbor_weights.shape
    H = np.zeros((Nq, dim, dim), dtype=np.float32)
    eigvals = np.zeros((Nq, dim), dtype=np.float32)
    eigvecs = np.zeros((Nq, dim, dim), dtype=np.float32)
    rel_coords = np.zeros((Nq, K, dim), dtype=np.float32)

    _compute_hmat_kernel(
        query_positions,
        neighbor_positions,
        neighbor_weights,
        box,
        periodic,
        dim,
        float(_EPS),
        H,
        eigvals,
        eigvecs,
        rel_coords,
    )
    return H, eigvals, eigvecs, rel_coords
