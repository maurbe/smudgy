import numpy as np
import quadrants as qd

from .kernels import (
    covariant_inputs_from_inverse,
    create_spherical_kernel,
    prepare_isotropic_inputs,
)
from .._common import (
    EPS,
    as_f32,
    as_i32,
    check_interpolation_mode,
    check_particle_structure,
    normalize_boxsize,
    support_scaled,
)
from . import check_index_limits
from .smoothing import _coordinate_difference_with_pbc


# =============================================================================
# Shared helpers
# =============================================================================
# The kernels below gather neighbor data through `nn_inds` inside the parallel
# loop and compute the (periodic) relative coordinates on the fly, instead of
# receiving pre-gathered (N, K, ...) arrays. Each relative coordinate is used
# exactly once, so precomputing it would do the same work plus a full write
# and re-read of an (N, K, D) array (and a host<->device copy on GPU).
@qd.func
def _relative_position(
    query_positions: qd.template(),
    n,
    positions: qd.template(),
    j,
    boxsize: qd.template(),
    dim: qd.template(),
    periodic: qd.template(),
):
    """r = x_n - x_j, wrapped to the minimum image if `periodic`."""
    r = qd.Vector.zero(qd.f32, dim)
    for a in qd.static(range(dim)):
        r[a] = _coordinate_difference_with_pbc(
            query_positions[n, a] - positions[j, a], boxsize[a], periodic
        )
    return r


@qd.func
def _kernel_inputs(
    r_vec,
    h: qd.template(),
    inv_det: qd.template(),
    i,
    dim: qd.template(),
    covariant: qd.template(),
    sigma_val: qd.f32,
    eps: qd.f32,
):
    """Kernel coordinates for r_vec, smoothed with h[i]: a scalar, or (covariant)
    the precomputed inverse tensor H^-1 with inv_det[i] = 1 / det(H)."""
    q = 0.0
    grad_q = qd.Vector.zero(qd.f32, dim)
    scale = 0.0
    if qd.static(covariant):
        H_inv = qd.Matrix(
            [[h[i, a, b] for b in qd.static(range(dim))] for a in qd.static(range(dim))]
        )
        q, grad_q, scale = covariant_inputs_from_inverse(
            r_vec, H_inv, inv_det[i], dim, sigma_val, eps
        )
    else:
        q, grad_q, scale = prepare_isotropic_inputs(r_vec, h[i], dim, sigma_val, eps)
    return q, grad_q, scale


@qd.kernel
def _invert_tensors(
    dim: qd.template(),
    h: qd.types.ndarray(dtype=qd.f32, ndim=3),  # (N, D, D)
    h_inv: qd.types.ndarray(dtype=qd.f32, ndim=3),  # (N, D, D)
    inv_det: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (N,)
):
    for n in range(h.shape[0]):
        H = qd.Matrix(
            [[h[n, a, b] for b in qd.static(range(dim))] for a in qd.static(range(dim))]
        )
        Hi = H.inverse()
        for a in qd.static(range(dim)):
            for b in qd.static(range(dim)):
                h_inv[n, a, b] = Hi[a, b]
        inv_det[n] = 1.0 / H.determinant()


def _smoothing_inputs(h, dim, covariant):
    """(h, inv_det) as the kernels read them: covariant tensors are inverted once
    per particle here, isotropic h passes through with a dummy inv_det."""
    if not covariant:
        return h, np.zeros(1, np.float32)
    h_inv = np.empty_like(h)
    inv_det = np.empty(h.shape[0], np.float32)
    _invert_tensors(dim, h, h_inv, inv_det)
    return h_inv, inv_det


def _common_inputs(query_positions, positions, weights, nn_inds, h, boxsize, dim):
    """Contiguous float32/int32 copies (no-ops if already so) + boxsize/periodic."""
    box, periodic = normalize_boxsize(boxsize, dim)
    nn_inds = as_i32(nn_inds)
    check_index_limits(nn_inds=nn_inds)
    return (
        as_f32(query_positions),
        as_f32(positions),
        as_f32(weights),
        nn_inds,
        as_f32(h),
        box,
        periodic,
    )


# =============================================================================
# Density
# =============================================================================
@qd.kernel
def _density_kernel(
    evaluate_fn: qd.template(),
    sigma_func: qd.template(),
    dim: qd.template(),
    covariant: qd.template(),
    periodic: qd.template(),
    eps: qd.f32,
    query_positions: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, D)
    positions: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (Nc, D)
    weights: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (Nc,)
    nn_inds: qd.types.ndarray(dtype=qd.i32, ndim=2),  # (N, K)
    boxsize: qd.types.ndarray(dtype=qd.f32, ndim=1),
    h: qd.types.ndarray(dtype=qd.f32),  # (N,) or (N, D, D) H^-1: the query's own
    inv_det: qd.types.ndarray(dtype=qd.f32, ndim=1),
    density_out: qd.types.ndarray(dtype=qd.f32, ndim=1),
):
    N, K = nn_inds.shape
    sigma_val = sigma_func(dim)
    for n in range(N):
        acc = 0.0
        for k in range(K):
            j = nn_inds[n, k]
            r_vec = _relative_position(
                query_positions, n, positions, j, boxsize, dim, periodic
            )
            q, _, scale = _kernel_inputs(r_vec, h, inv_det, n, dim, covariant, sigma_val, eps)
            acc += weights[j] * evaluate_fn(q, dim) * scale
        density_out[n] = acc


def density(
    *, kernel_name, dim, query_positions, positions, weights, nn_inds, h, boxsize, structure
):
    """Compute SPH densities (Quadrants backend).

    Parameters
    ----------
    kernel_name, dim, structure
        Kernel, spatial dimension and smoothing structure.
    query_positions : np.ndarray
        Query positions, shape (N, D).
    positions, weights : np.ndarray
        Particle positions (Nc, D) and weights (Nc,), indexed by `nn_inds`.
    nn_inds : np.ndarray
        Neighbour indices of each query, shape (N, K).
    h : np.ndarray
        Smoothing lengths (N,) or tensors (N, D, D) of the queries.
    boxsize : np.ndarray or None
        Box sizes for periodic boundary conditions.
    """
    check_particle_structure(structure, "density")
    kernel = create_spherical_kernel(kernel_name)
    h = support_scaled(h, kernel.support)
    query_positions, positions, weights, nn_inds, h, box, periodic = _common_inputs(
        query_positions, positions, weights, nn_inds, h, boxsize, dim
    )

    covariant = structure == "covariant"
    h, inv_det = _smoothing_inputs(h, dim, covariant)
    density_out = np.zeros(nn_inds.shape[0], dtype=np.float32)
    _density_kernel(
        kernel.evaluate,
        kernel.sigma,
        dim,
        covariant,
        periodic,
        float(EPS),
        query_positions,
        positions,
        weights,
        nn_inds,
        box,
        h,
        inv_det,
        density_out,
    )
    return density_out


# =============================================================================
# Interpolate
# =============================================================================
# Shared signature: `weights` are the per-particle m_j / rho_j, `h` the
# per-particle smoothing length (Nc,) or tensor (Nc, D, D) of each neighbor
# (gathered in-kernel via nn_inds), `fields` (Nc, F) or (Nc, F, D).
@qd.kernel
def _field_kernel(
    evaluate_fn: qd.template(),
    sigma_fn: qd.template(),
    dim: qd.template(),
    covariant: qd.template(),
    periodic: qd.template(),
    F: qd.template(),
    eps: qd.f32,
    query_positions: qd.types.ndarray(dtype=qd.f32, ndim=2),
    positions: qd.types.ndarray(dtype=qd.f32, ndim=2),
    weights: qd.types.ndarray(dtype=qd.f32, ndim=1),
    nn_inds: qd.types.ndarray(dtype=qd.i32, ndim=2),
    boxsize: qd.types.ndarray(dtype=qd.f32, ndim=1),
    h: qd.types.ndarray(dtype=qd.f32),
    inv_det: qd.types.ndarray(dtype=qd.f32, ndim=1),
    fields: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (Nc, F)
    out: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, F)
):
    N, K = nn_inds.shape
    sigma_val = sigma_fn(dim)
    for n in range(N):
        acc = qd.Vector.zero(qd.f32, F)
        for k in range(K):
            j = nn_inds[n, k]
            r_vec = _relative_position(
                query_positions, n, positions, j, boxsize, dim, periodic
            )
            q, _, scale = _kernel_inputs(r_vec, h, inv_det, j, dim, covariant, sigma_val, eps)
            w = evaluate_fn(q, dim) * scale * weights[j]
            for f in qd.static(range(F)):
                acc[f] += fields[j, f] * w
        for f in qd.static(range(F)):
            out[n, f] = acc[f]


@qd.kernel
def _gradient_kernel(
    gradient_fn: qd.template(),
    sigma_fn: qd.template(),
    dim: qd.template(),
    covariant: qd.template(),
    periodic: qd.template(),
    F: qd.template(),
    eps: qd.f32,
    query_positions: qd.types.ndarray(dtype=qd.f32, ndim=2),
    positions: qd.types.ndarray(dtype=qd.f32, ndim=2),
    weights: qd.types.ndarray(dtype=qd.f32, ndim=1),
    nn_inds: qd.types.ndarray(dtype=qd.i32, ndim=2),
    boxsize: qd.types.ndarray(dtype=qd.f32, ndim=1),
    h: qd.types.ndarray(dtype=qd.f32),
    inv_det: qd.types.ndarray(dtype=qd.f32, ndim=1),
    fields: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (Nc, F)
    out: qd.types.ndarray(dtype=qd.f32, ndim=3),  # (N, F, D)
):
    N, K = nn_inds.shape
    sigma_val = sigma_fn(dim)
    for n in range(N):
        acc = qd.Matrix.zero(qd.f32, F, dim)
        for k in range(K):
            j = nn_inds[n, k]
            r_vec = _relative_position(
                query_positions, n, positions, j, boxsize, dim, periodic
            )
            q, grad_q, scale = _kernel_inputs(
                r_vec, h, inv_det, j, dim, covariant, sigma_val, eps
            )
            grad_w = gradient_fn(q, grad_q, dim) * scale * weights[j]
            for f in qd.static(range(F)):
                for a in qd.static(range(dim)):
                    acc[f, a] += fields[j, f] * grad_w[a]
        for f in qd.static(range(F)):
            for a in qd.static(range(dim)):
                out[n, f, a] = acc[f, a]


@qd.kernel
def _divergence_kernel(
    gradient_fn: qd.template(),
    sigma_fn: qd.template(),
    dim: qd.template(),
    covariant: qd.template(),
    periodic: qd.template(),
    F: qd.template(),
    eps: qd.f32,
    query_positions: qd.types.ndarray(dtype=qd.f32, ndim=2),
    positions: qd.types.ndarray(dtype=qd.f32, ndim=2),
    weights: qd.types.ndarray(dtype=qd.f32, ndim=1),
    nn_inds: qd.types.ndarray(dtype=qd.i32, ndim=2),
    boxsize: qd.types.ndarray(dtype=qd.f32, ndim=1),
    h: qd.types.ndarray(dtype=qd.f32),
    inv_det: qd.types.ndarray(dtype=qd.f32, ndim=1),
    fields: qd.types.ndarray(dtype=qd.f32, ndim=3),  # (Nc, F, D)
    out: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, F)
):
    N, K = nn_inds.shape
    sigma_val = sigma_fn(dim)
    for n in range(N):
        acc = qd.Vector.zero(qd.f32, F)
        for k in range(K):
            j = nn_inds[n, k]
            r_vec = _relative_position(
                query_positions, n, positions, j, boxsize, dim, periodic
            )
            q, grad_q, scale = _kernel_inputs(
                r_vec, h, inv_det, j, dim, covariant, sigma_val, eps
            )
            grad_w = gradient_fn(q, grad_q, dim) * scale * weights[j]
            for f in qd.static(range(F)):
                s = 0.0
                for a in qd.static(range(dim)):
                    s += fields[j, f, a] * grad_w[a]
                acc[f] += s
        for f in qd.static(range(F)):
            out[n, f] = acc[f]


@qd.kernel
def _curl_kernel(
    gradient_fn: qd.template(),
    sigma_fn: qd.template(),
    dim: qd.template(),
    covariant: qd.template(),
    periodic: qd.template(),
    F: qd.template(),
    eps: qd.f32,
    query_positions: qd.types.ndarray(dtype=qd.f32, ndim=2),
    positions: qd.types.ndarray(dtype=qd.f32, ndim=2),
    weights: qd.types.ndarray(dtype=qd.f32, ndim=1),
    nn_inds: qd.types.ndarray(dtype=qd.i32, ndim=2),
    boxsize: qd.types.ndarray(dtype=qd.f32, ndim=1),
    h: qd.types.ndarray(dtype=qd.f32),
    inv_det: qd.types.ndarray(dtype=qd.f32, ndim=1),
    fields: qd.types.ndarray(dtype=qd.f32, ndim=3),  # (Nc, F, D)
    out: qd.types.ndarray(dtype=qd.f32, ndim=3),  # (N, F, 1 or D)
):
    N, K = nn_inds.shape
    sigma_val = sigma_fn(dim)
    C = qd.static(1 if dim == 2 else dim)
    for n in range(N):
        acc = qd.Matrix.zero(qd.f32, F, C)
        for k in range(K):
            j = nn_inds[n, k]
            r_vec = _relative_position(
                query_positions, n, positions, j, boxsize, dim, periodic
            )
            q, grad_q, scale = _kernel_inputs(
                r_vec, h, inv_det, j, dim, covariant, sigma_val, eps
            )
            grad_w = gradient_fn(q, grad_q, dim) * scale * weights[j]
            for f in qd.static(range(F)):
                if qd.static(dim == 2):
                    acc[f, 0] += fields[j, f, 0] * grad_w[1] - fields[j, f, 1] * grad_w[0]
                else:
                    fv = qd.Vector([fields[j, f, a] for a in qd.static(range(3))])
                    c = fv.cross(grad_w)
                    for a in qd.static(range(3)):
                        acc[f, a] += c[a]
        for f in qd.static(range(F)):
            for a in qd.static(range(C)):
                out[n, f, a] = acc[f, a]


def interpolate(
    *,
    kernel_name,
    dim,
    query_positions,
    positions,
    weights,
    nn_inds,
    fields,
    h,
    boxsize,
    mode,
    structure,
):
    """Interpolate fields at query positions (Quadrants backend).

    Parameters
    ----------
    kernel_name, dim, structure, mode
        Kernel, spatial dimension, smoothing structure and interpolation mode.
    query_positions : np.ndarray
        Query positions, shape (N, D).
    positions : np.ndarray
        Particle positions, shape (Nc, D), indexed by `nn_inds`.
    weights : np.ndarray
        Particle volumes m / rho, shape (Nc,).
    nn_inds : np.ndarray
        Neighbour indices of each query, shape (N, K).
    fields : np.ndarray
        Particle fields, shape (Nc, F), or (Nc, F, D) for divergence and curl.
    h : np.ndarray
        Smoothing lengths (Nc,) or tensors (Nc, D, D) of the particles.
    boxsize : np.ndarray or None
        Box sizes for periodic boundary conditions.
    """
    check_particle_structure(structure, "interpolation")
    check_interpolation_mode(mode, dim)
    kernel = create_spherical_kernel(kernel_name)
    h = support_scaled(h, kernel.support)
    query_positions, positions, weights, nn_inds, h, box, periodic = _common_inputs(
        query_positions, positions, weights, nn_inds, h, boxsize, dim
    )
    fields = as_f32(fields)
    check_index_limits(fields=fields, out=(*nn_inds.shape[:1], *fields.shape[1:]))

    N = nn_inds.shape[0]
    F = fields.shape[1]
    covariant = structure == "covariant"
    h, inv_det = _smoothing_inputs(h, dim, covariant)

    if mode == "field":
        fn, kernel_fn, out = _field_kernel, kernel.evaluate, np.zeros((N, F), np.float32)
    elif mode == "gradient":
        fn, kernel_fn = _gradient_kernel, kernel.gradient
        out = np.zeros((N, F, dim), dtype=np.float32)
    elif mode == "divergence":
        fn, kernel_fn, out = _divergence_kernel, kernel.gradient, np.zeros((N, F), np.float32)
    else:  # "curl" (validated above)
        C = 1 if dim == 2 else dim
        fn, kernel_fn = _curl_kernel, kernel.gradient
        out = np.zeros((N, F, C), dtype=np.float32)

    fn(
        kernel_fn,
        kernel.sigma,
        dim,
        covariant,
        periodic,
        F,
        EPS,
        query_positions,
        positions,
        weights,
        nn_inds,
        box,
        h,
        inv_det,
        fields,
        out,
    )
    if mode == "curl" and dim == 2:
        return out[..., 0]
    return out
