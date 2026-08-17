import numpy as np
import taichi as ti

from .kernels import (
    create_spherical_kernel,
    prepare_covariant_inputs,
    prepare_isotropic_inputs,
)

_EPS = 1e-7


# =============================================================================
# Density
# =============================================================================
@ti.kernel
def _density_scalar_kernel(
    evaluate_fn: ti.template(),
    prepare_fn: ti.template(),
    dim: ti.template(),
    sigma_func: ti.template(),
    eps: ti.f32,
    neighbor_weights: ti.types.ndarray(dtype=ti.f32, ndim=2),
    r_ij: ti.types.ndarray(dtype=ti.f32, ndim=3),
    h: ti.types.ndarray(dtype=ti.f32, ndim=1),
    density_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    N, K = neighbor_weights.shape
    sigma_val = sigma_func(dim)
    for n in range(N):
        acc = 0.0
        for k in range(K):
            r_vec = ti.Vector([r_ij[n, k, a] for a in ti.static(range(dim))])
            q, _, scale = prepare_fn(r_vec, h[n], dim, sigma_val, eps)
            acc += neighbor_weights[n, k] * evaluate_fn(q, dim) * scale
        density_out[n] = acc


@ti.kernel
def _density_covariant_kernel(
    evaluate_fn: ti.template(),
    prepare_fn: ti.template(),
    dim: ti.template(),
    sigma_func: ti.template(),
    eps: ti.f32,
    neighbor_weights: ti.types.ndarray(dtype=ti.f32, ndim=2),
    r_ij: ti.types.ndarray(dtype=ti.f32, ndim=3),
    h: ti.types.ndarray(dtype=ti.f32, ndim=3),
    density_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    N, K = neighbor_weights.shape
    sigma_val = sigma_func(dim)
    for n in range(N):
        H = ti.Matrix(
            [[h[n, a, b] for b in ti.static(range(dim))] for a in ti.static(range(dim))]
        )
        acc = 0.0
        for k in range(K):
            r_vec = ti.Vector([r_ij[n, k, a] for a in ti.static(range(dim))])
            q, _, scale = prepare_fn(r_vec, H, dim, sigma_val, eps)
            acc += neighbor_weights[n, k] * evaluate_fn(q, dim) * scale
        density_out[n] = acc


def density(*, kernel_name, dim, neighbor_weights, r_ij, h, structure):
    """Compute SPH density using the Taichi backend."""
    if structure == "separable":
        raise ValueError("structure='separable' is not supported for particle density.")

    kernel = create_spherical_kernel(kernel_name)

    neighbor_weights = np.ascontiguousarray(neighbor_weights, dtype=np.float32)
    r_ij = np.ascontiguousarray(r_ij, dtype=np.float32)
    h = np.ascontiguousarray(h, dtype=np.float32)

    N = h.shape[0]
    density_out = np.zeros(N, dtype=np.float32)
    eps = np.finfo(np.float32).eps

    sigma_func = kernel.sigma

    if structure == "covariant":
        _density_covariant_kernel(
            kernel.evaluate,
            prepare_covariant_inputs,
            dim,
            sigma_func,
            float(_EPS),
            neighbor_weights,
            r_ij,
            h,
            density_out,
        )
    else:
        _density_scalar_kernel(
            kernel.evaluate,
            prepare_isotropic_inputs,
            dim,
            sigma_func,
            float(_EPS),
            neighbor_weights,
            r_ij,
            h,
            density_out,
        )
    return density_out


# =============================================================================
# Interpolate
# =============================================================================
@ti.kernel
def _field_isotropic(
    evaluate_fn: ti.template(),
    prepare_fn: ti.template(),
    sigma_fn: ti.template(),
    dim: ti.template(),
    F: ti.template(),
    fields: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (N, K, F)
    weights: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, K)
    r_ij: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (N, K, D)
    h: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, K)
    eps: ti.f32,
    out: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
):
    N, K = fields.shape[0], fields.shape[1]
    sigma_val = sigma_fn(dim)
    for n in range(N):
        for f in ti.static(range(F)):
            out[n, f] = 0.0
        for k in range(K):
            r_vec = ti.Vector([r_ij[n, k, a] for a in ti.static(range(dim))])
            q, _, scale = prepare_fn(r_vec, h[n, k], dim, sigma_val, eps)
            w = evaluate_fn(q, dim) * scale * weights[n, k]
            for f in ti.static(range(F)):
                out[n, f] += fields[n, k, f] * w


@ti.kernel
def _field_covariant(
    evaluate_fn: ti.template(),
    prepare_fn: ti.template(),
    sigma_fn: ti.template(),
    dim: ti.template(),
    F: ti.template(),
    fields: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (N, K, F)
    weights: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, K)
    r_ij: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (N, K, D)
    h: ti.types.ndarray(dtype=ti.f32, ndim=4),  # (N, K, D, D)
    eps: ti.f32,
    out: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
):
    N, K = fields.shape[0], fields.shape[1]
    sigma_val = sigma_fn(dim)
    for n in range(N):
        for f in ti.static(range(F)):
            out[n, f] = 0.0
        for k in range(K):
            r_vec = ti.Vector([r_ij[n, k, a] for a in ti.static(range(dim))])
            H = ti.Matrix(
                [
                    [h[n, k, a, b] for b in ti.static(range(dim))]
                    for a in ti.static(range(dim))
                ]
            )
            q, _, scale = prepare_fn(r_vec, H, dim, sigma_val, eps)
            w = evaluate_fn(q, dim) * scale * weights[n, k]
            for f in ti.static(range(F)):
                out[n, f] += fields[n, k, f] * w


@ti.kernel
def _gradient_isotropic(
    gradient_fn: ti.template(),
    prepare_fn: ti.template(),
    sigma_fn: ti.template(),
    dim: ti.template(),
    F: ti.template(),
    fields: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (N, K, F)
    weights: ti.types.ndarray(dtype=ti.f32, ndim=2),
    r_ij: ti.types.ndarray(dtype=ti.f32, ndim=3),
    h: ti.types.ndarray(dtype=ti.f32, ndim=2),
    eps: ti.f32,
    out: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (N, F, D)
):
    N, K = fields.shape[0], fields.shape[1]
    sigma_val = sigma_fn(dim)
    for n in range(N):
        for f in ti.static(range(F)):
            for a in ti.static(range(dim)):
                out[n, f, a] = 0.0
        for k in range(K):
            r_vec = ti.Vector([r_ij[n, k, a] for a in ti.static(range(dim))])
            q, grad_q, scale = prepare_fn(r_vec, h[n, k], dim, sigma_val, eps)
            grad_w = gradient_fn(q, grad_q, dim) * scale * weights[n, k]
            for f in ti.static(range(F)):
                for a in ti.static(range(dim)):
                    out[n, f, a] += fields[n, k, f] * grad_w[a]


@ti.kernel
def _gradient_covariant(
    gradient_fn: ti.template(),
    prepare_fn: ti.template(),
    sigma_fn: ti.template(),
    dim: ti.template(),
    F: ti.template(),
    fields: ti.types.ndarray(dtype=ti.f32, ndim=3),
    weights: ti.types.ndarray(dtype=ti.f32, ndim=2),
    r_ij: ti.types.ndarray(dtype=ti.f32, ndim=3),
    h: ti.types.ndarray(dtype=ti.f32, ndim=4),  # (N, K, D, D)
    eps: ti.f32,
    out: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (N, F, D)
):
    N, K = fields.shape[0], fields.shape[1]
    sigma_val = sigma_fn(dim)

    for n in range(N):
        for f in ti.static(range(F)):
            for a in ti.static(range(dim)):
                out[n, f, a] = 0.0
        for k in range(K):
            r_vec = ti.Vector([r_ij[n, k, a] for a in ti.static(range(dim))])
            H = ti.Matrix(
                [
                    [h[n, k, a, b] for b in ti.static(range(dim))]
                    for a in ti.static(range(dim))
                ]
            )
            q, grad_q, scale = prepare_fn(r_vec, H, dim, sigma_val, eps)
            grad_w = gradient_fn(q, grad_q, dim) * scale * weights[n, k]
            for f in ti.static(range(F)):
                for a in ti.static(range(dim)):
                    out[n, f, a] += fields[n, k, f] * grad_w[a]


@ti.kernel
def _divergence_isotropic(
    gradient_fn: ti.template(),
    prepare_fn: ti.template(),
    sigma_fn: ti.template(),
    dim: ti.template(),
    F: ti.template(),
    fields: ti.types.ndarray(dtype=ti.f32, ndim=4),  # (N, K, F, D)
    weights: ti.types.ndarray(dtype=ti.f32, ndim=2),
    r_ij: ti.types.ndarray(dtype=ti.f32, ndim=3),
    h: ti.types.ndarray(dtype=ti.f32, ndim=2),
    eps: ti.f32,
    out: ti.types.ndarray(dtype=ti.f32, ndim=2),
):
    N, K = fields.shape[0], fields.shape[1]
    sigma_val = sigma_fn(dim)
    for n in range(N):
        for f in ti.static(range(F)):
            out[n, f] = 0.0
        for k in range(K):
            r_vec = ti.Vector([r_ij[n, k, a] for a in ti.static(range(dim))])
            q, grad_q, scale = prepare_fn(r_vec, h[n, k], dim, sigma_val, eps)
            grad_w = gradient_fn(q, grad_q, dim) * scale * weights[n, k]
            for f in ti.static(range(F)):
                s = 0.0
                for a in ti.static(range(dim)):
                    s += fields[n, k, f, a] * grad_w[a]
                out[n, f] += s


@ti.kernel
def _divergence_covariant(
    gradient_fn: ti.template(),
    prepare_fn: ti.template(),
    sigma_fn: ti.template(),
    dim: ti.template(),
    F: ti.template(),
    fields: ti.types.ndarray(dtype=ti.f32, ndim=4),
    weights: ti.types.ndarray(dtype=ti.f32, ndim=2),
    r_ij: ti.types.ndarray(dtype=ti.f32, ndim=3),
    h: ti.types.ndarray(dtype=ti.f32, ndim=4),
    eps: ti.f32,
    out: ti.types.ndarray(dtype=ti.f32, ndim=2),
):
    N, K = fields.shape[0], fields.shape[1]
    sigma_val = sigma_fn(dim)
    for n in range(N):
        for f in ti.static(range(F)):
            out[n, f] = 0.0
        for k in range(K):
            r_vec = ti.Vector([r_ij[n, k, a] for a in ti.static(range(dim))])
            H = ti.Matrix(
                [
                    [h[n, k, a, b] for b in ti.static(range(dim))]
                    for a in ti.static(range(dim))
                ]
            )
            q, grad_q, scale = prepare_fn(r_vec, H, dim, sigma_val, eps)
            grad_w = gradient_fn(q, grad_q, dim) * scale * weights[n, k]
            for f in ti.static(range(F)):
                s = 0.0
                for a in ti.static(range(dim)):
                    s += fields[n, k, f, a] * grad_w[a]
                out[n, f] += s


@ti.kernel
def _curl_isotropic(
    gradient_fn: ti.template(),
    prepare_fn: ti.template(),
    sigma_fn: ti.template(),
    dim: ti.template(),
    F: ti.template(),
    fields: ti.types.ndarray(dtype=ti.f32, ndim=4),  # (N, K, F, D)
    weights: ti.types.ndarray(dtype=ti.f32, ndim=2),
    r_ij: ti.types.ndarray(dtype=ti.f32, ndim=3),
    h: ti.types.ndarray(dtype=ti.f32, ndim=2),
    eps: ti.f32,
    out: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (N, F, 1 or D)
):
    N, K = fields.shape[0], fields.shape[1]
    sigma_val = sigma_fn(dim)
    C = ti.static(1 if dim == 2 else dim)

    for n in range(N):
        for f in ti.static(range(F)):
            for a in ti.static(range(C)):
                out[n, f, a] = 0.0
        for k in range(K):
            r_vec = ti.Vector([r_ij[n, k, a] for a in ti.static(range(dim))])
            q, grad_q, scale = prepare_fn(r_vec, h[n, k], dim, sigma_val, eps)
            grad_w = gradient_fn(q, grad_q, dim) * scale * weights[n, k]
            for f in ti.static(range(F)):
                if ti.static(dim == 2):
                    out[n, f, 0] += (
                        fields[n, k, f, 0] * grad_w[1] - fields[n, k, f, 1] * grad_w[0]
                    )
                else:
                    fv = ti.Vector([fields[n, k, f, a] for a in ti.static(range(3))])
                    c = fv.cross(grad_w)
                    for a in ti.static(range(3)):
                        out[n, f, a] += c[a]


@ti.kernel
def _curl_covariant(
    gradient_fn: ti.template(),
    prepare_fn: ti.template(),
    sigma_fn: ti.template(),
    dim: ti.template(),
    F: ti.template(),
    fields: ti.types.ndarray(dtype=ti.f32, ndim=4),
    weights: ti.types.ndarray(dtype=ti.f32, ndim=2),
    r_ij: ti.types.ndarray(dtype=ti.f32, ndim=3),
    h: ti.types.ndarray(dtype=ti.f32, ndim=4),
    eps: ti.f32,
    out: ti.types.ndarray(dtype=ti.f32, ndim=3),
):
    N, K = fields.shape[0], fields.shape[1]
    sigma_val = sigma_fn(dim)
    C = ti.static(1 if dim == 2 else dim)

    for n in range(N):
        for f in ti.static(range(F)):
            for a in ti.static(range(C)):
                out[n, f, a] = 0.0
        for k in range(K):
            r_vec = ti.Vector([r_ij[n, k, a] for a in ti.static(range(dim))])
            H = ti.Matrix(
                [
                    [h[n, k, a, b] for b in ti.static(range(dim))]
                    for a in ti.static(range(dim))
                ]
            )
            q, grad_q, scale = prepare_fn(r_vec, H, dim, sigma_val, eps)
            grad_w = gradient_fn(q, grad_q, dim) * scale * weights[n, k]
            for f in ti.static(range(F)):
                if ti.static(dim == 2):
                    out[n, f, 0] += (
                        fields[n, k, f, 0] * grad_w[1] - fields[n, k, f, 1] * grad_w[0]
                    )
                else:
                    fv = ti.Vector([fields[n, k, f, a] for a in ti.static(range(3))])
                    c = fv.cross(grad_w)
                    for a in ti.static(range(3)):
                        out[n, f, a] += c[a]


def interpolate(
    *,
    kernel_name,
    dim,
    fields,
    weights,
    r_ij,
    h,
    mode,
    structure,
):
    """Interpolate fields at query positions using the Taichi backend."""
    if structure == "separable":
        raise ValueError("structure='separable' is not supported for particle density.")

    if dim == 1 and mode == "curl":
        raise NotImplementedError

    kernel = create_spherical_kernel(kernel_name)

    fields = np.ascontiguousarray(fields, dtype=np.float32)
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    r_ij = np.ascontiguousarray(r_ij, dtype=np.float32)
    h = np.ascontiguousarray(h, dtype=np.float32)

    N = fields.shape[0]
    F = fields.shape[2]
    covariant = structure == "covariant"
    prepare_fn = prepare_covariant_inputs if covariant else prepare_isotropic_inputs

    if mode == "field":
        out = np.zeros((N, F), dtype=np.float32)
        fn = _field_covariant if covariant else _field_isotropic
        fn(
            kernel.evaluate,
            prepare_fn,
            kernel.sigma,
            dim,
            F,
            fields,
            weights,
            r_ij,
            h,
            _EPS,
            out,
        )
        return out

    if mode == "gradient":
        out = np.zeros((N, F, dim), dtype=np.float32)
        fn = _gradient_covariant if covariant else _gradient_isotropic
        fn(
            kernel.gradient,
            prepare_fn,
            kernel.sigma,
            dim,
            F,
            fields,
            weights,
            r_ij,
            h,
            _EPS,
            out,
        )
        return out

    if mode == "divergence":
        out = np.zeros((N, F), dtype=np.float32)
        fn = _divergence_covariant if covariant else _divergence_isotropic
        fn(
            kernel.gradient,
            prepare_fn,
            kernel.sigma,
            dim,
            F,
            fields,
            weights,
            r_ij,
            h,
            _EPS,
            out,
        )
        return out

    if mode == "curl":
        C = 1 if dim == 2 else dim
        out = np.zeros((N, F, C), dtype=np.float32)
        fn = _curl_covariant if covariant else _curl_isotropic
        fn(
            kernel.gradient,
            prepare_fn,
            kernel.sigma,
            dim,
            F,
            fields,
            weights,
            r_ij,
            h,
            _EPS,
            out,
        )
        return out[..., 0] if dim == 2 else out

    raise ValueError(
        f"Unsupported interpolation mode '{mode}'. Expected one of 'field', 'gradient', 'divergence', 'curl'."
    )
