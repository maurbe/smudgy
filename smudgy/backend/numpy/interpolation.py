import numpy as np

from ..neighbors import coordinate_difference_with_pbc
from .kernels import get_kernel, prepare_kernel_inputs


def _support_scaled(h, kernel):
    """Rescale h so the kernel's compact support does not extend past q = 1.

    In particle mode `h` is the distance to the K-th neighbour (isotropic) or
    the tensor whose ellipsoid passes through the farthest neighbour
    (covariant), and only those K neighbours are summed. A kernel whose
    support is `s` > 1 times h would be truncated there, so h -> h / s.
    Kernels with s <= 1 are left as they are. Returns a new array; `h`
    itself is never modified.
    """
    if kernel.support <= 1.0:
        return h
    return np.asarray(h) / np.float32(kernel.support)


def _gather_rel_coords(query_positions, positions, nn_inds, boxsize):
    """r_ij = x_i - x_j for every (query, neighbor) pair, shape (N, K, D)."""
    return coordinate_difference_with_pbc(
        np.asarray(query_positions)[:, np.newaxis, :],
        np.asarray(positions)[nn_inds],
        boxsize,
    )


# =============================================================================
# Density
# =============================================================================
def density(
    *,
    kernel_name,
    dim,
    query_positions,
    positions,
    weights,
    nn_inds,
    h,
    boxsize,
    structure,
):
    """Compute SPH density using isotropic or covariant smoothing.

    `positions`/`weights` are per-particle arrays that `nn_inds` (N, K)
    indexes into; `h` is the smoothing length/tensor of each query. The
    kernel is scaled so its support does not exceed `h` (see `_support_scaled`).
    """
    if structure == "separable":
        raise ValueError("structure='separable' is not supported for particle density.")

    kernel = get_kernel(
        kernel_name,
        dim=dim,
    )
    h = _support_scaled(h, kernel)
    r_ij = _gather_rel_coords(query_positions, positions, nn_inds, boxsize)
    neighbor_weights = np.asarray(weights)[nn_inds]

    q, _, scale = prepare_kernel_inputs(
        kernel=kernel,
        structure=structure,
        r_ij=r_ij,
        h=h,
        dim=dim,
    )
    w = kernel.evaluate_coords(q, scale)

    return np.sum(
        neighbor_weights * w,
        axis=1,
    )


# =============================================================================
# Interpolate
# =============================================================================
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
    """Interpolate fields using isotropic or covariant smoothing.

    `positions`, `weights` (m_j / rho_j), `fields` and `h` are per-particle
    arrays that `nn_inds` (N, K) indexes into. The kernel is scaled so its
    support does not exceed `h` (see `_support_scaled`).
    """
    if structure == "separable":
        raise ValueError(
            "structure='separable' is not supported for particle interpolation."
        )

    kernel = get_kernel(kernel_name, dim=dim)
    h = _support_scaled(h, kernel)
    r_ij = _gather_rel_coords(query_positions, positions, nn_inds, boxsize)
    fields = np.asarray(fields)[nn_inds]
    weights = np.asarray(weights)[nn_inds]
    h = np.asarray(h)[nn_inds]

    q, grad_q, scale = prepare_kernel_inputs(
        kernel=kernel,
        structure=structure,
        r_ij=r_ij,
        h=h,
        dim=dim,
    )

    if mode == "field":
        w = kernel.evaluate_coords(q, scale)
        return np.einsum("...kf,...k,...k->...f", fields, w, weights)

    w_grad = kernel.evaluate_gradient_coords(q, grad_q, scale)

    if mode == "gradient":
        return np.einsum("...kf,...kd,...k->...fd", fields, w_grad, weights)

    if mode == "divergence":
        return np.einsum("...kfd,...kd,...k->...f", fields, w_grad, weights)

    if mode == "curl":
        if dim == 1:
            raise NotImplementedError

        if dim == 2:
            curl = (
                fields[..., 0] * w_grad[..., 1, None]
                - fields[..., 1] * w_grad[..., 0, None]
            )
            return np.einsum("...kf,...k->...f", curl, weights)

        else:
            curl = np.cross(fields, w_grad[..., None, :], axis=-1)
            return np.einsum("...kfd,...k->...fd", curl, weights)

    raise ValueError(
        f"Unsupported interpolation mode '{mode}'. "
        "Expected one of 'field', 'gradient', 'divergence', 'curl'."
    )
