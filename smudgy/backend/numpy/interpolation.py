import numpy as np

from .kernels import get_kernel, prepare_kernel_inputs


# =============================================================================
# Density
# =============================================================================
def density(
    *,
    kernel_name,
    dim,
    neighbor_weights,
    r_ij,
    h,
    structure,
):
    """Compute SPH density using isotropic or covariant smoothing."""
    if structure == "separable":
        raise ValueError("structure='separable' is not supported for particle density.")

    kernel = get_kernel(
        kernel_name,
        dim=dim,
    )

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
    fields,
    weights,
    r_ij,
    h,
    mode,
    structure,
):
    """Interpolate fields using isotropic or covariant smoothing."""
    if structure == "separable":
        raise ValueError(
            "structure='separable' is not supported for particle interpolation."
        )

    kernel = get_kernel(kernel_name, dim=dim)

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
