"""SPH kernel functions and their gradients for isotropic and covariant smoothing."""

import numpy as np
import numpy.typing as npt

from ..kernel_table import SPHERICAL, spherical_constants

# =============================================================================
# Utility functions
# =============================================================================


def _prepare_isotropic_inputs(kernel, r_ij, h, eps):
    """Project relative vectors onto isotropic kernel coordinates."""
    r_mag = np.linalg.norm(r_ij, axis=-1)
    if h.ndim == r_ij.ndim - 2:
        h = h[..., None]

    q = r_mag / (h + eps)
    scale = kernel._kernel_sigma() / (h**kernel.dim)

    grad_q = np.zeros_like(r_ij)
    np.divide(
        r_ij,
        (r_mag[..., None] + eps) * (h[..., None] + eps),
        out=grad_q,
        where=r_mag[..., None] > 0,
    )

    return q, grad_q, scale


def _prepare_covariant_inputs(kernel, r_ij, H, eps):
    """Project relative vectors onto covariant kernel coordinates."""
    H_inv = np.linalg.inv(H)
    det_H = np.linalg.det(H)

    if H.ndim == r_ij.ndim + 1:
        # One smoothing tensor per query-neighbor pair: (M, K, D, D).
        xi = np.einsum("...ij,...j->...i", H_inv, r_ij)
        H_inv_T = np.swapaxes(H_inv, -1, -2)
        grad_numerator = np.einsum("...ij,...j->...i", H_inv_T, xi)
        scale = kernel._kernel_sigma() / det_H
    elif H.ndim == r_ij.ndim:
        # One smoothing tensor per query, shared by all its neighbors:
        # (M, D, D) with relative vectors shaped (M, K, D).
        xi = np.einsum("...ij,...kj->...ki", H_inv, r_ij)
        H_inv_T = np.swapaxes(H_inv, -1, -2)
        grad_numerator = np.einsum("...ij,...kj->...ki", H_inv_T, xi)
        scale = kernel._kernel_sigma() / det_H[..., None]
    else:
        raise ValueError(
            "Covariant smoothing tensors must have shape (M, D, D) or "
            "(M, K, D, D) for relative vectors shaped (M, K, D)."
        )

    q = np.linalg.norm(xi, axis=-1)
    grad_q = grad_numerator / (q[..., None] + eps)
    return q, grad_q, scale


def prepare_kernel_inputs(kernel, structure, r_ij, h, dim, eps=1e-7):
    """Normalize separations for kernel evaluation with smoothing lengths or tensors."""
    if dim not in (1, 2, 3):
        raise ValueError(f"Invalid dim '{dim}'. Must be 1, 2, or 3.")

    if structure == "isotropic":
        return _prepare_isotropic_inputs(kernel, r_ij, h, eps)
    if structure == "covariant":
        return _prepare_covariant_inputs(kernel, r_ij, h, eps)
    raise ValueError(f"Invalid structure '{structure}'")


# =============================================================================
# Base kernel class
# =============================================================================


class BaseKernelClass:
    """Base class for SPH kernels: shape W(q) here, constants from kernel_table."""

    name = "base_kernel"

    def __init__(self, dim: int) -> None:
        """Initialize the kernel for the given spatial dimension."""
        if not isinstance(dim, int) or dim not in (1, 2, 3):
            raise ValueError(
                f"`dim` must be an integer and one of 1, 2, or 3, but found {dim} of type {type(dim)}"
            )
        self.dim = dim
        self.support, sigma = spherical_constants(self.name)
        self.sigma = sigma[dim - 1]
        self.eps = 1e-7

    def evaluate_coords(self, q, scale):
        """Evaluate the kernel from canonical coordinates and normalization scale."""
        res = scale * self._kernel_values(q)
        if q.dtype.kind == "f":
            return res.astype(q.dtype, copy=False)
        return res

    def evaluate_gradient_coords(self, q, grad_q, scale):
        """Evaluate the kernel gradient from canonical coordinates and direction."""
        res = scale[..., None] * self._kernel_gradient_values(q)[..., None] * grad_q
        if q.dtype.kind == "f":
            return res.astype(q.dtype, copy=False)
        return res

    def _kernel_sigma(self):
        return self.sigma

    def _kernel_values(self, q):
        raise NotImplementedError("Must implement _kernel_values in subclass")

    def _kernel_gradient_values(self, q):
        raise NotImplementedError("Must implement _kernel_gradient_values in subclass")


# =============================================================================
# Spherical kernels
# =============================================================================


class TophatKernel(BaseKernelClass):
    """Spherically symmetric Tophat kernel."""

    name = "tophat"

    def _kernel_values(self, q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        mask = q <= self.support
        return np.where(mask, 1.0, 0.0)

    def _kernel_gradient_values(
        self, q: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        # Gradient of Tophat is a delta function at q=1, numerically problematic.
        return np.zeros_like(q)


class TSCKernel(BaseKernelClass):
    """Spherically symmetric triangular (tent) kernel in radius."""

    name = "tsc"

    def _kernel_values(self, q):
        q = np.abs(q)
        val = np.zeros_like(q)

        mask = q <= self.support
        val[mask] = 1.0 - q[mask] / self.support

        return val

    def _kernel_gradient_values(self, q):
        q = np.abs(q)
        grad = np.zeros_like(q)

        mask = q <= self.support
        grad[mask] = -(1.0 / self.support)

        return grad


class GaussianKernel(BaseKernelClass):
    """Gaussian kernel implementation for SPH."""

    name = "gaussian"

    def _kernel_values(self, q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        mask = q <= self.support
        return np.where(mask, np.exp(-(q**2)), 0.0)

    def _kernel_gradient_values(
        self, q: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        mask = q <= self.support
        return np.where(mask, -2 * q * np.exp(-(q**2)), 0.0)


class LucyKernel(BaseKernelClass):
    """Lucy kernel implementation for SPH."""

    name = "lucy"

    def _kernel_values(self, q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        mask = q <= self.support
        return np.where(mask, (1 + 3 * q) * (1 - q) ** 3, 0.0)

    def _kernel_gradient_values(
        self, q: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        mask = q <= self.support
        return np.where(mask, -12 * q * (1 - q) ** 2, 0.0)


class CubicSplineKernel(BaseKernelClass):
    """Cubic spline kernel implementation for SPH."""

    name = "cubic_spline"

    def _kernel_values(self, q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        mask1 = q <= 0.5
        mask2 = (q > 0.5) & (q <= 1)
        W = np.where(mask1, 1 - 6 * q**2 + 6 * q**3, 0.0)
        W = np.where(mask2, 2 * (1 - q) ** 3, W)
        return W

    def _kernel_gradient_values(
        self, q: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        mask1 = q <= 0.5
        mask2 = (q > 0.5) & (q <= 1)
        dW_dq = np.where(mask1, -6 * q * (2 - 3 * q), 0.0)
        dW_dq = np.where(mask2, -6 * (1.0 - q) ** 2, dW_dq)
        return dW_dq


class QuinticSplineKernel(BaseKernelClass):
    """Quintic spline kernel implementation for SPH."""

    name = "quintic_spline"

    def _kernel_values(self, q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        mask1 = (q >= 0) & (q <= 1)
        mask2 = (q > 1) & (q <= 2)
        mask3 = (q > 2) & (q <= 3)
        W = np.where(mask1, (3 - q) ** 5 - 6 * (2 - q) ** 5 + 15 * (1 - q) ** 5, 0.0)
        W = np.where(mask2, (3 - q) ** 5 - 6 * (2 - q) ** 5, W)
        W = np.where(mask3, (3 - q) ** 5, W)
        return W

    def _kernel_gradient_values(
        self, q: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        mask1 = q <= 1
        mask2 = (q > 1) & (q <= 2)
        mask3 = (q > 2) & (q <= 3)
        dW_dq = np.where(
            mask1, -5 * (3 - q) ** 4 + 30 * (2 - q) ** 4 - 75 * (1 - q) ** 4, 0.0
        )
        dW_dq = np.where(mask2, -5 * (3 - q) ** 4 + 30 * (2 - q) ** 4, dW_dq)
        dW_dq = np.where(mask3, -5 * (3 - q) ** 4, dW_dq)
        return dW_dq


class WendlandC2Kernel(BaseKernelClass):
    """Wendland C2 kernel implementation for SPH."""

    name = "wendland_c2"

    def _kernel_values(self, q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        mask = q <= self.support
        if self.dim == 1:
            return np.where(mask, (1 - q / 2.0) ** 3 * (1.5 * q + 1), 0.0)
        else:
            return np.where(mask, (1 - q / 2.0) ** 4 * (2 * q + 1), 0.0)

    def _kernel_gradient_values(
        self, q: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        mask = q <= self.support
        z = 1 - 0.5 * q
        if self.dim == 1:
            return np.where(mask, -1.5 * z**2 * (1.5 * q + 1) + 1.5 * z**3, 0.0)
        else:
            return np.where(mask, -2 * z**3 * (2 * q + 1) + 2 * z**4, 0.0)


class WendlandC4Kernel(BaseKernelClass):
    """Wendland C4 kernel implementation for SPH."""

    name = "wendland_c4"

    def _kernel_values(self, q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        mask = q <= self.support
        if self.dim == 1:
            return np.where(mask, (1 - q / 2.0) ** 5 * (2 * q**2 + 2.5 * q + 1), 0.0)
        else:
            return np.where(
                mask, (1 - q / 2.0) ** 6 * (35 / 12 * q**2 + 3 * q + 1), 0.0
            )

    def _kernel_gradient_values(
        self, q: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        mask = q <= self.support
        z = 1 - 0.5 * q
        if self.dim == 1:
            f, df = z**5, -2.5 * z**4
            g, dg = 2 * q**2 + 2.5 * q + 1, 4 * q + 2.5
        else:
            f, df = z**6, -3 * z**5
            g, dg = (35.0 / 12.0) * q**2 + 3 * q + 1, (35.0 / 6.0) * q + 3
        return np.where(mask, df * g + f * dg, 0.0)


class WendlandC6Kernel(BaseKernelClass):
    """Wendland C6 kernel implementation for SPH."""

    name = "wendland_c6"

    def _kernel_values(self, q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        mask = q <= self.support
        if self.dim == 1:
            return np.where(
                mask,
                (1 - q / 2.0) ** 7
                * (21.0 / 8.0 * q**3 + 19.0 / 4.0 * q**2 + 3.5 * q + 1),
                0.0,
            )
        else:
            return np.where(
                mask,
                (1 - q / 2.0) ** 8 * (4.0 * q**3 + 6.25 * q**2 + 4.0 * q + 1),
                0.0,
            )

    def _kernel_gradient_values(
        self, q: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        mask = q <= self.support
        z = 1 - 0.5 * q
        if self.dim == 1:
            f, df = z**7, -3.5 * z**6
            g, dg = (21.0 / 8.0 * q**3 + 19.0 / 4.0 * q**2 + 3.5 * q + 1), (
                63.0 / 8.0 * q**2 + 19.0 / 2.0 * q + 3.5
            )
        else:
            f, df = z**8, -4 * z**7
            g, dg = (4.0 * q**3 + 6.25 * q**2 + 4.0 * q + 1), (
                12.0 * q**2 + 12.5 * q + 4.0
            )
        return np.where(mask, df * g + f * dg, 0.0)


# =============================================================================
# Registries
# =============================================================================

KERNEL_CLASSES = {
    cls.name: cls
    for cls in (
        TophatKernel,
        TSCKernel,
        GaussianKernel,
        LucyKernel,
        CubicSplineKernel,
        QuinticSplineKernel,
        WendlandC2Kernel,
        WendlandC4Kernel,
        WendlandC6Kernel,
    )
}
assert set(KERNEL_CLASSES) == set(SPHERICAL)


def get_kernel(kernel_name: str, dim: int) -> BaseKernelClass:
    """Return a kernel instance for the given kernel name and spatial dimension.

    Parameters
    ----------
    kernel_name : str
            Name of the kernel to create.
    dim : int
            Spatial dimension (1, 2, or 3).

    Returns
    -------
    BaseKernelClass
            An instance of the specified kernel class.

    Raises
    ------
            If ``kernel_name`` is not recognized or if ``dim`` is not valid.

    """
    if kernel_name not in KERNEL_CLASSES:
        raise ValueError(
            f"Invalid kernel_name '{kernel_name}'. Must be one of {list(KERNEL_CLASSES.keys())}."
        )

    if dim not in (1, 2, 3):
        raise ValueError(f"Invalid dim '{dim}'. Must be 1, 2, or 3.")

    return KERNEL_CLASSES[kernel_name](dim=dim)
