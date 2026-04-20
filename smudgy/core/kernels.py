"""SPH kernel functions and their gradients for isotropic and anisotropic smoothing."""

import math

import numpy as np
import numpy.typing as npt


class BaseKernelClass:
    """Base class for SPH kernels."""

    def __init__(self, dim: int = None, support: float = None) -> None:

        if not isinstance(dim, int) or dim not in (1, 2, 3):
            raise ValueError(f"`dim` must be an integer and one of 1, 2, or 3, but found {dim} of type {type(dim)}")
        self.dim = dim
        self.support = support
        self.eps = 1e-7
        self.name = "base_kernel"

    def evaluate(self, r_ij, h) -> npt.NDArray[np.floating]:

        if not isinstance(r_ij, np.ndarray):
            raise ValueError("r_ij must be a numpy array")
        if not isinstance(h, np.ndarray):
            raise ValueError("smoothing lengths must be a numpy array")

        if h.ndim < 3:
            return self._evaluate_isotropic(r_ij, h).astype(h.dtype)
        else:
            return self._evaluate_anisotropic(r_ij, h).astype(h.dtype)

    def evaluate_gradient(self, r_ij_vec, h) -> npt.NDArray[np.floating]:

        if not isinstance(r_ij_vec, np.ndarray):
            raise ValueError("r_ij_vec must be a numpy array")
        if not isinstance(h, np.ndarray):
            raise ValueError("smoothing lengths must be a numpy array")

        if h.ndim < 3:
            return self._evaluate_gradient_isotropic(r_ij_vec, h).astype(h.dtype)
        else:
            return self._evaluate_gradient_anisotropic(r_ij_vec, h).astype(h.dtype)

    def _evaluate_isotropic(
        self, r_ij: npt.NDArray[np.floating], h: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:

        if r_ij.ndim not in (1, 2):
            raise ValueError(
                f"`r_ij` must be of shape (N,) or (N, 1) for isotropic case but found shape {r_ij.shape}"
            )
        if h.ndim not in (1, 2):
            raise ValueError(
                f"`h` must be of shape (N,) or (N, 1) for isotropic case but found shape {h.shape}"
            )

        h = h[..., None] if h.ndim == 1 else h
        q = np.abs(r_ij) / h
        norm = h**self.dim
        return self._kernel_sigma() / norm * self._kernel_values(q)

    def _evaluate_anisotropic(
        self, r_ij: npt.NDArray[np.floating], H: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:

        if r_ij.ndim != 3:
            raise ValueError(
                f"`r_ij` must be of shape (N, M, d) for anisotropic kernels but found shape {r_ij.shape}"
            )
        if H.ndim != 3:
            raise ValueError(
                f"`H` must be a 3D array of shape (N, D, D) but found shape {H.shape}"
            )
        if H.shape[1] != self.dim or H.shape[2] != self.dim:
            raise ValueError(
                f"`H` must be (N, D, D) with D=dim but found shape {H.shape}"
            )

        H_inv = np.linalg.inv(H)  # (N, d, d)
        norm = np.linalg.det(H)  # (N,)
        xi = np.einsum("mij,mkj->mki", H_inv, r_ij)
        q = np.linalg.norm(xi, axis=-1)  # (N, M)
        return self._kernel_sigma() / norm[..., None] * self._kernel_values(q)

    def _evaluate_gradient_isotropic(
        self,
        r_ij_vec: npt.NDArray[np.floating],
        h: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        if not isinstance(h, np.ndarray):
            raise ValueError("smoothing_lengths must be a numpy array if provided")
        if h.ndim not in (1, 2):
            raise ValueError("smoothing_lengths must be 1D or 2D array")
        if r_ij_vec.ndim != 3:
            raise ValueError("r_ij_vec must be 3D (N, M, d) for isotropic case")

        h = np.asarray(h)[:, None, None] if h.ndim == 1 else h  # (N, 1, 1)
        r_ij_mag = np.linalg.norm(r_ij_vec, axis=-1)[..., None]  # (N, M, 1)
        q = r_ij_mag / h  # (N, M, 1)
        norm = h**self.dim  # (N, 1, 1)

        dW_dq = self._kernel_gradient_values(q)
        dW_dr = dW_dq / h  # (N, M, 1)

        # Safe division for when r_ij_mag is zero: set gradient to zero in that case
        er = np.zeros_like(r_ij_vec)
        np.divide(r_ij_vec, r_ij_mag, out=er, where=r_ij_mag != 0.0)
        return self._kernel_sigma() / norm * dW_dr * er  # (N, M, d)

    def _evaluate_gradient_anisotropic(
        self,
        r_ij_vec: npt.NDArray[np.floating],
        H: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        assert r_ij_vec.ndim == 3, "For anisotropic kernels, r_ij_vec must be (N, M, d)"
        assert H.ndim == 3, "Smoothing tensors must be a 3D array of shape (N, D, D)"
        assert (
            H.shape[1] == H.shape[2] == self.dim
        ), "Smoothing tensors must be (N, D, D) with D=dim"

        det_H = np.linalg.det(H)  # (N,)
        H_inv = np.linalg.inv(H)  # (N, d, d)
        H_inv_T = np.transpose(H_inv, (0, 2, 1))  # (N, d, d)

        xi = np.einsum("nij,nmj->nmi", H_inv, r_ij_vec)  # (N, M, d)
        q = np.linalg.norm(xi, axis=-1)[..., None]  # (N, M, 1)

        dK_dq = self._kernel_gradient_values(q)  # (N, M, 1)
        grad_q = np.einsum("nij,nmj->nmi", H_inv_T, xi) / (q + self.eps)  # (N, M, d)
        print("dK_dq", dK_dq.shape, "grad_q", grad_q.shape, "det_H", det_H.shape)
        return self._kernel_sigma() / det_H[:, None, None] * dK_dq * grad_q

    # def _regularize_tensor(self,
    #                       H: npt.NDArray[np.floating]
    #                       ) -> npt.NDArray[np.floating]:
    #    """Add small regularization to the smoothing tensor to prevent singularity."""
    #    return H + self.eps * np.eye(self.dim)[None, :, :]

    def _kernel_sigma(self):
        pass

    def _kernel_values(self):
        pass

    def _kernel_gradient_values(self):
        pass


class TophatKernel(BaseKernelClass):
    """Spherically symmetric Tophat kernel."""

    def __init__(self, dim):
        super().__init__(dim, support=0.5)
        self.name = "tophat"

    def _kernel_sigma(self) -> float:
        if self.dim == 1:
            return 1.0
        elif self.dim == 2:
            return 4.0 / math.pi
        elif self.dim == 3:
            return 6.0 / math.pi

    def _kernel_values(self, q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        mask = q <= self.support
        return np.where(mask, 1.0, 0.0)

    def _kernel_gradient_values(
        self, q: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        # Gradient of Tophat is a delta function at q=1, numerically problematic.
        return np.zeros_like(q)


class TophatSepKernel(BaseKernelClass):
    """Rectangular Tophat kernel (Product of 1D Tophats)."""

    def __init__(self, dim):
        super().__init__(dim, support=0.5)
        self.name = "tophat_separable"

    def _kernel_sigma(self) -> float:
        return 1.0

    def _kernel_values(self, q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        # For TophatRect, support is 0.5 in each dimension.
        q = np.abs(q)
        mask = q <= self.support
        return np.where(mask, 1.0, 0.0)

    def _kernel_gradient_values(
        self, q: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        return np.zeros_like(q)


class TSCKernel(BaseKernelClass):
    """Spherically symmetric Triangular-Shaped Cloud (TSC) kernel."""

    def __init__(self, dim):
        super().__init__(dim, support=1.5)
        self.name = "tsc"
        self.node_1 = self.support / 3

    def _kernel_sigma(self) -> float:
        if self.dim == 1:
            return 1.0
        elif self.dim == 2:
            return 32.0 / (13.0 * math.pi)
        elif self.dim == 3:
            return 2.0 / math.pi

    def _kernel_values(self, q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        val = np.zeros_like(q)
        mask1 = q <= self.node_1
        mask2 = (q > self.node_1) & (q <= self.support)
        val[mask1] = 0.75 - q[mask1] ** 2
        val[mask2] = 0.5 * (1.5 - q[mask2]) ** 2
        return val

    def _kernel_gradient_values(
        self, q: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        grad = np.zeros_like(q)
        mask1 = q <= self.node_1
        mask2 = (q > self.node_1) & (q <= self.support)
        grad[mask1] = -2.0 * q[mask1]
        grad[mask2] = -(self.support - q[mask2])
        return grad


class TSCSepKernel(BaseKernelClass):
    """Rectangular TSC kernel (Product of 1D TSCs)."""

    def __init__(self, dim):
        super().__init__(dim, support=1.5)
        self.name = "tsc_separable"
        self.node_1 = self.support / 3

    def _kernel_sigma(self) -> float:
        return 1.0

    def _kernel_values(self, q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        q = np.abs(q)
        val = np.zeros_like(q)
        mask1 = q <= self.node_1
        mask2 = (q > self.node_1) & (q <= self.support)
        val[mask1] = 0.75 - q[mask1] ** 2
        val[mask2] = 0.5 * (self.support - q[mask2]) ** 2
        return val

    def _kernel_gradient_values(
        self, q: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        q = np.abs(q)
        grad = np.zeros_like(q)
        mask1 = q <= self.node_1
        mask2 = (q > self.node_1) & (q <= self.support)
        grad[mask1] = -2.0 * q[mask1]
        grad[mask2] = -(self.support - q[mask2])
        return grad


class GaussianKernel(BaseKernelClass):
    """Gaussian kernel implementation for SPH."""

    def __init__(self, dim: int) -> None:
        super().__init__(dim=dim, support=3.0)
        self.name = "gaussian"

    def _kernel_sigma(self) -> float:
        if self.dim == 1:
            return 1.0 / math.pi**0.5
        elif self.dim == 2:
            return 1.0 / math.pi
        elif self.dim == 3:
            return 1.0 / math.pi**1.5

    def _kernel_values(self, q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        mask = q <= self.support
        return np.where(mask, np.exp(-(q**2)), 0.0)

    def _kernel_gradient_values(
        self, q: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        mask = q <= self.support
        return np.where(mask, -2 * q * np.exp(-(q**2)), 0.0)


class GaussianSepKernel(BaseKernelClass):
    """Gaussian kernel implementation for SPH."""

    def __init__(self, dim: int) -> None:
        super().__init__(dim=dim, support=3.0)
        self.name = "gaussian_separable"

    def _kernel_sigma(self) -> float:
        if self.dim == 1:
            return 1.0 / math.pi**0.5
        elif self.dim == 2:
            return 1.0 / math.pi
        elif self.dim == 3:
            return 1.0 / math.pi**1.5

    def _kernel_values(self, q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        q = np.abs(q)
        mask = q <= self.support
        return np.where(mask, np.exp(-(q**2)), 0.0)

    def _kernel_gradient_values(
        self, q: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        q = np.abs(q)
        mask = q <= self.support
        return np.where(mask, -2 * q * np.exp(-(q**2)), 0.0)


class LucyKernel(BaseKernelClass):
    """Lucy kernel implementation for SPH."""

    def __init__(self, dim: int) -> None:
        super().__init__(dim=dim, support=1.0)
        self.name = "lucy"

    def _kernel_sigma(self) -> float:
        """Compute the normalization constant for the Lucy kernel."""
        if self.dim == 1:
            return 5.0 / 4.0
        elif self.dim == 2:
            return 5.0 / math.pi
        elif self.dim == 3:
            return 105.0 / (16.0 * math.pi)

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

    def __init__(self, dim: int) -> None:
        super().__init__(dim=dim)
        self.name = "cubic_spline"

    def _kernel_sigma(self) -> float:
        if self.dim == 1:
            return 4.0 / 3.0
        elif self.dim == 2:
            return 40.0 / (7.0 * math.pi)
        elif self.dim == 3:
            return 8.0 / math.pi

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

    def __init__(self, dim: int) -> None:
        super().__init__(dim=dim)
        self.name = "quintic_spline"

    def _kernel_sigma(self) -> float:
        if self.dim == 1:
            return 1.0 / 120.0
        elif self.dim == 2:
            return 7.0 / (478.0 * math.pi)
        elif self.dim == 3:
            return 1.0 / (120.0 * math.pi)

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

    def __init__(self, dim: int) -> None:
        super().__init__(dim=dim)
        self.name = "wendland_c2"

    def _kernel_sigma(self) -> float:
        if self.dim == 1:
            return 5.0 / 8.0
        elif self.dim == 2:
            return 7.0 / (4.0 * math.pi)
        elif self.dim == 3:
            return 21.0 / (16.0 * math.pi)

    def _kernel_values(self, q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        mask = q <= 2
        if self.dim == 1:
            return np.where(mask, (1 - q / 2.0) ** 3 * (1.5 * q + 1), 0.0)
        else:
            return np.where(mask, (1 - q / 2.0) ** 4 * (2 * q + 1), 0.0)

    def _kernel_gradient_values(
        self, q: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        mask = q <= 2
        z = 1 - 0.5 * q
        if self.dim == 1:
            return np.where(mask, -1.5 * z**2 * (1.5 * q + 1) + 1.5 * z**3, 0.0)
        else:
            return np.where(mask, -2 * z**3 * (2 * q + 1) + 2 * z**4, 0.0)


class WendlandC4Kernel(BaseKernelClass):
    """Wendland C4 kernel implementation for SPH."""

    def __init__(self, dim: int) -> None:
        super().__init__(dim=dim)
        self.name = "wendland_c4"

    def _kernel_sigma(self) -> float:
        if self.dim == 1:
            return 3.0 / 4.0
        elif self.dim == 2:
            return 9.0 / (4.0 * math.pi)
        elif self.dim == 3:
            return 495.0 / (256.0 * math.pi)

    def _kernel_values(self, q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        mask = q <= 2
        if self.dim == 1:
            return np.where(mask, (1 - q / 2.0) ** 5 * (2 * q**2 + 2.5 * q + 1), 0.0)
        else:
            return np.where(
                mask, (1 - q / 2.0) ** 6 * (35 / 12 * q**2 + 3 * q + 1), 0.0
            )

    def _kernel_gradient_values(
        self, q: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        mask = q <= 2
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

    def __init__(self, dim: int) -> None:
        super().__init__(dim=dim)
        self.name = "wendland_c6"

    def _kernel_sigma(self) -> float:
        if self.dim == 1:
            return 55.0 / 64.0
        elif self.dim == 2:
            return 39.0 / (14.0 * math.pi)
        elif self.dim == 3:
            return 1365.0 / (512.0 * math.pi)

    def _kernel_values(self, q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        mask = q <= 2
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
        mask = q <= 2
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


KERNEL_CLASSES = {
    "tophat_separable": TophatSepKernel,
    "tsc_separable": TSCSepKernel,
    "gaussian_separable": GaussianSepKernel,
    "tophat": TophatKernel,
    "tsc": TSCKernel,
    "gaussian": GaussianKernel,
    "lucy": LucyKernel,
    "cubic_spline": CubicSplineKernel,
    "quintic_spline": QuinticSplineKernel,
    "wendland_c2": WendlandC2Kernel,
    "wendland_c4": WendlandC4Kernel,
    "wendland_c6": WendlandC6Kernel,
}


def get_kernel(kernel_name: str, dim: int) -> BaseKernelClass:
    """Factory function to create kernel instances based on name and dimension.

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
    assert (
        kernel_name in KERNEL_CLASSES
    ), f"Invalid kernel_name '{kernel_name}'. Must be one of {list(KERNEL_CLASSES.keys())}."
    assert dim in (1, 2, 3), f"Invalid dim '{dim}'. Must be 1, 2, or 3."

    return KERNEL_CLASSES[kernel_name](dim=dim)
