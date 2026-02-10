import numpy as np
import numpy.typing as npt
from typing import Optional, Literal
import math


class Kernel:
	"""Base class for SPH kernels supporting isotropic and anisotropic smoothing.

	Implements evaluation of kernel functions and their gradients for various
	SPH kernel types in 1D, 2D, and 3D.
	"""
	
	def __init__(
		self, 
		kernel_name: str, 
		dim: int
	) -> None:
		"""Initialize a kernel.

		Parameters
		----------
		kernel_name
			Kernel name: ``"gaussian"``, ``"cubic_spline"``, ``"quintic_spline"``,
			``"wendland_c2"``, ``"wendland_c4"``, ``"wendland_c6"``, or ``"super_gaussian"``.
		dim
			Spatial dimension (1, 2, or 3).
		"""
		self.kernel_name = kernel_name
		self.dim = dim


	def evaluate_kernel(
		self, 
		r_ij: npt.NDArray[np.floating],
		h: Optional[npt.NDArray[np.floating]] = None, 
		H: Optional[npt.NDArray[np.floating]] = None
	) -> npt.NDArray[np.floating]:
		"""Evaluate the kernel function $W$ at given distances.

		Supports both isotropic smoothing (scalar ``h``) and anisotropic smoothing (tensor ``H``).

		Parameters
		----------
		r_ij
			Distance array with shape ``(M, K)`` for isotropic case or ``(M, K, D)``
			for anisotropic case (relative position vectors). ``M`` is the number of
			query positions, ``K`` the number of neighbors, and ``D`` the dimension.
		h
			Smoothing lengths with shape ``(M, 1)`` or ``(M,)`` (isotropic only).
		H
			Smoothing tensors with shape ``(M, D, D)`` (anisotropic only).

		Returns
		-------
		numpy.ndarray
			Kernel values with shape ``(M, K)``, normalized by the kernel constant
			and smoothing scale(s).

		Raises
		------
		ValueError
			If neither ``h`` nor ``H`` is provided, or if shapes are incompatible.
		"""

		# isotropic case
		if h is not None:
			h = h[:, np.newaxis] if len(h.shape) == 1 else h
			q = r_ij / h
			norm = h ** self.dim

		# anisotropic case
		else:
			if H is None:
				raise ValueError("Anisotropic kernel evaluation requires 'H'")
			if len(r_ij.shape) != 3:
				raise ValueError("For anisotropic kernels, r_ij must be vectors of relative positions with shape (N, M, d)")
			H_inv = np.linalg.inv(H)  						# (N, d, d)
			det_H = np.linalg.det(H)  						# (N,)
			
			# Transform to smoothing space
			xi = np.einsum('mij,mkj->mki', H_inv, r_ij)
			q = np.linalg.norm(xi, axis=-1)               	# (N, M)
			norm = det_H  									# (N,)

		# compute the kernel values
		W = self._kernel_sigma() / norm * self._kernel_values(q)
		return W


	def evaluate_gradient(
		self, 
		r_ij_vec: npt.NDArray[np.floating],
		h: Optional[npt.NDArray[np.floating]] = None, 
		H: Optional[npt.NDArray[np.floating]] = None
	) -> npt.NDArray[np.floating]:
		"""Evaluate the kernel gradient $\\nabla W$ at given positions.

		Supports both isotropic and anisotropic smoothing with proper chain rule handling.

		Parameters
		----------
		r_ij_vec
			Relative position vectors with shape ``(M, K, D)`` where ``M`` is the
			number of query positions, ``K`` the number of neighbors, and ``D`` the
			dimension.
		h
			Smoothing lengths with shape ``(M, 1)`` or ``(M,)`` (isotropic only).
		H
			Smoothing tensors with shape ``(M, D, D)`` (anisotropic only).

		Returns
		-------
		numpy.ndarray
			Kernel gradients with shape ``(M, K, D)``, normalized by the kernel
			constant and smoothing scale(s).

		Raises
		------
		ValueError
			If neither ``h`` nor ``H`` is provided, or if shapes are incompatible.
		"""

		# ======================
		# Isotropic case
		# ======================
		if h is not None:
			h = h[:, np.newaxis] if len(h.shape) == 1 else h  # (N, 1)

			r_ij_mag = np.linalg.norm(r_ij_vec, axis=-1)      # (N, M)
			q = r_ij_mag / h                                  # (N, M)
			
			dW_dq = self._kernel_gradient_values(q)
			dW_dr = dW_dq / h                          		  # (N, M)

			er = r_ij_vec / r_ij_mag[..., np.newaxis]		  # (N, M, d)
			grad_W = self._kernel_sigma() / h**self.dim * dW_dr[..., np.newaxis] * er              # (N, M, d)
			return grad_W

		# ======================
		# Anisotropic case
		# ======================
		else:
			if H is None:
				raise ValueError("Anisotropic kernel gradient requires 'H'")
			if len(r_ij_vec.shape) != 3:
				raise ValueError(
					"For anisotropic kernels, r_ij_vec must have shape (N, M, d)"
				)

			H_inv = np.linalg.inv(H)                           # (N, d, d)
			H_inv_T = np.transpose(H_inv, (0, 2, 1))           # (N, d, d)
			det_H = np.linalg.det(H)                           # (N,)

			# Smoothing-space coordinates
			xi = np.einsum('nij,nmj->nmi', H_inv, r_ij_vec)    # (N, M, d)
			q = np.linalg.norm(xi, axis=-1)                    # (N, M)

			# dK/dq
			dK_dq = self._kernel_gradient_values(q)

			# ∇q = H⁻ᵀ ξ / q
			grad_q = np.einsum('nij,nmj->nmi', H_inv_T, xi) / (q[..., None] + 1e-12)                         # (N, M, d)

			# ∇W = (1/detH) dK/dq ∇q
			grad_W = self._kernel_sigma() / det_H[:, None, None] * dK_dq[..., None] * grad_q
			return grad_W


	def _kernel_sigma(self) -> float:
		"""Compute the normalization constant for the kernel.

		Returns
		-------
		float
			Normalization constant depending on kernel type and dimension.

		Raises
		------
		ValueError
			If ``kernel_name`` is not recognized.
		"""

		if self.kernel_name == 'gaussian':
			if self.dim == 1:
				return 1.0 / math.pi ** 0.5
			elif self.dim == 2:
				return 1.0 / math.pi
			elif self.dim == 3:
				return 1.0 / math.pi ** (3.0 / 2.0)

		if self.kernel_name == 'super_gaussian':
			return 1.0 / math.pi ** (self.dim / 2.0)
		
		if self.kernel_name == 'cubic_spline':
			if self.dim == 1:
				return 4.0 / 3.0
			elif self.dim == 2:
				return 40.0 / (7.0 * math.pi)
			elif self.dim == 3:
				return 8.0 / math.pi

		if self.kernel_name == 'quintic_spline':
			if self.dim == 1:
				return 1.0 / 120.0
			elif self.dim == 2:
				return 7.0 / (478.0 * math.pi)
			elif self.dim == 3:
				return 3.0 / (359.0 * math.pi)
		
		if self.kernel_name == 'wendland_c2':
			if self.dim == 1:
				return 5.0 / 8.0
			elif self.dim == 2:
				return 7.0 / (4.0 * math.pi)
			elif self.dim == 3:
				return 21.0 / (16.0 * math.pi)
		
		if self.kernel_name == 'wendland_c4':
			if self.dim == 1:
				return 3.0 / 4.0
			elif self.dim == 2:
				return 9.0 / (4.0 * math.pi)
			elif self.dim == 3:
				return 495.0 / (256.0 * math.pi)
		
		if self.kernel_name == 'wendland_c6':
			if self.dim == 1:
				return 55.0 / 64.0
			elif self.dim == 2:
				return 78.0 / (28.0 * math.pi)
			elif self.dim == 3:
				return 1365.0 / (512.0 * math.pi)


	def _kernel_values(self, q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
		"""Evaluate the kernel support function for normalized distance $q = r/h$.

		Parameters
		----------
		q
			Normalized distances (dimensionless), typically in range ``[0, 3]``
			depending on kernel support.

		Returns
		-------
		numpy.ndarray
			Kernel values $W(q)$ of the same shape as input with compact support applied.
		"""
		
		if self.kernel_name == 'gaussian':
			mask = q <= 3
			return np.where(mask, np.exp(-q**2), 0.0)
		
		if self.kernel_name == 'super_gaussian':
			mask = q <= 3
			return np.where(mask, np.exp(-q ** 2) * (self.dim / 2 + 1 - q ** 2), 0.0)
		
		if self.kernel_name == 'cubic_spline':
			mask1 = (q <= 0.5)
			mask2 = (q > 0.5) & (q <= 1)
			W = np.where(mask1, 1 - 6 * q ** 2 + 6 * q ** 3, 0.0)
			W = np.where(mask2, 2 * (1 - q) ** 3, W)
			return W
		
		if self.kernel_name == 'quintic_spline':
			mask1 = (q >= 0) & (q <= 1)
			mask2 = (q > 1)  & (q <= 2)
			mask3 = (q > 2)  & (q <= 3)
			W = np.where(mask1, (3-q)**5 - 6*(2-q)**5 + 15*(1-q)**5, 0.0)
			W = np.where(mask2, (3-q)**5 - 6*(2-q)**5, W)
			W = np.where(mask3, (3-q)**5, W)
			return W
		
		if self.kernel_name == 'wendland_c2':
			mask = q <= 2
			if self.dim == 1:
				return np.where(mask, (1 - q / 2.0) ** 3 * (1.5 * q + 1), 0.0)
			else:
				return np.where(mask, (1 - q / 2.0) ** 4 * (2 * q + 1), 0.0)
		
		if self.kernel_name == 'wendland_c4':
			mask = q <= 2
			if self.dim == 1:
				return np.where(mask, (1 - q / 2.0) ** 5 * (2 * q ** 2 + 2.5 * q + 1), 0.0)
			else:
				return np.where(mask, (1 - q / 2.0) ** 6 * (35 / 12 * q ** 2 + 3 * q + 1), 0.0)
		
		if self.kernel_name == 'wendland_c6':
			mask = q <= 2
			if self.dim == 1:
				return np.where(mask, (1 - q / 2.0) ** 7 * (21.0 / 8.0 * q ** 3 + 19.0 / 4.0 * q ** 2 + 3.5 * q + 1), 0.0)
			else:
				return np.where(mask, (1 - q / 2.0) ** 8 * (4.0 * q ** 3 + 6.25 * q ** 2 + 4.0 * q + 1), 0.0)


	def _kernel_gradient_values(self, q: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
		"""Evaluate the kernel support derivative $dW/dq$.

		Parameters
		----------
		q
			Normalized distances (dimensionless).

		Returns
		-------
		numpy.ndarray
			Kernel gradient values $dW/dq$ of the same shape as input.
		"""
		
		if self.kernel_name == 'gaussian':
			mask = q <= 3
			return np.where(mask, -2 * q * np.exp(-q**2), 0.0)
		
		if self.kernel_name == 'super_gaussian':
			mask = q <= 3
			return np.where(mask, -2 * q * (self.dim / 2 + 2 - q ** 2) * np.exp(-q ** 2), 0.0)
		
		if self.kernel_name == 'cubic_spline':
			mask1 = (q <= 0.5)
			mask2 = (q > 0.5) & (q <= 1)
			dW_dq = np.where(mask1, -6 * q * (2 - 3 * q), 0.0)
			dW_dq = np.where(mask2, -6 * (1.0 - q) ** 2, dW_dq)
			return dW_dq
		
		if self.kernel_name == 'quintic_spline':
			mask1 = (q >= 0) & (q <= 1)
			mask2 = (q > 1)  & (q <= 2)
			mask3 = (q > 2)  & (q <= 3)
			dW_dq = np.where(mask1, -5 * (3 - q)**4 + 30 * (2 - q)**4 - 75 * (1 - q)**4, 0.0)
			dW_dq = np.where(mask2, -5 * (3 - q)**4 + 30 * (2 - q)**4, dW_dq)
			dW_dq = np.where(mask3, -5 * (3 - q)**4, dW_dq)
			return dW_dq
		
		if self.kernel_name == 'wendland_c2':
			mask = q <= 2
			z = 1 - 0.5 * q
			if self.dim == 1:
				dW_dq = np.where(mask, -1.5 * z ** 2 * (1.5 * q + 1) + 1.5 * z ** 3, 0.0)
			else:
				dW_dq = np.where(mask, -2 * z ** 3 * (2 * q + 1) + 2 * z ** 4, 0.0)
			return dW_dq
		
		if self.kernel_name == 'wendland_c4':
			mask = q <= 2
			z = 1 - 0.5 * q
			if self.dim == 1:
				f = z ** 5
				df = -2.5 * z ** 4
				g = 2 * q ** 2 + 2.5 * q + 1
				dg = 4 * q + 2.5
			else:
				f = z ** 6
				df = -3 * z ** 5
				g = (35.0 / 12.0) * q ** 2 + 3 * q + 1
				dg = (35.0 / 6.0) * q + 3
			dW_dq = np.where(mask, df * g + f * dg, 0.0)
			return dW_dq
		
		if self.kernel_name == 'wendland_c6':
			mask = q <= 2
			z = 1 - 0.5 * q
			if self.dim == 1:
				f = z ** 7
				df = -3.5 * z ** 6
				g = 21.0 / 8.0 * q ** 3 + 19.0 / 4.0 * q ** 2 + 3.5 * q + 1
				dg = 63.0 / 8.0 * q ** 2 + 19.0 / 2.0 * q + 3.5
			else:
				f = z ** 8
				df = -4 * z ** 7
				g = 4.0 * q ** 3 + 6.25 * q ** 2 + 4.0 * q + 1
				dg = 12.0 * q ** 2 + 12.5 * q + 4.0

			dW_dq = np.where(mask, df * g + f * dg, 0.0)
			return dW_dq

