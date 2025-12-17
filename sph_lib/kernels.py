import numpy as np
import math

# DONE
class gaussian():

	def __init__(self, dim):
		"""
		dim: spatial dimension
		"""
		self.dim = dim

	def sigma(self):
		"""
		Compute the normalization factor depending on the spatial dimension.
		"""
		if self.dim == 1:
			sigma = 1.0 / math.pi ** 0.5
		elif self.dim == 2:
			sigma = 1.0 / math.pi
		elif self.dim == 3:
			sigma = 1.0 / math.pi ** (3.0 / 2.0)
		return sigma

	def evaluate_kernel(self, r_ij, h):
		"""
		Evaluate the Gaussian kernel.
		r_ij: 		(N, M) relative distances between particles
		h:   		(N,) or (N, 1) smoothing lengths
		Returns: 	(N, M) values of the Gaussian kernel
		"""
		h = h[:, np.newaxis] if len(h.shape) == 1 else h
		norm = self.sigma() / h ** self.dim
		
		q = r_ij / h
		mask = q <= 3
		
		W = np.where(mask, np.exp(-q**2), 0.0)
		return norm * W

	def evaluate_gradient(self, r_ij_vec, h):
		"""
		Evaluate the kernel value W for pairwise distances r_ij.
		"""
		h = h[:, np.newaxis] if len(h.shape) == 1 else h  # (N, 1)
		norm = self.sigma() / h ** self.dim         	  # (N, 1)

		r_ij_mag = np.linalg.norm(r_ij_vec, axis=-1)      # (N, M)
		q = r_ij_mag / h                                  # (N, M)

		mask = q <= 3
		dW_dq = np.where(mask, -2 * q * np.exp(-q ** 2), 0.0)  # (N, M)
		dW_dr = norm * dW_dq / h

		er = r_ij_vec / r_ij_mag[..., np.newaxis]        # (N, M, D)
		grad_W = dW_dr[..., np.newaxis] * er             # (N, M, D)
		return grad_W

# DONE
class super_gaussian():

	def __init__(self, dim):
		self.dim = dim

	def sigma(self):
		return 1.0 / math.pi ** (self.dim / 2.0)

	def evaluate_kernel(self, r_ij, h):
		"""
		Compute Super-Gaussian kernel.

		Parameters:
		- r_ij: pairwise distances, shape (N, M)
		- h: smoothing lengths, shape (N,) or (N, 1)

		Returns:
		- W: kernel values, shape (N, M)
		"""
		h = h[:, np.newaxis] if h.ndim == 1 else h
		norm = self.sigma() / h ** self.dim
		q = r_ij / h

		mask = q <= 3
		W = np.where(mask, np.exp(-q ** 2) * (self.dim / 2 + 1 - q ** 2), 0.0)
		return norm * W

	def evaluate_gradient(self, r_ij_vec, h):
		"""
		Compute gradient of the Super-Gaussian kernel.

		Parameters:
		- r_ij_vec: pairwise vectors, shape (N, M, D)
		- h: smoothing lengths, shape (N,) or (N, 1)

		Returns:
		- grad: kernel gradients, shape (N, M, D)
		"""
		h = h[:, np.newaxis] if h.ndim == 1 else h
		sigma = self.sigma() / h ** self.dim

		r_ij_mag = np.linalg.norm(r_ij_vec, axis=-1)  # (N, M)
		er = r_ij_vec / r_ij_mag[..., np.newaxis]
		q = r_ij_mag / h

		# dW/dq
		dW_dq = -2 * q * (self.dim / 2 + 2 - q ** 2) * np.exp(-q ** 2)
		dW_dr = sigma * dW_dq / h
		grad = dW_dr[..., np.newaxis] * er  # shape (N, M, D)
		return grad

# DONE
class cubic_spline():

	def __init__(self, dim):
		"""
		dim: spatial dimension
		h:   smoothing lengths
		"""
		self.dim = dim

	def sigma(self):
		"""
		Compute the normalization constant for the cubic spline kernel.
		"""
		if self.dim == 1:
			sigma = 4.0 / 3.0
		elif self.dim == 2:
			sigma = 40.0 / (7.0 * math.pi)
		elif self.dim == 3:
			sigma = 8.0 / math.pi
		return sigma

	def evaluate_kernel(self, r_ij, h):
		"""
		Evaluate the kernel value W for pairwise distances r_ij.
		"""
		h = h[:, np.newaxis] if len(h.shape) == 1 else h
		norm = self.sigma() / h ** self.dim

		q = r_ij / h
		mask1 = (q <= 0.5)
		mask2 = (q > 0.5) & (q <= 1)
		W = np.where(mask1, 1 - 6 * q ** 2 + 6 * q ** 3, 0.0)
		W = np.where(mask2, 2 * (1 - q) ** 3, W)
		return norm * W

	def evaluate_gradient(self, r_ij_vec, h):
		"""
		Compute the gradient ∇W of the cubic spline kernel.

		Parameters:
			r_ij_vec: (N, M, dim) relative position vectors
			h:        (N,) or (N, 1) smoothing lengths

		Returns:
			grad:     (N, M, dim) gradient of the kernel
		"""
		h = h[:, np.newaxis] if h.ndim == 1 else h               # (N, 1)
		norm = self.sigma() / h ** self.dim               		 # (N, 1)

		r_ij_mag = np.linalg.norm(r_ij_vec, axis=-1)             # (N, M)
		q = r_ij_mag / h                                         # (N, M)

		# Unit direction vector er = r_ij_vec / |r_ij_vec|
		er = r_ij_vec / r_ij_mag[..., np.newaxis]

		# Piecewise derivative dW/dq (Cubic Spline)
		mask1 = (q <= 0.5)
		mask2 = (q > 0.5) & (q <= 1)

		dW_dq = np.where(mask1, -6 * q * (2 - 3 * q), 0.0)
		dW_dq = np.where(mask2, -6 * (1.0 - q) ** 2, dW_dq)

		# Final gradient: ∇W = norm * dW/dq * r̂
		dW_dr = norm * dW_dq / h                                # (N, M)
		grad = dW_dr[..., np.newaxis] * er                      # (N, M, D)

		return grad

# DONE
class quintic_spline():

	def __init__(self, dim):
		self.dim = dim
	
	def sigma(self):

		if self.dim == 1:
			sigma = 1.0 / 120
		elif self.dim == 2:
			sigma = 7.0 / (478 * math.pi)
		elif self.dim == 3:
			sigma = 3.0 / (359 * math.pi)

		return sigma
	
	def evaluate_kernel(self, r_ij, h):
		"""
		Compute Quintic Spline kernel.
		
		r_ij: pairwise distances, shape (N, M)
		h: smoothing lengths, shape (N,) or (N, 1)
		"""
		h = h[:, np.newaxis] if len(h.shape)==1 else h
		norm = self.sigma() / h ** self.dim

		q = r_ij / h
		mask1 = (q >= 0) & (q <= 1)
		mask2 = (q > 1)  & (q <= 2)
		mask3 = (q > 2)  & (q <= 3)
		W = np.where(mask1, (3-q)**5 - 6*(2-q)**5 + 15*(1-q)**5, 0.0)
		W = np.where(mask2, (3-q)**5 - 6*(2-q)**5, W)
		W = np.where(mask3, (3-q)**5, W)
		return norm * W


	def evaluate_gradient(self, r_ij_vec, h):
		"""
		Compute the gradient ∇W of the quintic spline kernel.

		r_ij_vec: pairwise vector distances, shape (N, M, D)
		h:        smoothing lengths, shape (N,) or (N, 1)
		Returns:  gradients, shape (N, M, D)
		"""
		# Ensure correct shape for h
		h = h[:, np.newaxis] if len(h.shape) == 1 else h
		norm = self.sigma() / h ** self.dim

		r_ij_mag = np.linalg.norm(r_ij_vec, axis=-1)          # (N, M)
		q = r_ij_mag / h                                      # (N, M)

		# Unit vectors r̂ = r_ij_vec / |r_ij_vec|
		er = r_ij_vec / r_ij_mag[..., np.newaxis]

		# Derivative dW/dq
		mask1 = (q >= 0) & (q <= 1)
		mask2 = (q > 1)  & (q <= 2)
		mask3 = (q > 2)  & (q <= 3)

		dW_dq = np.where(mask1, -5 * (3 - q)**4 + 30 * (2 - q)**4 - 75 * (1 - q)**4, 0.0)
		dW_dq = np.where(mask2, -5 * (3 - q)**4 + 30 * (2 - q)**4, dW_dq)
		dW_dq = np.where(mask3, -5 * (3 - q)**4, dW_dq)

		dW_dr = norm * dW_dq / h                              # (N, M)
		grad = dW_dr[..., np.newaxis] * er                    # (N, M, D)

		return grad

# DONE
class wendland_c2():

	def __init__(self, dim):
		self.dim = dim

	def sigma(self):
		if self.dim == 1:
			return 5.0 / 8.0
		elif self.dim == 2:
			return 7.0 / (4.0 * math.pi)
		elif self.dim == 3:
			return 21.0 / (16.0 * math.pi)

	def evaluate_kernel(self, r_ij, h):
		"""
		Compute Wendland C2 kernel.
		
		r_ij: pairwise distances, shape (N, M)
		h: smoothing lengths, shape (N,) or (N, 1)
		"""
		h = h[:, np.newaxis] if h.ndim == 1 else h
		norm = self.sigma() / h ** self.dim

		q = r_ij / h
		mask = q <= 2
		if self.dim == 1:
			W = np.where(mask, (1 - q / 2.0) ** 3 * (1.5 * q + 1), 0.0)
		else:
			W = np.where(mask, (1 - q / 2.0) ** 4 * (2 * q + 1), 0.0)
		return norm * W


	def evaluate_gradient(self, r_ij_vec, h):
		h = h[:, np.newaxis] if h.ndim == 1 else h
		norm = self.sigma() / h ** self.dim

		r_ij_mag = np.linalg.norm(r_ij_vec, axis=-1)  # (N, M)
		q = r_ij_mag / h                              # (N, M)

		# Safe unit vector: (N, M, D)
		er = np.divide(r_ij_vec, r_ij_mag[..., np.newaxis],
					out=np.zeros_like(r_ij_vec),
					where=r_ij_mag[..., np.newaxis] != 0)

		mask = q <= 2
		z = 1 - 0.5 * q

		if self.dim == 1:
			# 1D-specific derivative
			dW_dq = np.where(
				mask,
				-1.5 * z ** 2 * (1.5 * q + 1) + 1.5 * z ** 3,
				0.0
			)
		else:
			# General (≥2D) case
			dW_dq = np.where(
				mask,
				-2 * z ** 3 * (2 * q + 1) + 2 * z ** 4,
				0.0
			)

		dW_dr = norm * dW_dq / h
		grad = dW_dr[..., np.newaxis] * er
		return grad

# DONE
class wendland_c4():

	def __init__(self, dim):
		self.dim = dim

	def sigma(self):
		if self.dim == 1:
			return 3.0 / 4.0
		elif self.dim == 2:
			return 9.0 / (4.0 * math.pi)
		elif self.dim == 3:
			return 495.0 / (256.0 * math.pi)

	def evaluate_kernel(self, r_ij, h):
		"""
		Compute Wendland C4 kernel.

		Parameters:
		- r_ij: pairwise distances, shape (N, M)
		- h: smoothing lengths, shape (N,) or (N, 1)

		Returns:
		- W: kernel values, shape (N, M)
		"""
		h = h[:, np.newaxis] if h.ndim == 1 else h
		norm = self.sigma() / h ** self.dim

		q = r_ij / h
		mask = q <= 2
		if self.dim == 1:
			W = np.where(mask, (1 - q / 2.0) ** 5 * (2 * q ** 2 + 2.5 * q + 1), 0.0)
		else:
			W = np.where(mask, (1 - q / 2.0) ** 6 * (35 / 12 * q ** 2 + 3 * q + 1), 0.0)
		return norm * W


	def evaluate_gradient(self, r_ij_vec, h):
		h = h[:, np.newaxis] if h.ndim == 1 else h
		norm = self.sigma() / h ** self.dim

		r_ij_mag = np.linalg.norm(r_ij_vec, axis=-1)  # (N, M)
		q = r_ij_mag / h                              # (N, M)

		# Unit vectors
		er = np.divide(r_ij_vec, r_ij_mag[..., np.newaxis],
					out=np.zeros_like(r_ij_vec),
					where=r_ij_mag[..., np.newaxis] != 0)

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
		dW_dr = norm * dW_dq / h
		grad = dW_dr[..., np.newaxis] * er
		return grad

# Done
class wendland_c6:

	def __init__(self, dim):
		self.dim = dim

	def sigma(self):
		if self.dim == 1:
			return 55.0 / 64.0
		elif self.dim == 2:
			return 78.0 / (28.0 * math.pi)
		elif self.dim == 3:
			return 1365.0 / (512.0 * math.pi)

	def evaluate_kernel(self, r_ij, h):
		"""
		Wendland C6 kernel evaluation.
		"""
		h = h[:, np.newaxis] if h.ndim == 1 else h
		norm = self.sigma() / h ** self.dim

		q = r_ij / h
		mask = q <= 2

		if self.dim == 1:
			W = np.where(mask, (1 - q / 2.0) ** 7 * (21.0 / 8.0 * q ** 3 + 19.0 / 4.0 * q ** 2 + 3.5 * q + 1), 0.0)
		else:
			W = np.where(mask,
				(1 - q / 2.0) ** 8 * (4.0 * q ** 3 + 6.25 * q ** 2 + 4.0 * q + 1), 0.0)
		return norm * W

	def evaluate_gradient(self, r_ij_vec, h):
		h = h[:, np.newaxis] if h.ndim == 1 else h
		norm = self.sigma() / h ** self.dim

		r_ij_mag = np.linalg.norm(r_ij_vec, axis=-1)
		q = r_ij_mag / h

		# Unit vector, safe division
		er = np.divide(r_ij_vec, r_ij_mag[..., np.newaxis],
					out=np.zeros_like(r_ij_vec),
					where=r_ij_mag[..., np.newaxis] != 0)

		mask = q <= 2
		z = 1 - 0.5 * q

		if self.dim == 1:
			# 1D coefficients (keep as-is if your kernel is truly different)
			f = z ** 7
			df = -3.5 * z ** 6
			g = 21.0 / 8.0 * q ** 3 + 19.0 / 4.0 * q ** 2 + 3.5 * q + 1
			dg = 63.0 / 8.0 * q ** 2 + 19.0 / 2.0 * q + 3.5
		else:
			# Matching the kernel expression in evaluate_kernel()
			f = z ** 8
			df = -4 * z ** 7
			g = 4.0 * q ** 3 + 6.25 * q ** 2 + 4.0 * q + 1
			dg = 12.0 * q ** 2 + 12.5 * q + 4.0

		dW_dq = np.where(mask, df * g + f * dg, 0.0)
		dW_dr = norm * dW_dq / h
		grad = dW_dr[..., np.newaxis] * er
		return grad
