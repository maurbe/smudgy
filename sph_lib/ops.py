from typing import Any, Optional, Sequence, Tuple, Union, List, Literal
import numpy as np
import numpy.typing as npt

from .utils import (build_kdtree,
					query_kdtree,
					compute_hsm, 
                    compute_hsm_tensor, 
                    compute_pcellsize_half, 
                    project_hsm_tensor_to_2d,
					coordinate_difference_with_pbc
					)
from kernels import Kernel
from .python import functions as pyfunc
from .cpp import functions as cppfunc


class MainClass:

	def __init__(self,
				 positions: np.ndarray,
				 masses: np.ndarray,
				 boxsize: Optional[Union[float, Sequence[float]]] = None,
				 verbose: bool = True
				 ):
		
		self.dim = positions.shape[-1]
		assert self.dim==2 or self.dim==3, f"Particle positions must be of shape (N, 2) or (N, 3) but found {positions.shape}"
		
		self.pos = positions
		self.mass = masses
		self.verbose = verbose

		if boxsize is None:
			self.periodic = False
		else:
			self.periodic = True

		# construct boxsize array
		self.boxsize = boxsize
		if np.ndim(boxsize) == 0:
			self.boxsize = np.repeat(boxsize, self.dim)
		else:
			self.boxsize = np.asarray(boxsize)


	def compute_smoothing_lengths(
		self, 
		num_neighbors: int,
		mode: Literal['adaptive', 'isotropic', 'anisotropic'] = 'isotropic',
		query_pos: Optional[npt.ArrayLike] = None
	) -> None:
		"""Compute smoothing lengths or tensors for SPH calculations.

		Args:
			num_neighbors: Number of nearest neighbors used for smoothing length estimation.
			mode: Type of smoothing to compute. Options are:
				- 'adaptive': Compute rectangular particle extent as half-distance to Nth neighbor.
				- 'isotropic': Compute scalar smoothing length as half-distance to Nth neighbor.
				- 'anisotropic': Compute smoothing tensor using covariance-based method.
			query_pos: Optional array of shape (M, D) with positions where smoothing is evaluated.
				If None, uses particle positions from the instance.

		Returns:
			None. Results are stored as instance attributes:
				- `hsm`: Smoothing lengths (adaptive/isotropic modes).
				- `h_tensor`, `h_eigvals`, `h_eigvecs`: Smoothing tensor, eigenvalues and -vectors (anisotropic mode).
				- `nn_inds`: Nearest neighbor indices for all modes.
				- `nn_dists`: Nearest neighbor distances (isotropic mode only).
		"""
		
		self.num_neighbors = num_neighbors
		self.mode = mode

		if not hasattr(self, 'tree'):
			if self.verbose:
				print("Building kd-tree from particle positions")
			self.tree = build_kdtree(self.pos, boxsize=self.boxsize)

		if self.verbose:
			print(f"Computing smoothing lengths/tensors using mode='{mode}' with num_neighbors={num_neighbors}")

		# construct kwargs
		kwargs = {
			'tree': self.tree,
			'query_pos': query_pos,
			'num_neighbors': num_neighbors,
			'boxsize': self.boxsize
		}

		if mode == 'adaptive':
			self.hsm, self.nn_inds = compute_pcellsize_half(**kwargs)

		if mode == 'isotropic':
			self.hsm, self.nn_inds, self.nn_dists = compute_hsm(**kwargs)
		
		elif mode == 'anisotropic':
			self.h_tensor, self.h_eigvals, self.h_eigvecs, self.nn_inds, self.nn_dists, self.rel_coords = compute_hsm_tensor(**kwargs, masses=self.mass)	
				
		else:
			raise AssertionError(f"'mode' must be either, 'adaptive', 'isotropic' or 'anisotropic' but found {mode}")


	def compute_density(self, kernel_name: str) -> None:
		"""Compute particle densities using SPH kernels.

		Supports both isotropic and anisotropic smoothing modes. Requires that
		`compute_smoothing_lengths` has been called first.

		Args:
			kernel_name: Name of the SPH kernel to use for density estimation.

		Returns:
			None. Result is stored in the instance attribute `density`.
		"""
		assert self.mode in ['isotropic', 'anisotropic'], "`mode` must be one of ['isotropic', 'anisotropic'] to compute density"
		self.kernel = Kernel(kernel_name, dim=self.dim)
		
		# if hsm or h_tensor not computed yet, raise error
		if not hasattr(self, 'hsm') and not hasattr(self, 'h_tensor'):
			raise AttributeError("You must first compute smoothing lengths or tensors using the 'compute_smoothing_lengths' method before computing density")

		# Build kernel arguments based on mode
		kwargs = {}
		if self.mode == 'isotropic':
			kwargs['r_ij'] = self.nn_dists
			kwargs['h'] = self.hsm
		elif self.mode == 'anisotropic':
			kwargs['H'] = self.h_tensor

		# Kernel evaluation and density computation
		w = self.kernel.evaluate_kernel(**kwargs)
		self.density = np.sum(self.mass[self.nn_inds] * w, axis=1)
	

	def interpolate_fields(
		self,
		fields: npt.ArrayLike,
		query_positions: npt.ArrayLike,
		kernel_name: Optional[str] = None,
		compute_gradients: bool = False
	) -> npt.NDArray[np.floating]:
		"""Interpolate particle fields to arbitrary query positions using SPH.

		Requires that `compute_smoothing_lengths` and `compute_density` have been 
		called first. Automatically computes density if not already available.

		Args:
			fields: Array of shape (N, num_fields) or (N,) with particle field data.
			query_positions: Array of shape (M, D) with positions where fields are interpolated.
			kernel_name: Name of the SPH kernel to use. If None, uses the previously set kernel.
			compute_gradients: If True, compute field gradients at query positions instead
				of field values.

		Returns:
			Array of interpolated field values or gradients:
				- If compute_gradients=False: shape (M, num_fields) with interpolated values.
				- If compute_gradients=True: shape (M, num_fields, D) with interpolated gradients.

		Raises:
			ValueError: If kernel_name is not implemented.
			AssertionError: If fields length does not match number of particles.
		"""
		
		if not isinstance(fields, np.ndarray):
			fields = np.asarray(fields)
		
		if fields.shape[0] != self.pos.shape[0]:
			raise AssertionError(f"'fields' array must have the same length as positions array but found position length = {self.pos.shape[0]} and fields length={fields.shape[0]}")
		
		if fields.ndim == 1:
			fields = fields[:, np.newaxis]  # make it (N, 1)
		
		if not hasattr(self, 'density'):
			if self.verbose:
				print("Particle density has not been computed yet, computing now")
			self.compute_density(kernel_name)

		if kernel_name is None:
			kernel = self.kernel
		else:
			try:
				kernel = Kernel(kernel_name, dim=self.dim)
			except AttributeError:
				raise ValueError(f"Kernel '{kernel_name}' is not implemented")
		
		return self._interpolate_fields(
			self.tree,
			self.pos,
			self.mass,
			self.density,
			fields, 
			kernel,
			num_neighbors=self.num_neighbors,
			query_positions=query_positions,
			mode=self.mode,
			compute_gradients=compute_gradients,
		)


	def interpolate_grad_fields(
		self,
		query_positions: npt.ArrayLike,
		kernel_name: Optional[str] = None
	) -> npt.NDArray[np.floating]:
		"""Compute gradients of particle fields at arbitrary query positions using SPH.

		Convenience wrapper around `interpolate_fields` with `compute_gradients=True`.

		Args:
			query_positions: Array of shape (M, D) with positions where gradients are evaluated.
			kernel_name: Name of the SPH kernel to use. If None, uses the previously set kernel.

		Returns:
			Array of shape (M, num_fields, D) with interpolated field gradients.
		"""
		return self.interpolate_fields(query_positions, kernel_name, compute_gradients=True)
			

	def _interpolate_fields(
		self,
		tree: Any,
		positions: npt.NDArray[np.floating],
		masses: npt.NDArray[np.floating],
		density: npt.NDArray[np.floating],
		fields: npt.NDArray[np.floating],
		kernel: Any,
		num_neighbors: int,
		query_positions: npt.ArrayLike,
		mode: Literal['isotropic', 'anisotropic'],
		compute_gradients: bool = False
	) -> npt.NDArray[np.floating]:
		"""Interpolate or compute gradients of particle fields at query positions.

		Internal helper for SPH field interpolation supporting both isotropic and 
		anisotropic smoothing kernels.

		Args:
			tree: Spatial index structure (cKDTree) for nearest neighbor queries.
			positions: Particle positions with shape (N, D).
			masses: Particle masses with shape (N,).
			density: Particle densities with shape (N,). Used for SPH weighting.
			fields: Particle field values with shape (N, num_fields).
			kernel: Kernel instance with evaluate_kernel() and evaluate_gradient() methods.
			num_neighbors: Number of nearest neighbors to use for interpolation.
			query_positions: Positions where fields are evaluated, shape (M, D).
			mode: Smoothing mode - 'isotropic' uses scalar smoothing lengths, 
				'anisotropic' uses smoothing tensors.
			compute_gradients: If True, compute field gradients via kernel.evaluate_gradient().
				If False, compute field values via kernel.evaluate_kernel().

		Returns:
			- If compute_gradients=False: array of shape (M, num_fields) with interpolated field values.
			- If compute_gradients=True: array of shape (M, num_fields, D) with interpolated gradients.
		"""
		
		if mode == 'isotropic':
			# Compute smoothing lengths at query positions
			hsm, nn_inds, nn_dists = compute_hsm(
				tree, 
				query_positions, 
				k=num_neighbors,
			)
			if compute_gradients:
				# For gradients, need relative coordinate vectors
				rel_coords = coordinate_difference_with_pbc(
					positions[nn_inds],
					query_positions[:, np.newaxis, :],
					self.boxsize,
				)
				kernel_kwargs = {'r_ij_vec': rel_coords, 'h': hsm}
			else:
				kernel_kwargs = {'r_ij': nn_dists, 'h': hsm}

		elif mode == 'anisotropic':
			# Compute smoothing tensors at query positions
			h_tensor, _, _, nn_inds, _, rel_coords = compute_hsm_tensor(
				tree,
				masses=masses,
				num_neighbors=num_neighbors,
				query_positions=query_positions,
			)
			kernel_key = 'r_ij_vec' if compute_gradients else 'r_ij'
			kernel_kwargs = {kernel_key: rel_coords, 'H': h_tensor}

		else:
			raise ValueError(f"Unsupported interpolation mode '{mode}'")

		# Unified weight computation and kernel evaluation
		weights = masses[nn_inds] / (density[nn_inds] + 1e-8)
		fields_ = fields[nn_inds]
		
		if compute_gradients:
			w = kernel.evaluate_gradient(**kernel_kwargs)
			result = np.einsum('mkf,mkd,mk->mfd', fields_, w, weights)
		else:
			w = kernel.evaluate_kernel(**kernel_kwargs)
			result = np.einsum('mkf,mk,mk->mf', fields_, w, weights)
		
		return result


	def deposit_to_grid(self,
					 fields: np.ndarray,
					 averaged: Sequence[bool],
					 gridnums: Union[int, Sequence[int]],
					 method: str,
					 extent: Optional[Sequence[Sequence[float]]] = None,
					 plane_projection: Optional[str] = None,
					 return_weights: bool = False,
					 kernel: str = 'quintic',
					 integration: str = 'midpoint',
					 use_python: bool = False,
					 use_openmp: bool = True,
					 omp_threads: Optional[int] = None,
					 ):
		"""Deposit particle fields onto a structured grid using the requested scheme.

		:param fields: Particle quantities with shape (num_particles, num_fields).
		:type fields: np.ndarray
		:param averaged: Flags indicating which fields should be averaged by the weight grid.
		:type averaged: Sequence[bool]
		:param gridnums: Number of grid cells per spatial dimension (scalar or per-axis sequence).
		:type gridnums: Union[int, Sequence[int]]
		:param method: Deposition method (e.g. "cic", "tsc", "isotropic").
		:type method: str
		:param extent: Optional sequence of [min, max] pairs per axis defining the region to deposit.
		:type extent: Optional[Sequence[Sequence[float]]]
		:param plane_projection: Optional plane spec ("(0, 1)", "(0, 2)", "(1, 2)") to project 3D tensors to 2D before deposition.
		:type plane_projection: Optional[str]
		:param return_weights: Whether to also return the grid of accumulated weights.
		:type return_weights: bool
		:param kernel: SPH kernel name used for isotropic/anisotropic methods.
		:type kernel: str
		:param integration: Quadrature rule for SPH kernel integration.
		:type integration: str
		:param use_python: Use Python backend instead of C++ backend (this is mainly for debugging purposes).
		:type use_python: bool
		:param use_openmp: Enable OpenMP parallelism for the C++ backend. Ignored when ``use_python=True``.
		:type use_openmp: bool
		:param omp_threads: Optional positive integer overriding the number of OpenMP threads. ``None`` keeps the runtime default.
		:type omp_threads: Optional[int]
		"""

		if omp_threads is not None:
			if not isinstance(omp_threads, (int, np.integer)):
				raise TypeError("'omp_threads' must be an integer if provided")
			if omp_threads <= 0:
				raise ValueError("'omp_threads' must be > 0 when specified")
			omp_threads_value = int(omp_threads)
		else:
			omp_threads_value = 0

		# deposition grid dimension is coordinate dimension unless projecting to 2D plane
		deposition_dim = self.pos.shape[1] if plane_projection is None else 2


		# check that plane_projection is only specified for 3D point clouds
		if plane_projection and self.dim != 3:
			raise ValueError(f"Plane projection can only be specified for 3D particle positions, but found positions with shape {self.pos.shape}")


		# check and typecast the 'fields' parameter
		fields = np.asarray(fields, dtype=np.float32)
		if fields.shape[0] != self.pos.shape[0]:
			raise ValueError(
				f"'fields' array length ({fields.shape[0]}) must match number of particles ({self.pos.shape[0]})"
			)


		# check that either smoothing lengths or tensors have been computed if necessary
		if method in ['isotropic', 'anisotropic'] and not hasattr(self, 'hsm') and not hasattr(self, 'h_tensor'):
			raise AttributeError("You must first compute smoothing lengths or tensors using the 'compute_smoothing_lengths' method before depositing to grid")
		

		# check and typecast the 'averaged' parameter
		averaged = list(averaged) if isinstance(averaged, (list, tuple)) else [averaged]


		# check and typecast the 'gridnums' parameter
		if extent is None: # assume periodic box and deposit over [0, boxsize]
			if self.boxsize is None:
				raise ValueError("Either 'boxsize' must be set on the class or 'extent' must be provided")
			boxsize_array = np.asarray(self.boxsize, dtype=np.float32)
			if boxsize_array.ndim == 0:
				boxsize_array = np.repeat(boxsize_array, deposition_dim)
			if boxsize_array.size != deposition_dim:
				raise ValueError(f"Boxsize must define {deposition_dim} extents, received {boxsize_array.size}")
			domain_min = np.zeros(deposition_dim, dtype=np.float32)
			domain_max = boxsize_array.astype(np.float32)
			periodic_temp = np.full(deposition_dim, bool(self.periodic), dtype=bool)
		else:
			extent_array = np.asarray(extent, dtype=np.float32)
			if extent_array.ndim != 2 or extent_array.shape[0] != deposition_dim or extent_array.shape[1] != 2:
				raise ValueError(
					f"Extent must be shaped ({deposition_dim}, 2); received {extent_array.shape}"
				)
			domain_min = extent_array[:, 0]
			domain_max = extent_array[:, 1]
			if np.any(domain_max <= domain_min):
				raise ValueError("Each extent axis must have max > min")
			periodic_temp = np.zeros(deposition_dim, dtype=bool)
			if self.periodic and self.boxsize is not None:
				boxsize_array = np.asarray(self.boxsize, dtype=np.float32)
				if boxsize_array.ndim == 0:
					boxsize_array = np.repeat(boxsize_array, deposition_dim)
				if boxsize_array.size != deposition_dim:
					raise ValueError(f"Boxsize must define {deposition_dim} extents, received {boxsize_array.size}")
				span = domain_max - domain_min
				periodic_temp = np.isclose(span, boxsize_array, rtol=1e-6, atol=1e-6) & bool(self.periodic)


		# check and typecast the 'periodic' parameter
		periodic_temp = np.asarray(periodic_temp, dtype=bool)
		if periodic_temp.ndim == 0:
			periodic_temp = np.repeat(periodic_temp, deposition_dim)
		if periodic_temp.size != deposition_dim:
			raise ValueError(f"Expected {deposition_dim} periodic flags, received {periodic_temp.size}")
		periodic_temp = np.ascontiguousarray(periodic_temp, dtype=np.bool_)


		# check and typecast the 'gridnums' parameter
		gridnums_array = np.asarray(gridnums, dtype=np.int32)
		if gridnums_array.ndim == 0:
			gridnums_array = np.repeat(gridnums_array, deposition_dim)
		if gridnums_array.size != deposition_dim:
			raise ValueError(f"Expected {deposition_dim} gridnum numbers, received {gridnums_array.size}")
		gridnums_array = np.ascontiguousarray(gridnums_array, dtype=np.int32)


		# shift the particle positions to be within [0, boxsize_i] for each axis i
		positions = self.pos
		domain_lengths = domain_max - domain_min
		masks = []
		for axis in range(deposition_dim):
			axis_min = domain_min[axis]
			axis_max = domain_max[axis]
			masks.append((positions[:, axis] >= axis_min) & (positions[:, axis] <= axis_max))
		final_mask = np.logical_and.reduce(masks) if masks else np.ones(positions.shape[0], dtype=bool)

		pos_temp = positions[final_mask]
		pos_temp = pos_temp - domain_min
		fields_temp = fields[final_mask]
		hsm_temp = self.hsm[final_mask] if hasattr(self, 'hsm') else None

		h_eigvals_temp = None
		h_eigvecs_temp = None
		if hasattr(self, 'h_tensor'):
			if deposition_dim == 2 and self.dim == 3:
				self.h_tensor_2d, self.h_eigvals_2d, self.h_eigvecs_2d = project_hsm_tensor_to_2d(
					self.h_tensor,
					plane=plane_projection
				)
				eigvals_source = self.h_eigvals_2d
				eigvecs_source = self.h_eigvecs_2d
			else:
				eigvals_source = self.h_eigvals
				eigvecs_source = self.h_eigvecs
			h_eigvals_temp = eigvals_source[final_mask]
			h_eigvecs_temp = eigvecs_source[final_mask]

		if self.verbose and not use_python:
			if use_openmp:
				thread_desc = "runtime default" if omp_threads_value == 0 else str(omp_threads_value)
				print(f"[sph_lib] OpenMP threads: {thread_desc}")
			else:
				print("[sph_lib] OpenMP disabled for this deposition call")

		res = self._deposit_to_grid(
					positions=pos_temp,
					quantities=fields_temp,
					averaged=averaged,
					gridnums=gridnums_array,
					boxsizes=domain_lengths,
					periodic=periodic_temp,
					hsm=hsm_temp,
					hmat_eigvecs=h_eigvecs_temp,
					hmat_eigvals=h_eigvals_temp,
					return_weights=return_weights,
					method=method,
					use_python=use_python,
					kernel=kernel,
					integration=integration,
					use_openmp=use_openmp,
					omp_threads=omp_threads_value,
					)
		return res


	def _deposit_to_grid(self,
		positions: np.ndarray,
		quantities: np.ndarray,
		averaged: Sequence[bool],
		gridnums: Union[int, Sequence[int]],
		boxsizes: np.ndarray,
		periodic: Union[int, Sequence[bool]],
		*,
		method: str,
		hsm: Optional[np.ndarray],
		hmat_eigvecs: Optional[np.ndarray],
		hmat_eigvals: Optional[np.ndarray],
		return_weights: bool,
		use_python: bool,
		kernel: str,
		integration: str,
		use_openmp: bool,
		omp_threads: Optional[int],
	) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
		"""Dispatch particle-to-grid deposition to the selected backend.

		Parameters
		----------
		positions
			Particle coordinates with shape (N, dim).
		quantities
			Fields to deposit with shape (N, num_fields).
		averaged
			Flags indicating which fields should be normalized by weights.
		gridnums
			Number of cells per dimension (scalar replicated across axes is allowed).
		boxsizes
			Domain extents for each axis expressed as maximum lengths.
		periodic
			Per-axis flags.
		method
			Name of the deposition method (e.g., "ngp", "cic", "tsc").
		use_openmp
			Enable OpenMP parallelism for the compiled backend branch.
		omp_threads
			Optional positive integer overriding the OpenMP thread count (0 keeps the runtime default).

		Returns
		-------
		Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
			Deposited fields, optionally accompanied by weights.
		"""
		dim = positions.shape[-1]

		# select the deposition function based on the method, dim and use_python
		namespace = pyfunc if use_python else cppfunc
		func = getattr(namespace, f"{method}_{dim}d")
		
		if self.verbose:
			print(f"Using deposition function: {func.__name__}")

		if "adaptive" in method:
			args = (
				positions,
				quantities,
				boxsizes,
				gridnums,
				periodic,
				hsm,
			)

		elif "isotropic" == method:
			args = (
				positions,
				quantities,
				boxsizes,
				gridnums,
				periodic,
				hsm,
				kernel,
				integration,
			)

		elif "anisotropic" == method:
			args = (
				positions,
				quantities,
				boxsizes,
				gridnums,
				periodic,
				hmat_eigvecs,
				hmat_eigvals,
				kernel,
				integration,
			)

		else:
			args = (
				positions,
				quantities,
				boxsizes,
				gridnums,
				periodic,
			)

		if use_python:
			fields, weights = func(*args)
		else:
			threads_arg = 0 if omp_threads is None else int(omp_threads)
			fields, weights = func(*args, use_openmp, threads_arg)

		# divide averaged fields by weight
		for i in range(len(averaged)):
			if averaged[i] == True:
				fields[..., i] /= (weights + 1e-10)

		if return_weights:
			return fields, weights
		else:
			return fields



