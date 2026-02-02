from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy import spatial


FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.int_]
BoxInput = float | Sequence[float] | npt.ArrayLike


def build_kdtree(
	points: npt.ArrayLike,
	boxsize: Optional[BoxInput] = None
) -> spatial.cKDTree:
	"""Construct a cKDTree with optional per-dimension periodic box sizes.

	Args:
		points: Array of shape (N, D) with particle coordinates.
		boxsize: Scalar or (D,) array defining periodic box lengths, or None.

	Returns:
		scipy.spatial.cKDTree built from ``points``.
	"""
	return spatial.cKDTree(points, boxsize=boxsize)


def query_kdtree(
	tree: spatial.cKDTree,
	points: npt.ArrayLike,
	k: int
) -> Tuple[FloatArray, IntArray]:
	"""Query a cKDTree for the k nearest neighbors of given points.

	Args:
		tree: cKDTree instance to query.
		points: Array of shape (M, D) with query coordinates.
		k: Number of nearest neighbors to return.

	Returns:
		Tuple of ``(distances, indices)`` from ``cKDTree.query``.
	"""
	return tree.query(points, k=k, workers=-1)


def shift_particle_positions(pos: npt.ArrayLike) -> FloatArray:
	"""Shift particle positions so the minimum per-axis value is at zero.

	Args:
		pos: Array of shape (N, D) with particle coordinates.

	Returns:
		Array of shifted coordinates with the same shape as ``pos``.
	"""
	pos_array = np.asarray(pos)
	return pos_array - pos_array.min(axis=0)


def coordinate_difference_with_pbc(
	x: npt.ArrayLike,
	y: npt.ArrayLike,
	boxsize: BoxInput
) -> FloatArray:
	"""Compute coordinate differences with periodic boundary conditions.

	Args:
		x: Array-like coordinates.
		y: Array-like coordinates to subtract from ``x``.
		boxsize: Scalar or (D,) array defining periodic box lengths.

	Returns:
		Array of coordinate differences wrapped into the range
		$[-0.5 * boxsize, 0.5 * boxsize)$ per dimension.
	"""
	diff = np.asarray(x) - np.asarray(y)
	box_arr = np.asarray(boxsize)

	if box_arr.ndim == 0:
		half_box = 0.5 * box_arr
		return (diff + half_box) % box_arr - half_box

	if box_arr.ndim != 1:
		raise ValueError("'boxsize' must be a scalar or 1D array")

	if diff.ndim == 0:
		raise ValueError("Vector 'boxsize' requires vector inputs for x and y")

	if diff.shape[-1] != box_arr.shape[0]:
		raise ValueError("Dimension mismatch between coordinates and 'boxsize'")

	half_box = 0.5 * box_arr
	return (diff + half_box) % box_arr - half_box


def compute_pcellsize_half(
	tree: spatial.cKDTree,
	query_pos: npt.ArrayLike,
	num_neighbors: int
) -> Tuple[FloatArray, IntArray]:
	"""Estimate rectangular particle extent as half-distance to the Nth nearest neighbor.

	Args:
		tree: cKDTree built from particle positions.
		query_pos: Array of shape (N, D) with particle coordinates.
		num_neighbors: Number of neighbors used for the estimate.

	Returns:
		Tuple of ``(h_cellsize, nn_inds)`` where ``h_cellsize`` has shape (N,)
		and contains half the distance to the Nth neighbor, and ``nn_inds``
		has shape (N, num_neighbors).
	"""
	# this follows the same "definition" of the smoothing length as 0.5 * distance to Nth nearest neighbor
	nn_dists, nn_inds = query_kdtree(tree, query_pos, k=num_neighbors)
	h_cellsize = nn_dists[:, -1] * 0.5
	return h_cellsize, nn_inds


def compute_hsm(
	tree: spatial.cKDTree,
	query_pos: npt.ArrayLike,
	num_neighbors: int
) -> Tuple[FloatArray, IntArray, FloatArray]:
	"""Estimate smoothing length as half the distance to the Nth nearest neighbor.

	Args:
		tree: cKDTree built from particle positions.
		query_pos: Array of shape (M, D) with positions where smoothing length is evaluated.
		num_neighbors: Number of neighbors used for the estimate.

	Returns:
		Tuple of ``(hsm, nn_inds, nn_dists)`` where ``hsm`` has shape (M,),
		``nn_dists`` has shape (M, num_neighbors), and ``nn_inds`` has shape
		(M, num_neighbors).
	"""
	nn_dists, nn_inds = query_kdtree(tree, query_pos, k=num_neighbors)
	hsm = nn_dists[:, -1] * 0.5
	return hsm, nn_inds, nn_dists


def compute_hsm_tensor(
	tree: spatial.cKDTree,
	masses: npt.ArrayLike,
	num_neighbors: int,
	query_pos: Optional[npt.ArrayLike] = None
) -> Tuple[FloatArray, FloatArray, FloatArray, IntArray, FloatArray]:
	"""Compute anisotropic smoothing tensor using covariance-based method.

	Implements the method from Marinho (2021), generalized for 2D and 3D.

	Args:
		tree: cKDTree built from particle positions.
		masses: Array of shape (N,) with particle masses.
		num_neighbors: Number of neighbors used for covariance estimation.
		query_pos: Array of shape (M, D) where smoothing tensor is evaluated.
			If None, uses particle positions from the tree.

	Returns:
		Tuple of ``(H, eigvals, eigvecs, nn_inds, nn_dists, rel_coords)`` where ``H`` has shape (M, D, D),
		``eigvals`` has shape (M, D), ``eigvecs`` has shape (M, D, D),
		``nn_inds`` has shape (M, num_neighbors) and ``nn_dists`` has shape (M, num_neighbors) and ``rel_coords`` has shape (M, num_neighbors, D).
	"""
	
	# Get particle positions and boxsize from the tree object
	pos = tree.data
	boxsize = tree.boxsize
	D = pos.shape[-1]
	
	if D not in (2, 3):
		raise ValueError("Only 2D and 3D positions are supported for anisotropic smoothing tensors.")

	# Use particle positions if query_pos not provided
	if query_pos is None:
		query_pos = pos
	
	# Ensure masses is 1D
	masses = masses.flatten() if masses.ndim > 1 else masses

	# Find nearest neighbors
	nn_dists, nn_inds = query_kdtree(tree, query_pos, k=num_neighbors)
	neighbor_coords = pos[nn_inds]
	neighbor_masses = masses[nn_inds]
	
	# Account for periodic boundary conditions
	r_jc = coordinate_difference_with_pbc(neighbor_coords, query_pos[:, np.newaxis, :], boxsize)
	
	# Compute mass-weighted covariance matrix
	outer = np.einsum('...i, ...j -> ...ij', r_jc, r_jc)
	outer = outer * neighbor_masses[..., np.newaxis, np.newaxis]
	Sigma = np.sum(outer, axis=1) / np.sum(neighbor_masses, axis=1)[..., np.newaxis, np.newaxis]

	# Compute eigendecomposition and construct smoothing tensor H = VΛV^T
	eigvals, eigvecs = np.linalg.eigh(Sigma)
	eigvals = np.sqrt(eigvals)
	Λ = eigvals[..., np.newaxis] * np.eye(D)
	H = np.matmul(np.matmul(eigvecs, Λ), np.transpose(eigvecs, axes=(0, 2, 1)))

	# compute the relative coordinate vectors from neighbors to query positions
	rel_coords = coordinate_difference_with_pbc(neighbor_coords, query_pos[:, np.newaxis, :], boxsize)
	
	return H, eigvals, eigvecs, nn_inds, nn_dists, rel_coords


def project_hsm_tensor_to_2d(
	h_tensor: npt.ArrayLike,
	plane: Optional[str] = None,
	basis: Optional[Tuple[Sequence[float], Sequence[float]]] = None
) -> Tuple[FloatArray, FloatArray, FloatArray]:
	"""Project 3D smoothing tensors onto a 2D plane.

	Args:
		h_tensor: Array of shape (N, 3, 3) with 3D smoothing tensors.
		plane: String specifying projection plane: 'xy', 'xz', or 'yz'.
			Mutually exclusive with ``basis``.
		basis: 2-tuple of basis vectors (e1, e2) spanning the projection plane.
			Each vector should be array-like of length 3.
			Mutually exclusive with ``plane``.

	Returns:
		Tuple of ``(h_tensor_2d, eigvals, eigvecs)`` where ``h_tensor_2d`` has shape (N, 2, 2),
		``eigvals`` has shape (N, 2), and ``eigvecs`` has shape (N, 2, 2).
	"""
	# Validate inputs
	if plane is None and basis is None:
		raise ValueError("Either 'plane' or 'basis' must be provided")
	if plane is not None and basis is not None:
		raise ValueError("'plane' and 'basis' are mutually exclusive")
	
	# Define projection basis vectors
	if basis is not None:
		if len(basis) != 2:
			raise ValueError("'basis' must be a 2-tuple of vectors")
		e1, e2 = basis
	else:
		if plane == 'xy':
			e1, e2 = [1, 0, 0], [0, 1, 0]
		elif plane == 'xz':
			e1, e2 = [1, 0, 0], [0, 0, 1]
		elif plane == 'yz':
			e1, e2 = [0, 1, 0], [0, 0, 1]
		else:
			raise ValueError("'plane' must be one of 'xy', 'xz', or 'yz'")

	# Compute projected tensors: (P @ H^-1 @ P^T)^-1
	projection_matrix = np.array([e1, e2], dtype='float32')  # (2, 3)
	h_tensor_inv = np.linalg.inv(h_tensor)  # (N, 3, 3)
	
	# Vectorized computation: P @ H_inv @ P^T for all particles
	temp = np.einsum('ij,njk,lk->nil', projection_matrix, h_tensor_inv, projection_matrix)
	h_tensor_2d = np.linalg.inv(temp)  # (N, 2, 2)
	
	# Compute eigendecomposition of 2D tensors
	eigvals, eigvecs = np.linalg.eigh(h_tensor_2d)
	return h_tensor_2d, eigvals, eigvecs