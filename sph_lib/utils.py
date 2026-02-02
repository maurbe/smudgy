import numpy as np
from scipy import spatial


def build_kdtree(points, boxsize=None):
	"""Construct a cKDTree with optional per-dimension periodic box sizes."""
	return spatial.cKDTree(points, boxsize=boxsize)

def query_kdtree(tree, points, k):
	"""Query a cKDTree for the k nearest neighbors of given points."""
	return tree.query(points, k=k, workers=-1)

def shift_particle_positions(pos):
    shifted_pos = pos - pos.min(axis=0)
    return shifted_pos

def coordinate_difference_with_pbc(x, y, boxsize):
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


def compute_pcellsize_half(tree, pos, num_neighbors):
	# [0]=distances; [:, -1] we only need the last NN
	# this follows the same "definition" of the smoothing length as 0.5 * distance to Nth nearest neighbor
	nn_dists, nn_inds = tree.query(x=pos, k=num_neighbors, workers=-1)
	nn_dists = nn_dists[:, -1] * 0.5
	nn_dists = nn_dists.astype('float32')
	return nn_dists, nn_inds


def compute_hsm(tree, pos, num_neighbors, query_pos=None):
	"""
	Estimate smoothing length as half the distance to the Nth nearest neighbor.

	Args:
		tree:     cKDTree built from particle positions
		pos:      (N, D) particle positions
		num_neighbors:       int, number of neighbors
		query_pos: (M, D) positions where the smoothing length is evaluated
	
	Returns:
		hsm:      (N,) estimated smoothing lengths
		nn_dists: (N, num_neighbors) distances to nearest neighbors
		nn_inds:  (N, num_neighbors) indices of nearest neighbors
	"""
	if query_pos is None:
		query_pos = pos
	nn_dists, nn_inds = query_kdtree(tree, query_pos, k=num_neighbors)
	hsm = nn_dists[:, -1] * 0.5
	return hsm, nn_dists, nn_inds


def compute_hsm_tensor(tree, pos, masses, num_neighbors, query_pos=None):
	"""
    Computes the smoothing tensor H for each particle using the covariance-based method
    from Marinho (2021), generalized for arbitrary dimension (D=2 or D=3).

    Args:
		tree:     cKDTree built from particle positions
        pos:      (N, D) particle positions
        masses:   (N,) particle masses (1D array)
        num_neighbors:       int, number of neighbors
        query_pos: (M, D) positions where the smoothing tensor is evaluated

    Returns:
        H:        (N, D, D) smoothing tensor for each particle
        eigvals:  (N, D) eigenvalues of the covariance matrix (scaled)
        eigvecs:  (N, D, D) eigenvectors of the covariance matrix
		nn_inds:  (N, num_neighbors) indices of nearest neighbors
    """
	
	N, D = pos.shape
	assert D in (2, 3), "Only 2D and 3D supported"

	# Ensure masses is 1D (flattens any extra dimensions from indexing)
	masses = masses.flatten() if masses.ndim > 1 else masses

	# get the boxsize from the tree object
	boxsize = tree.boxsize

	# use compute_hsm() to find the nn_inds
	nn_dists, nn_inds = query_kdtree(tree, pos, k=num_neighbors)
	
	neighbor_coords = pos[nn_inds]
	neighbor_masses = masses[nn_inds]
	
	# here we can switch between particle or coordinate-based query
	if query_pos is None:
		query_pos = pos
	r_jc = neighbor_coords - query_pos[:, np.newaxis, :]
	
	# we have to account for pbc
	r_jc = coordinate_difference_with_pbc(neighbor_coords, query_pos[:, np.newaxis, :], boxsize)
	
	outer = np.einsum('...i, ...j -> ...ij', r_jc, r_jc)
	outer = outer * neighbor_masses[..., np.newaxis, np.newaxis]
	Sigma = np.sum(outer, axis=1) / np.sum(neighbor_masses, axis=1)[..., np.newaxis, np.newaxis]

	# eigvecs are returned normalized
	# eigvecs are the same for H
	# eigvals are the sqrt of the ones of Sigma
	eigvals, eigvecs = np.linalg.eigh(Sigma)
	eigvals = np.sqrt(eigvals)
	
	# also compute H = VΛV.T
	Λ = eigvals[..., np.newaxis] * np.eye(query_pos.shape[-1])
	H = np.matmul(np.matmul(eigvecs, Λ), np.transpose(eigvecs, axes=(0, 2, 1)))
	
	return H, eigvals, eigvecs, nn_inds


def project_hsm_tensor_to_2d(hmat, plane):
	# need to project the smoothing ellipses on the cartesian plane
	if plane == (0, 1):
		e1 = [1, 0, 0]
		e2 = [0, 1, 0]
	elif plane == (0, 2):
		e1 = [1, 0, 0]
		e2 = [0, 0, 1]
	elif plane == (1, 2):
		e1 = [0, 1, 0]
		e2 = [0, 0, 1]
	else:
		print("Plane must be either xy, yz or xz.")

	# Compute (P * M_inverse * P_transpose)^-1
	projection_matrix = np.array([e1, e2]).astype('float32')  # (2, 3)
	hmat_inv = np.linalg.inv(hmat)
	hmat_2d = []

	for i in range(len(hmat_inv)):
		# Compute P @ H_inv @ P^T, then invert
		temp = np.dot(projection_matrix, np.dot(hmat_inv[i], projection_matrix.T))  # (2,3) @ (3,3) @ (3,2) = (2,2)
		hmat_2d.append(np.linalg.inv(temp))
	hmat_2d = np.asarray(hmat_2d)

	# compute the new eigenvectors (2d)
	evals, evecs = np.linalg.eigh(hmat_2d)
	return hmat_2d, evals, evecs
