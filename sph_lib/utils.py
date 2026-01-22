import numpy as np
import scipy.spatial as spatial

def create_grid_1d(nx, boxsize):

	Δx = boxsize / nx
	x = np.linspace(Δx / 2.0, boxsize - Δx/2.0, nx)
	x = x[:, np.newaxis]
	return x.astype('float32')


def create_grid_2d(nx, ny, boxsize):

	Δx = boxsize / nx
	Δy = boxsize / ny

	x = np.linspace(Δx / 2.0, boxsize - Δx/2.0, nx)
	y = np.linspace(Δy / 2.0, boxsize - Δy/2.0, ny)

	xx, yy = np.meshgrid(x, y, indexing='ij')
	grid_positions = np.stack((xx.ravel(), 
							   yy.ravel()), axis=-1).astype('float32')
	return grid_positions


def create_grid_3d(nx, ny, nz, boxsize):

	Δx = boxsize / nx
	Δy = boxsize / ny
	Δz = boxsize / nz

	x = np.linspace(Δx / 2.0, boxsize - Δx/2.0, nx)
	y = np.linspace(Δy / 2.0, boxsize - Δy/2.0, ny)
	z = np.linspace(Δz / 2.0, boxsize - Δz/2.0, nz)

	xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
	grid_positions = np.stack((xx.ravel(), 
							   yy.ravel(), 
							   zz.ravel()), 
							  axis=-1).astype('float32')
	return grid_positions


def shift_particle_positions(pos):
    shifted_pos = pos - pos.min(axis=0)
    return shifted_pos


def coordinate_difference_with_pbc(x, y, boxsize):
    return (x - y + 0.5 * boxsize) % boxsize - 0.5 * boxsize


def compute_pcellsize_half(pos, num_neighbors, boxsize=None):
	if boxsize is None:
		tree = spatial.KDTree(pos)
	else:
		tree = spatial.cKDTree(pos, boxsize=boxsize + 1e-8)
	# [0]=distances; [:, -1] we only need the last NN
	# this follows the same "definition" of the smoothing length as 0.5 * distance to Nth nearest neighbor
	nn_dists, nn_inds = tree.query(x=pos, k=num_neighbors, workers=-1)
	nn_dists = nn_dists[:, -1] * 0.5
	nn_dists = nn_dists.astype('float32')
	return nn_dists, nn_inds, tree


def compute_hsm(pos, num_neighbors, boxsize=None):
	"""
	Estimate smoothing length as half the distance to the Nth nearest neighbor.
	"""
	if boxsize is None:
		tree = spatial.KDTree(pos)
	else:
		tree = spatial.cKDTree(pos, boxsize=boxsize)

	nn_dists, nn_inds = tree.query(pos, k=num_neighbors, workers=-1)
	hsm = nn_dists[:, -1] * 0.5
	return hsm, nn_dists, nn_inds, tree

"""
def compute_hsm_tensor_OLD(pos, masses, num_neighbors, boxsize=None):
    
    N, D = pos.shape
    assert D in (2, 3), "Only 2D and 3D supported"
    
    # Ensure masses is 1D (flattens any extra dimensions from indexing)
    masses = masses.flatten() if masses.ndim > 1 else masses

    # Step 1: Find neighbors
    _, _, nn_inds, _ = compute_hsm(pos, num_neighbors, boxsize)
    neighbor_masses = masses[nn_inds]          # (N, k)
    neighbor_coords = pos[nn_inds]             # (N, k, D)

    # Step 2: Handle periodic boundaries
    if boxsize is None:
        r_jc = neighbor_coords - pos[:, np.newaxis, :]   # (N, k, D)
    else:
        r_jc = coordinate_difference_with_pbc(neighbor_coords, pos[:, np.newaxis, :], boxsize)
    neighbor_coords = r_jc + pos[:, np.newaxis, :]       # (N, k, D)

    # Step 3: Center of mass of each cluster
    # neighbor_masses: (N, k) → reshape to (N, k, 1) for broadcasting with neighbor_coords (N, k, D)
    total_mass = np.sum(neighbor_masses, axis=1)   # (N,)
    weighted_coords = neighbor_coords * neighbor_masses[:, :, np.newaxis]  # (N, k, D) * (N, k, 1)
    rc = np.sum(weighted_coords, axis=1) / total_mass[:, np.newaxis]  # (N, D)

    # Displacement vectors from center
    deltas = neighbor_coords - rc[:, np.newaxis, :]   # (N, k, D)

    # Step 4: Mass-weighted covariance matrix Σ
    outer = np.einsum('...i,...j->...ij', deltas, deltas)   # (N, k, D, D)
    # Scale by masses: reshape masses from (N, k) to (N, k, 1, 1) for broadcasting
    outer = outer * neighbor_masses[:, :, np.newaxis, np.newaxis]  # (N, k, D, D) * (N, k, 1, 1)
    Sigma = np.sum(outer, axis=1) / total_mass[:, np.newaxis, np.newaxis]   # (N, D, D)

    # Step 5: Eigen-decomposition of Σ
    eigvals, eigvecs = np.linalg.eigh(Sigma)     # eigvals: (N, D), eigvecs: (N, D, D)
    sqrt_eigvals = np.sqrt(eigvals)              # (N, D)

    # Step 6: Compute ζ_max (max Mahalanobis distance)
    inv_Sigma = np.linalg.inv(Sigma)   # (N, D, D)
    mahal_sq = np.einsum('nik,nkl,nil->ni', deltas, inv_Sigma, deltas)   # (N, k)
    zeta_max = np.sqrt(np.max(mahal_sq, axis=1))   # (N,)

    # Step 7: Construct H = ζ_max * V * sqrt(Λ) * V.T
    Λ = np.zeros_like(Sigma)  # (N, D, D)
    for i in range(D):
        Λ[:, i, i] = sqrt_eigvals[:, i]

    H_raw = np.matmul(eigvecs, Λ)                         # (N, D, D)
    H = np.matmul(H_raw, eigvecs.transpose(0, 2, 1))      # (N, D, D)
    H *= zeta_max[:, np.newaxis, np.newaxis]              # Scale by ζ_max

    # Step 8: Rescale eigenvalues
    eigvals_H = zeta_max[:, np.newaxis] * sqrt_eigvals    # (N, D)

    return H, eigvals_H, eigvecs
"""

def compute_hsm_tensor(pos, masses, num_neighbors, boxsize):
	"""
    Computes the smoothing tensor H for each particle using the covariance-based method
    from Marinho (2021), generalized for arbitrary dimension (D=2 or D=3).

    Args:
        pos:      (N, D) particle positions
        masses:   (N,) particle masses (1D array)
        num_neighbors:       int, number of neighbors
        boxsize:  float or None, periodic box size

    Returns:
        H:        (N, D, D) smoothing tensor for each particle
        eigvals:  (N, D) eigenvalues of the covariance matrix (scaled)
        eigvecs:  (N, D, D) eigenvectors of the covariance matrix
    """
	
	N, D = pos.shape
	assert D in (2, 3), "Only 2D and 3D supported"

	# Ensure masses is 1D (flattens any extra dimensions from indexing)
	masses = masses.flatten() if masses.ndim > 1 else masses

	# use compute_hsm() to find the nn_inds
	_, _, nn_inds, _ = compute_hsm(pos, num_neighbors, boxsize)
	
	neighbor_coords = pos[nn_inds]
	neighbor_masses = masses[nn_inds]
	
	# we have to account for pbc
	r_jc = neighbor_coords - pos[:, np.newaxis, :]
	r_jc = np.where(np.abs(r_jc) >= boxsize / 2.0, r_jc - np.sign(r_jc) * boxsize, r_jc)
	
	outer = np.einsum('...i, ...j -> ...ij', r_jc, r_jc)
	outer = outer * neighbor_masses[..., np.newaxis, np.newaxis]
	Sigma = np.sum(outer, axis=1) / np.sum(neighbor_masses, axis=1)[..., np.newaxis, np.newaxis]

	# eigvecs are returned normalized
	# eigvecs are the same for H
	# eigvals are the sqrt of the ones of Sigma
	eigvals, eigvecs = np.linalg.eigh(Sigma)
	eigvals = np.sqrt(eigvals)
	
	# also compute H = VΛV.T
	Λ = eigvals[..., np.newaxis] * np.eye(pos.shape[-1])
	H = np.matmul(np.matmul(eigvecs, Λ), np.transpose(eigvecs, axes=(0, 2, 1)))
	
	return H, eigvals, eigvecs


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





def project_hsm_tensor_to_2d_old(hmat, plane):
	
	# If input is already 2D, just eigendecompose directly
	if hmat.shape[-1] == 2:
		evals, evecs = np.linalg.eigh(hmat)
		return hmat, evals, evecs
	
	# need to project the smoothing ellipses onto cartesian plane
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
		print(f"Plane must be either xy (0, 1), yz (1, 2) or xz (0, 2) but found {plane}")

	# Compute P * M_inverse * P_transpose
	projection_matrix = np.array([e1, e2]).astype('float32')
	hmat_inv = np.linalg.inv(hmat)

	hmat_2d = []
	for i in range(len(hmat_inv)):
		hmat_2d.append(np.linalg.inv(np.dot(projection_matrix, np.dot(hmat_inv[i], projection_matrix.T))))
	hmat_2d = np.asarray(hmat_2d)

	# compute the new eigenvectors (2d)
	evals, evecs = np.linalg.eigh(hmat_2d)
	return hmat_2d, evals, evecs