# +
import torch
import numpy as np
import scipy.spatial as spatial

from .python import functions as pyfunc
from .cpp import functions as cppfunc


def p2g(positions,
                   quantities,
                   averaged,
                   gridnum,
                   extent,
                   periodic=1,
				   num_nn=None,
				   hsm=None,
				   hmat_eigvecs=None,
				   hmat_eigvals=None,
				   return_weights=False,
				   method=None,
				   accelerator='python',
				   kernel_name='quintic',
				   integration='midpoint',
                   ):

	dim = positions.shape[-1]
	assert dim==2 or dim==3, f"Particle positions must be of shape (N, 2) or (N, 3) but found {positions.shape}"

	if not isinstance(averaged, (list, tuple)):
		averaged = [averaged]

	if not isinstance(extent, list):
		extent = [extent]
	extent = np.asarray(extent)

	# prepare the function arguments
	positions  = positions.astype('float32')
	quantities = quantities.astype('float32')
	extent     = extent.astype('float32')

	if accelerator == 'cpp':
		positions = torch.from_numpy(positions)
		quantities = torch.from_numpy(quantities)
		extent = torch.from_numpy(extent)
		if hsm is not None:
			hsm = torch.from_numpy(hsm.astype('float32'))
		if hmat_eigvecs is not None:
			hmat_eigvecs = torch.from_numpy(hmat_eigvecs.astype('float32'))
			hmat_eigvals = torch.from_numpy(hmat_eigvals.astype('float32'))

	if "adaptive" in method:
		# +1e-8 due to error when pos exactly equals boxsize
		point_tree = spatial.cKDTree(positions, boxsize=extent[1] + 1e-8)
		# [0]=distances; [:, -1] we only need the last NN
		# this follows the same "definition" of the smoothing length as 0.5 * distance to Nth nearest neighbor
		pcellsizesHalf = point_tree.query(x=positions, k=num_nn, workers=-1)[0][:, -1] * 0.5
		pcellsizesHalf = pcellsizesHalf.astype('float32')
		if accelerator == 'cpp':
			pcellsizesHalf = torch.from_numpy(pcellsizesHalf)

	"""
	# identify correct deposition function
	if dim==2:
		if method == 'ngp':
			func = ngp_2d_cpp if accelerator == 'cpp' else ngp_2d

		elif method == 'cic':
			func = cic_2d_cpp if accelerator == 'cpp' else cic_2d

		elif method == 'cic_adaptive':
			func = cic_2d_adaptive_cpp

		elif method == 'tsc':			
			func = tsc_2d_cpp if accelerator == 'cpp' else tsc_2d

		elif method == 'tsc_adaptive':
			func = tsc_2d_adaptive_cpp
		
		elif method == 'sph_isotropic':
			func = isotropic_kernel_deposition_2d_cpp

		elif method == 'sph_anisotropic':
			func = anisotropic_kernel_deposition_2d_cpp
		
		else:
			raise ValueError(f"Unknown method: {method}. Use 'ngp', 'cic', 'cic_adaptive', 'tsc', 'tsc_adaptive'.")
		
	else:
		if method == 'ngp':
			func = ngp_3d_cpp if accelerator == 'cpp' else ngp_3d

		elif method == 'cic':
			func = cic_3d_cpp if accelerator == 'cpp' else cic_3d

		elif method == 'tsc':
			func = tsc_3d_cpp if accelerator == 'cpp' else tsc_3d

		elif method == 'cic_adaptive':
			func = cic_3d_adaptive_cpp
		
		elif method == 'tsc_adaptive':
			func = tsc_3d_adaptive_cpp

		elif method == 'sph_isotropic':
			func = isotropic_kernel_deposition_3d_cpp

		elif method == 'sph_anisotropic':
			func = anisotropic_kernel_deposition_3d_cpp
		
		else:
			raise ValueError(f"Unknown method: {method}. Use 'ngp', 'cic', 'cic_adaptive', 'tsc', 'tsc_adaptive'.")
	"""

	# select the deposition function based on the method, dim and accelerator
	namespace = pyfunc if accelerator=="python" else cppfunc
	func = getattr(namespace, f"{method}_{dim}d")


	# perform deposition
	if "adaptive" in method:
		fields, weights = func(positions, quantities, extent, gridnum, periodic,
						 pcellsizesHalf, 
						 )
	
	elif "isotropic" in method:
		fields, weights = func(positions, quantities, extent, gridnum, periodic, 
						 hsm, 
						 kernel_name,
						 integration
						 )
	
	elif "anisotropic" in method:
		fields, weights = func(positions, quantities, extent, gridnum, periodic, 
						 hmat_eigvecs, 
						 hmat_eigvals, 
						 kernel_name,
						 integration
						 )
	
	else:
		fields, weights = func(positions, quantities,extent, gridnum, periodic
						 )


	# divide averaged fields by weight
	for i in range(len(averaged)):
		if averaged[i] == True:
			fields[..., i] /= (weights + 1e-10)

	if return_weights:
		return fields, weights
	else:
		return fields

"""
def isotropic_kernel_deposition(positions,
								hsm,
								quantities,
								averaged,
								gridnum,
								extent,
								periodic=1
								):
	
	dim = positions.shape[-1]
	assert dim==2 or dim==3, f"Particle positions must be of shape (N, 2) or (N, 3) but found {positions.shape}"
	
	if len(quantities.shape)==1:
		quantities = quantities[:, np.newaxis]

	# cast to float32
	positions  = positions.astype('float32')
	hsm        = hsm.astype('float32')
	quantities = quantities.astype('float32')
	extent 	   = extent.astype('float32')
	gridnum	   = int(gridnum)

	deposition_strategy = isotropic_kernel_deposition_2d if dim==2 else isotropic_kernel_deposition_3d
	fields, weights = deposition_strategy(positions,
										  hsm,
										  quantities,
										  extent, 
										  gridnum, 
										  periodic)

	# divide averaged fields by weight
	for i in range(len(averaged)):
		if averaged[i] == True:
			fields[..., i] /= (weights + 1e-10)

	return fields, weights


def anisotropic_kernel_deposition(positions,
								  hmat,
								  quantities,
								  averaged,
								  gridnum,
								  extent,
								  periodic=1,
								  plane='xy',
								  evals=None,
								  evecs=None,
								  return_evals=False
								  ):
	
	dim = positions.shape[-1]
	assert dim==2 or dim==3, f"Particle positions must be of shape (N, 2) or (N, 3) but found {positions.shape}"
	
	if len(quantities.shape)==1:
		quantities = quantities[:, np.newaxis]

	if dim == 2:
		# need to project the smoothing ellipses on the cartesian plane
		if plane == 'xy':
			e1 = [1, 0, 0]
			e2 = [0, 1, 0]
		elif plane == 'xz':
			e1 = [1, 0, 0]
			e2 = [0, 0, 1]
		elif plane == 'yz':
			e1 = [0, 1, 0]
			e2 = [0, 0, 1]
		else:
			print("Plane must be either xy, yz or xz.")


		# Compute P * M_inverse * P_transpose
		projection_matrix = np.array([e1, e2]).astype('float32')
		hmat_inv = np.linalg.inv(hmat)
		hmat_2d = []

		for i in range(len(hmat_inv)):
			hmat_2d.append(np.linalg.inv(np.dot(projection_matrix, np.dot(hmat_inv[i], projection_matrix.T))))
		hmat_2d = np.asarray(hmat_2d)

		# compute the new eigenvectors (2d)
		evals, evecs = np.linalg.eigh(hmat_2d)


	# cast to float32
	positions  = positions.astype('float32')
	quantities = quantities.astype('float32')
	extent 	   = extent.astype('float32')
	evals	   = evals.astype('float32')
	evecs	   = evecs.astype('float32')
	gridnum	   = int(gridnum)

	# cache it, for some reason the cython functions change the evals and evecs
	if return_evals:
		evals_copy = np.copy(evals)
		evecs_copy = np.copy(evecs)

	deposition_strategy = anisotropic_kernel_deposition_2d if dim==2 else anisotropic_kernel_deposition_3d
	fields, weights = deposition_strategy(positions,
										  evecs,
										  evals,
										  quantities,
										  extent, 
										  gridnum, 
										  periodic)

	# divide averaged fields by weight
	for i in range(len(averaged)):
		if averaged[i] == True:
			fields[..., i] /= (weights + 1e-10)

	return (fields, weights, evals_copy, evecs_copy) if return_evals else (fields, weights)

"""
# .......................................................................................... 
import numpy as np
from scipy import spatial

import numpy as np

def coordinate_difference_with_pbc(x, y, boxsize):
    return (x - y + 0.5 * boxsize) % boxsize - 0.5 * boxsize


def compute_hsm(pos, nn, boxsize=None):
	"""
	Estimate smoothing length as half the distance to the Nth nearest neighbor.
	"""
	if boxsize is None:
		tree = spatial.KDTree(pos)
	else:
		tree = spatial.cKDTree(pos, boxsize=boxsize)

	nn_dists, nn_inds = tree.query(pos, k=nn, workers=-1)
	hsm = nn_dists[:, -1] * 0.5
	return hsm, nn_dists, nn_inds, tree

import numpy as np

def compute_hsm_tensor(pos, masses, NN, boxsize=None):
    """
    Computes the smoothing tensor H for each particle using the covariance-based method
    from Marinho (2021), generalized for arbitrary dimension (D=2 or D=3).

    Args:
        pos:      (N, D) particle positions
        masses:   (N,) particle masses
        NN:       int, number of neighbors
        boxsize:  float or None, periodic box size

    Returns:
        H:        (N, D, D) smoothing tensor for each particle
        eigvals:  (N, D) eigenvalues of the covariance matrix (scaled)
        eigvecs:  (N, D, D) eigenvectors of the covariance matrix
    """
    N, D = pos.shape
    assert D in (2, 3), "Only 2D and 3D supported"

    # Step 1: Find neighbors
    _, _, nn_inds, _ = compute_hsm(pos, NN, boxsize)
    neighbor_masses = masses[nn_inds]          # (N, k)
    neighbor_coords = pos[nn_inds]             # (N, k, D)

    # Step 2: Handle periodic boundaries
    if boxsize is None:
        r_jc = neighbor_coords - pos[:, np.newaxis, :]   # (N, k, D)
    else:
        r_jc = coordinate_difference_with_pbc(neighbor_coords, pos[:, np.newaxis, :], boxsize)
    neighbor_coords = r_jc + pos[:, np.newaxis, :]       # (N, k, D)

    # Step 3: Center of mass of each cluster
    total_mass = np.sum(neighbor_masses, axis=1)   # (N,)
    rc = np.sum(neighbor_coords * neighbor_masses[..., np.newaxis], axis=1) / total_mass[..., np.newaxis]  # (N, D)

    # Displacement vectors from center
    deltas = neighbor_coords - rc[:, np.newaxis, :]   # (N, k, D)

    # Step 4: Mass-weighted covariance matrix Σ
    outer = np.einsum('...i,...j->...ij', deltas, deltas)   # (N, k, D, D)
    outer = outer * neighbor_masses[..., np.newaxis, np.newaxis]
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
def compute_hsm_tensor(pos, masses, NN, boxsize=None):

	N, D = pos.shape
	#assert D == 3, "Only 3D supported"

	# Step 1: Find neighbors
	_, _, nn_inds, _ = compute_hsm(pos, NN, boxsize)
	neighbor_masses = masses[nn_inds]  # (N, k)
	neighbor_coords = pos[nn_inds]  # (N, k, 3)

	# Step 2: Handle periodic boundaries
	if boxsize is None:
		r_jc = neighbor_coords - pos[:, np.newaxis, :]  # (N, k, 3)
	else:
		# Apply periodic boundary conditions
		r_jc = coordinate_difference_with_pbc(neighbor_coords, pos[:, np.newaxis, :], boxsize)
	neighbor_coords = r_jc + pos[:, np.newaxis, :] # (N, k, 3), yes this is correct!


	# Step 3: Center of mass of each cluster
	total_mass = np.sum(neighbor_masses, axis=1)  # (N,)
	rc = np.sum(neighbor_coords * neighbor_masses[..., np.newaxis], axis=1) / total_mass[..., np.newaxis]  # (N, 3)

	# Displacement vectors from center
	# No need to apply periodic boundary conditions here, as neighbor_coords is already shifted around central pos
	deltas = neighbor_coords - rc[:, np.newaxis, :]


	# Step 4: Compute mass-weighted covariance matrix Σ
	outer = np.einsum('...i,...j->...ij', deltas, deltas)  # (N, k, 3, 3)
	outer = outer * neighbor_masses[..., np.newaxis, np.newaxis]
	Sigma = np.sum(outer, axis=1) / total_mass[:, np.newaxis, np.newaxis]  # (N, 3, 3)

	# Step 5: Eigen-decomposition of Σ
	eigvals, eigvecs = np.linalg.eigh(Sigma)  # eigvals: (N, 3), eigvecs: (N, 3, 3)
	sqrt_eigvals = np.sqrt(eigvals)  # (N, 3)

	# Step 6: Compute ζ_max (max Mahalanobis distance)
	inv_Sigma = np.linalg.inv(Sigma)  # (N, 3, 3)
	mahal_sq = np.einsum('nik,nkl,nil->ni', deltas, inv_Sigma, deltas)  # ✅ Correct
	zeta_max = np.sqrt(np.max(mahal_sq, axis=1))  # (N,)

	# Step 7: Construct H = ζ_max * V * sqrt(Λ) * V.T
	Λ = np.zeros_like(Sigma)  # (N, 3, 3)
	for i in range(3):
		Λ[:, i, i] = sqrt_eigvals[:, i]

	H_raw = np.matmul(eigvecs, Λ)  # (N, 3, 3)
	H = np.matmul(H_raw, eigvecs.transpose(0, 2, 1))  # V * Λ * V.T
	H *= zeta_max[:, np.newaxis, np.newaxis]  # Scale by ζ_max

	# Step 8: Rescale eigenvalues
	eigvals_H = zeta_max[:, np.newaxis] * np.sqrt(eigvals)  # (N, 3)

	return H, eigvals_H, eigvecs
"""
"""
def compute_hsm(pos, nn, boxsize=None):
	
	pos:           	particle positions
	nn:            	number of neighbors to consider
	boxsize:		if specified, use periodic kdtree, else non-periodic
	returns:       	computes the smoothing length for each 
				   	particle as 0.5 * distance to Nth nn.
	
	
	if boxsize is None:
		tree = spatial.KDTree(pos)	
	else:
		tree = spatial.cKDTree(pos, boxsize=boxsize)
	nn_dists, nn_inds = tree.query(pos, k=nn, workers=-1)

	# compute hsm
	hsm = nn_dists[:, -1] * 0.5
	return hsm, nn_dists, nn_inds, tree


def compute_hsm_tensor(pos, masses, NN, boxsize):

	# use compute_hsm() to find the nn_inds
	_, _, nn_inds, _ = compute_hsm(pos, NN, boxsize)
	
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
"""

def project_hsm_tensor_to_2d(hmat, plane):
	
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


def compute_div(masses, rho, A, nn_inds, grad_w):
	
	# need to ignore particle itself for gradient computation
	nn_inds = nn_inds[:, 1:]
	A_j = A[nn_inds]
	
	div_A  = 1 / rho[:, np.newaxis] * np.sum( masses[nn_inds][..., np.newaxis] * (A_j[..., np.newaxis] - A[:, np.newaxis, np.newaxis]) * grad_w, axis=1)
	return div_A


def compute_rot(masses, rho, A, nn_inds, grad_w):
	
	# need to ignore particle itself for gradient computation
	nn_inds = nn_inds[:, 1:]
	A_j = A[nn_inds]
	
	rot_A  = 1 / rho[:, np.newaxis] * np.sum( masses[nn_inds][..., np.newaxis] * np.cross(A_j - A[:, np.newaxis], grad_w), axis=1)
	return rot_A


def compute_density(dim, hsm, masses, nn_dists, nn_inds):
	
	#dim:           how many spatial dims
	#hsm:           particle smoothing lengths
	#masses:		   particle masses
	#nn_dists:	   distances to neighbors
	#nn_inds:       list of particle neighbors
	#returns:       computes the density at particle positions
	#
	w_ij = quintic_spline(dim, hsm, nn_dists)
	ρ_i  = np.sum( masses[nn_inds] * w_ij , axis=1)
	return ρ_i


def compute_density_gradient(dim, hsm, masses, positions, nn_inds):
	"""
	Computes the gradient of density for a set of particles.

	Parameters:
	- dim:        int, the spatial dimension (2 or 3).
	- hsm:        np.ndarray, smoothing lengths for particles.
	- masses:     np.ndarray, masses of particles.
	- positions:  np.ndarray, positions of particles (N x dim).
	- nn_inds:    np.ndarray, indices of the nearest neighbors for each particle.

	Returns:
	- grad_rho: np.ndarray, gradient of density for each particle (N x dim).
	"""
	# need to ignore particle itself for gradient computation
	nn_inds = nn_inds[:, 1:]

	# Compute kernel gradients
	kernel_grad = quintic_spline_gradient(dim, pos=positions, h=hsm, nn_inds=nn_inds)

	# Mask out self-contributions (ignore gradient from particle to itself)
	self_mask = np.arange(nn_inds.shape[0])[:, None] == nn_inds
	kernel_grad[self_mask] = 0.0

	# Compute density gradient for each particle
	grad_rho = np.sum(kernel_grad * masses[nn_inds][..., None], axis=1)

	return grad_rho


def compute_vsm(vel, nn_inds):
	"""
	vel:           particle velocities
	nn_inds:       list of particle neighbors
	returns:       computes the velocity dispersion vector
	"""
	vel_nn = vel[nn_inds]
	#vel_nn = np.concatenate([vel_nn, vel[:, np.newaxis]], axis=1).mean(axis=1) # attach the particle vel itself
	vsm = vel - vel_nn.mean(axis=1)
	return vsm




"""
def grad_W(dim, h, r_i, r_js):
	#h:             smoothing lengths
	#q:             abs(relative coords)
 
	if len(h.shape)==1:
		h = h[:, np.newaxis]

	if dim == 1:
		sigma = 1.0 / (120 * h)
	elif dim == 2:
		sigma = 7.0 / (478 * math.pi * h ** 2)
	elif dim == 3:
		sigma = 1.0 / (120 * math.pi * h ** 3)
	
	for j in range(r_js.shape[-2]):
		r_js[:, j] = r_i - r_js[:, j]

	x_ij = r_js # this holds vector differences!
	r_ij = np.linalg.norm(x_ij, axis=-1)
	q = r_ij / h
	
	dwdq = np.zeros_like(q)
	dwdq = np.where(np.logical_and(0<=q, q<=1), (-5) * (3-q)**4 + 5*6*(2-q)**4 - 5*15*(1-q)**4, dwdq)
	dwdq = np.where(np.logical_and(1< q, q<=2), (-5) * (3-q)**4 + 5*6*(2-q)**4, dwdq)
	dwdq = np.where(np.logical_and(2< q, q<=3), (-5) * (3-q)**4, dwdq)
	dwdq = sigma * dwdq
	
	dwdq = dwdq / (h * r_ij)
	dwdx = dwdq * x_ij[..., 0]
	dwdy = dwdq * x_ij[..., 1]
	
	if dim == 3:
		dwdz = dwdq * x_ij[..., 2]
		grad = np.stack([dwdx, dwdy, dwdz], axis=-1)
	
	elif dim == 2:
		grad = np.stack([dwdx, dwdy], axis=-1)

	return grad

def grad_rho(rho, masses, grad_w, nn_inds):
	rho_i = rho[:, np.newaxis, np.newaxis]
	rho_j = rho[nn_inds][..., np.newaxis]
	m_j = masses[:, np.newaxis, np.newaxis]
	return np.sum( m_j * (rho_j - rho_i) / rho_j * grad_w, axis=-2)

def grad_rho2(rho, grad_rho, masses, grad_w, nn_inds):
	
	m_j = masses[nn_inds]
	m_j = m_j[..., np.newaxis, np.newaxis]
	
	#gradW_j = grad_w[nn_inds][..., np.newaxis]
	print(grad_w.shape)
	
	rho_i = rho[..., np.newaxis, np.newaxis, np.newaxis]
	rho_j = rho[nn_inds][..., np.newaxis, np.newaxis]
	
	grad_rho_i = grad_rho[..., np.newaxis, :, np.newaxis]
	grad_rho_j = grad_rho[nn_inds][..., np.newaxis, :]
	
	print(rho_i.shape, rho_j.shape)
	print(grad_rho_i.shape, grad_rho_j.shape)
	
	print((grad_rho_i / rho_j).shape, (grad_rho_j / rho_i).shape)
	
	outer_prod = np.matmul(grad_rho_i / rho_j, 
						   grad_rho_j / rho_i)
	print(outer_prod.shape)
	

	tensor = np.sum(m_j * outer_prod, axis=1)
	
	# compute three eigenvectors
	eigvals, eigvectors = np.linalg.eigh(tensor)
	print(eigvals.shape, eigvectors.shape)
	return eigvectors
	#return np.sum(m_j * outer_prod * gradW_j, axis=1)

h_sphere, nn_dists, nn_inds, tree = _compute_hsm(pos, boxsize, NN)
density = _compute_density(2, h_sphere, masses, nn_dists, nn_inds)
grad    = grad_W(dim=2, h=h_sphere, r_i=pos, r_js=pos[nn_inds])

rho_grad = grad_rho(density, masses, grad, nn_inds)
eigenvectors = grad_rho2(density, rho_grad, masses, grad, nn_inds)

print(eigenvectors.shape)

# re-scale the eigenvectors with hsm
eigenvectors = eigenvectors * h_sphere[..., np.newaxis, np.newaxis]
"""



