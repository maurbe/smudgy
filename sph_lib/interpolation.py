import math
import numpy as np
import scipy.spatial as spatial
from . import kernels


class sph_interpolator():
	
	def __init__(self,
				 particle_positions,
				 particle_masses,
				 boxsize,
				 number_of_nn,
				 kernel='gaussian'
				 ):
		
		self.pmasses = particle_masses
		self.particle_positions = particle_positions
		self.nn = number_of_nn
		self.dim = particle_positions.shape[-1]

		# compute relevant quantities for particles
		print(f'Computing {self.dim}d smoothing lengths and densities...')
		self.hsm, self.nn_dists, self.nn_inds, self.tree = self.compute_hsm(particle_positions, self.nn, boxsize)

		# set kernel
		try:
			kernel_class = getattr(kernels, kernel)
			self.kernel = kernel_class(dim=self.dim)
		except AttributeError:
			raise ValueError(f"Kernel '{kernel}' is not implemented.")

		# compute density
		self.compute_density()
	

	def compute_hsm(self, pos, nn, boxsize=None):
		"""
		pos:           	particle positions
		nn:            	number of neighbors to consider
		boxsize:		if specified, use periodic kdtree, else non-periodic
		returns:       	computes the smoothing length for each 
						particle as 0.5 * distance to Nth nn.
		"""
		
		if boxsize is None:
			tree = spatial.KDTree(pos)	
		else:
			tree = spatial.cKDTree(pos, boxsize=boxsize)
		nn_dists, nn_inds = tree.query(pos, k=nn+1, workers=-1) # do not consider particle as neighbor of itself

		# compute hsm
		hsm = nn_dists[:, -1] * 0.5
		return hsm, nn_dists, nn_inds, tree


	def compute_density(self):
		"""		Computes the density at particle positions.
		Returns: density vector ρ_i at particle positions.
		"""
		w_ij = self.kernel.evaluate_kernel(r_ij=self.nn_dists, h=self.hsm)
		self.rho = np.sum( self.pmasses[self.nn_inds] * w_ij , axis=1)
	

	def interpolate_fields_at_positions(self, 
								 particle_fields,
								 pos_interpol
								 ):
		"""
		particle_fields:	list of quantities at particle positions, shape (num_fields, N)
		masses:         	particle masses
		densities:      	particle densites
		hsm:            	particle smoothing lengths
		tree:           	ckdtree
		pos_interpol:   	positions at which to compute the particle_fields values

		returns:        	interpolated fields at query positions
		"""
		if not isinstance(particle_fields, list):
			particle_fields = [particle_fields]
		
		# the "h" in the formula is the smoothing length of particle j, i.e. the "gather" approach
		# no need to recompute new smoothing lengths for "virtual particle" query positions
		nn_dists, nn_inds = self.tree.query(pos_interpol, k=self.nn, workers=-1)
		h_i = (nn_dists[:, -1] * 0.5)[:, np.newaxis]
		h_ij= h_i
		w_ij= self.kernel.evaluate_kernel(r_ij=nn_dists, h=h_ij)

		# prepare variables and cast to correct shapes
		particle_fields = [f[..., np.newaxis] for f in particle_fields]
		particle_fields = np.concatenate(particle_fields, axis=-1)

		pf = particle_fields[nn_inds]
		pm 	 = self.pmasses[nn_inds]
		prho = self.rho[nn_inds]
		
		w_ij = w_ij[..., np.newaxis]
		pm 	 = pm[..., np.newaxis]
		prho = prho[..., np.newaxis]

		# main interpolation formula
		A_i = np.sum( pf * w_ij * pm / prho, axis=1)

		# move fields to first axis for consistency with input structure
		A_i = np.moveaxis(A_i, -1, 0)
		return A_i


	def interpolate_gradients_at_positions(self, 
										  particle_fields,
										  pos_interpol
										  ):

		if not isinstance(particle_fields, list):
			particle_fields = [particle_fields]
		
		# the "h" in the formula is the smoothing length of particle j
		# no need to recompute new smoothing lengths for "virtual particle" query positions
		nn_dists, nn_inds = self.tree.query(pos_interpol, k=self.nn, workers=-1)
		h_i = (nn_dists[:, -1] * 0.5)[:, np.newaxis]
		h_ij= h_i

		r_ij_vec = self.particle_positions[nn_inds] - pos_interpol[:, np.newaxis, :]
		grad_w_ij= self.kernel.evaluate_gradient(r_ij_vec=r_ij_vec, h=h_ij) 

		# prepare variables and cast to correct shapes
		particle_fields = [f[..., np.newaxis] for f in particle_fields]
		particle_fields = np.concatenate(particle_fields, axis=-1)
		grad_w_ij = grad_w_ij[..., np.newaxis, :]  # add a new axis for the field dimension
		pf = particle_fields[nn_inds][..., np.newaxis]
		pm 	 = self.pmasses[nn_inds][..., np.newaxis, np.newaxis]
		prho = self.rho[nn_inds][..., np.newaxis, np.newaxis]
		
		# main interpolation formula
		grad_A_i = np.sum( pf * grad_w_ij * pm / (prho + 1e-8), axis=1)

		# move fields to first axis for consistency with input structure
		grad_A_i = np.moveaxis(grad_A_i, -2, 0)
		return grad_A_i

