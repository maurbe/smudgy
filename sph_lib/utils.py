import numpy as np

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