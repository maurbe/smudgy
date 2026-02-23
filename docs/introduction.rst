Introduction
============

Welcome to the sph-lib documentation!

sph-lib provides advanced tools for smoothing and interpolating point cloud
data, specifically tailored for particle-based simulations using
(anisotropic) Smoothed Particle Hydrodynamics (SPH).

sph-lib is designed to be fast, scalable and memory-efficient. 
While providing user-friendly Python wrappers, it leverages C++ and OpenMP for parallelization.

The package's core functionality provides efficient interpolation of particle fields either onto 
new coordinates or structured grids, while handling both open and periodic boundary conditions, 
as well as non-uniform pixel and voxel sizes.

Quick start
-----------

.. code-block:: python
    
    import numpy as np
    from sph_lib import PointCloud

    N = 1000
    positions = np.random.uniform(0, 1, (N, 3))
    masses = np.random.uniform(0, 1, N)
    
    sim = PointCloud(positions=positions, masses=masses, boxsize=1.0)
    
    grid = sim.deposit_to_grid(
        fields=masses,
        averaged=False,
        gridnums=64,
        method="cic",
        )
    
    print(grid.shape)  # (64, 64, 64)
