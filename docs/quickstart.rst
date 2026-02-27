Quick Start
===========

Welcome to the smudgy documentation!

smudgy provides tools for smoothing and interpolating point cloud
data, specifically geared towards particle-based simulations.
smudgy is lightning fast, scalable and memory-efficient via a C++ backbone and OpenMP parallelization.

You can interpolate particle fields either to new coordinates or deposit particle data onto structured grids.
Both open and periodic boundary conditions, as well as non-uniform pixel and voxel sizes are supported.


.. code-block:: python
    
    import numpy as np
    from smudgy import PointCloud

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
