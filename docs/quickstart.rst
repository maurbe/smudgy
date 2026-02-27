Quick Start
===========

Welcome to the ``smudgy`` documentation!

``smudgy`` is a python package with a C++ and (optionally) OpenMP backend for point cloud smoothing and interpolation using SPH operations -- lightning fast, scalable and memory-efficient.

At its core it provides routines for smoothing and interpolating particle fields, as well as depositing particle data onto structured grids.
Both open and periodic boundary conditions, as well as non-uniform pixel and voxel sizes are supported.

A typical workflow may look like this:

.. code-block:: python
    
    import numpy as np
    from smudgy import PointCloud

    N = 1000
    positions       = np.random.uniform(0, 1, (N, 3))
    weights         = np.random.uniform(0, 1, N)
    field_values    = np.random.uniform(0, 1, N)
    
    sim = PointCloud(positions=positions, weights=weights, boxsize=1.0)
    
    grid = sim.deposit_to_grid(
        fields=field_values,
        averaged=True,
        gridnums=64,
        method="cic",
        )
    
    print(grid.shape)  # (64, 64, 64)
