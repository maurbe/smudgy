# Quick Start

Welcome to the `smudgy` documentation!

`smudgy` is a python package with C++ and (optionally) OpenMP backend for point cloud smoothing and interpolation using SPH operations -- lightning fast, scalable and memory-efficient.

`smudgy` can perform interpolation and deposition of particle fields onto structured grids.
I supports both open and periodic boundary conditions, as well as non-uniform pixel and voxel sizes.

A typical workflow may look like this:

```python
import numpy as np
import smudgy as sm

N = 1000
positions       = np.random.uniform(0, 1, (N, 3))
weights         = np.random.uniform(0, 1, N)
field_values    = np.random.uniform(0, 1, N)

points = sm.PointCloud(positions=positions, weights=weights, boxsize=1.0)

grid = points.deposit_to_grid(
    fields=field_values,
    averaged=True,
    gridnums=64,

    # perform CIC deposition
    kernel_name='tophat',
    structure='separable'
    )

print(grid.shape)  # (64, 64, 64, 1)
```
