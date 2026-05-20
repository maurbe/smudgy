# Quick Start

Welcome to the `smudgy` documentation!

`smudgy` is a python package with a C++ backend and optional OpenMP acceleration for point cloud smoothing, interpolation and grid deposition -- lightning fast, scalable and memory-efficient.

A typical workflow may look like this:

```python
import numpy as np
import smudgy as sm

N = 1000
boxsize = 1.0
num_fields = 3

positions       = np.random.uniform(0, boxsize, (N, 3))
field_values    = np.random.normal(0, 1, (N, num_fields))
weights         = np.ones(N)

points = sm.PointCloud(positions=positions, weights=weights, boxsize=boxsize)

grid = points.deposit_to_grid(
    fields=field_values,
    averaged=True,
    gridnums=64,

    # e.g. perform cloud-in-cell deposition
    kernel_name='tophat',
    structure='separable'
    )

print(grid.shape)  # (64, 64, 64, num_fields)
```
