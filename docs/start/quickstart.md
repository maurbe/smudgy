# Quick Start

Welcome to the `smudgy` documentation!

`smudgy` is a high-performance Python package for smoothing, interpolation, and grid deposition of point-cloud data -- lightning fast, scalable and memory-efficient.

Whether you’re working on a laptop or an HPC cluster, `smudgy` is built to make efficient use of your hardware. It leverages the [taichi](https://www.taichi-lang.org/) programming language for automatic CPU and GPU parallelization, and can seamlessly scale to multiple nodes using MPI through [mpi4py](https://mpi4py.github.io/mpi4py/stable/html/index.html).

A typical workflow may look like this:

```python
import numpy as np
import smudgy as sm

N = 1000
boxsize = 1.0

positions = np.random.uniform(0, boxsize, (N, 3))
weights = np.ones(N)
field = np.random.normal(size=(N, 3))

pc = sm.PointCloud(
    positions=positions,
    weights=weights,
    boxsize=boxsize,
    backend="taichi",
    arch="gpu",
)

pc.global_setup(
    num_neighbors=32,
    structure="covariant",
    kernel_name="cubic_spline",
)
pc.compute_smoothing()

grid = pc.deposit(
    field,
    averaged=True,
    gridnums=128,
)

print(grid.shape) # (1, 128, 128, 128)
```

To launch the same script on 8 MPI ranks (e.g. on an HPC cluster), run:

```bash
srun -n 8 python3 my_script.py
```