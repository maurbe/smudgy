# Quick Start

Welcome to the `smudgy` documentation!

`smudgy` is a high-performance Python package for smoothing, interpolation, and grid deposition of point-cloud data -- lightning fast, scalable and memory-efficient.

Whether you’re working on a laptop or an HPC cluster, `smudgy` is built to make efficient use of your hardware. It leverages the [quadrants](https://github.com/Genesis-Embodied-AI/quadrants) programming language for automatic CPU and GPU parallelization, and can seamlessly scale to multiple nodes using MPI through [mpi4py](https://mpi4py.github.io/mpi4py/stable/html/index.html).

A typical workflow may look like this:

```python
import numpy as np
import smudgy as sm

N = 10_000
boxsize = 1.0

positions = np.random.uniform(0, boxsize, (N, 3))
masses = np.ones(N)
velocity = np.random.normal(size=(N, 3))

pc = sm.PointCloud(
    positions=positions,
    weights=masses,
    boxsize=boxsize,
    backend="quadrants",
    arch="gpu",
)
pc.add_fields("velocity", velocity)

pc.global_setup(
    num_neighbors=32,
    structure="covariant",
    kernel_name="cubic_spline",
)
pc.compute_smoothing()   # smoothing tensors H_i
pc.compute_density()     # densities rho_i

# mass map and mass-weighted mean velocity on a 128^3 grid
mass_grid = pc.deposit(fields=None, averaged=False, gridnums=128)
vel_grid = pc.deposit("velocity", averaged=True, gridnums=128)

print(mass_grid.shape)  # (128, 128, 128)
print(vel_grid.shape)   # (3, 128, 128, 128)

# interpolate the velocity field and its curl at arbitrary positions
query = np.random.uniform(0, boxsize, (1000, 3))
v_q = pc.gather_queries(pc.interpolate("velocity", query_positions=query))
curl_q = pc.gather_queries(pc.interpolate("velocity", query_positions=query, mode="curl"))
```

To launch the same script on 8 MPI ranks (e.g. on an HPC cluster), run:

```bash
srun -n 8 python3 my_script.py
# or, with a generic MPI installation
mpiexec -n 8 python3 my_script.py
```

No changes to the script are needed: the input arrays are only read on rank 0, and the point cloud is automatically split across the ranks. Deposited grids are summed over all ranks, while per-particle results stay distributed and are collected on rank 0 via `pc.gather_queries`, `pc.gather_particles` or `pc.get` (they return `None` on the other ranks). See {doc}`backends <../wiki/backend>` for details.
