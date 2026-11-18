<!--
<br />
<p align="center">
  <img src="src/banner.png" alt="Smudgy Banner" width="100%">
</p>
-->
Welcome to the `smudgy` GitHub page!  `smudgy` is a high-performance Python package for smoothing, interpolation, and grid deposition of point-cloud data -- lightning fast, scalable and memory-efficient.

Whether you’re working on a laptop or an HPC cluster, `smudgy` is built to make efficient use of your hardware. It leverages the [Quadrants](https://github.com/Genesis-Embodied-AI/quadrants) programming language for automatic CPU and GPU parallelization, and can seamlessly scale to multiple nodes using MPI through [mpi4py](https://mpi4py.github.io/mpi4py/stable/html/index.html).

Large point clouds are split across MPI ranks along a Hilbert curve, with an exact neighbour search across rank boundaries via ghost particles, so the same script runs unmodified on one or many ranks.

```bash
pip install smudgy
```

```python
import smudgy as sm

pc = sm.PointCloud(positions, weights=masses, boxsize=1.0)
pc.global_setup(structure="covariant", kernel_name="cubic_spline", num_neighbors=32)
pc.compute_smoothing()
mass_grid = pc.deposit(fields=None, averaged=False, gridnums=256)
```

Please refer to the official [docs](https://smudgy.readthedocs.io/en/latest/) for installation guides and tutorials.