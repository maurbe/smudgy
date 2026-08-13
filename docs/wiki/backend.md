# Backends and Parallelization

`smudgy` is designed to make efficient use of modern hardware without changing your workflow. Whether you are working on a laptop, a workstation, or an HPC cluster, the same high-level API can transparently execute on different parallel architectures.

## Choosing a backend and accelerator

`smudgy` currently provides two computational backends:

| Backend | Description |
|---------|-------------|
| `"taichi"` (default) | High-performance backend supporting multi-core CPUs and GPUs. Recommended for all workloads. |
| `"numpy"` | Reference implementation based on NumPy. Useful for debugging, validation and environments where Taichi is unavailable. |

The backend can be selected when constructing a {py:class}`~smudgy.pointcloud.PointCloud`:

```python
import smudgy as sm

pc = sm.PointCloud(
    positions,
    backend="taichi",
)
```

or changed later at any time:

```python
pc.set_backend("numpy")
```

The default backend is implemented using the
[Taichi programming language](https://www.taichi-lang.org/),
allowing `smudgy` to automatically exploit parallel hardware while exposing a simple NumPy-like interface. Unlike explicit parallel programming, no threading or GPU kernels have to be written by the user -- the Taichi backend takes care of this automatically.

When using the `"taichi"` backend, an execution target (accelerator) can also be selected.

### CPU (multi-threaded)

```python
pc = sm.PointCloud(
    positions,
    backend="taichi",
    arch="cpu",
) 
# or
pc.set_backend("taichi", arch="cpu")
```

Taichi automatically distributes work across all available CPU cores using a shared-memory parallel execution model. If desired, the number of threads can be limited:

```python
pc.set_backend(
    "taichi",
    arch="cpu",
    cpu_max_num_threads=8,
)
```

### GPU

If a supported GPU is available, simply use

```python
pc.set_backend(
    "taichi",
    arch="gpu",
)
```

Taichi automatically selects the best supported GPU backend available on your system (CUDA, Metal, Vulkan, etc.). No code changes are required -- the same `smudgy` API executes on either CPU or GPU.

---

## Parallelization

Most operations in `smudgy` consist of performing the same computation independently for many particles, neighbours or grid cells. For example,

- evaluating kernel weights,
- interpolating particle properties,
- depositing particles onto a grid,

can all be computed independently for large numbers of elements.

Internally, the Taichi backend expresses these operations as parallel loops. Instead of executing

```text
particle 1
particle 2
particle 3
...
```

sequentially, Taichi distributes the iterations across all available hardware resources.

On a multi-core CPU, iterations are divided among worker threads:

```text
Thread 0   particle 0 ... particle 24999
Thread 1   particle 25000 ... particle 49999
Thread 2   ...
Thread 3   ...
```

On a GPU, thousands of lightweight threads execute these operations simultaneously.

This data-parallel execution model is particularly well suited for interpolation and deposition, where each particle contributes independently to neighbouring particles or grid cells. As a result, most workloads scale nearly automatically with the available hardware without requiring any user intervention.

---

## Distributed parallelization with MPI

While Taichi efficiently utilizes the resources of a *single* machine, many scientific datasets are simply too large to fit into the memory of a single computer or to be processed efficiently on one machine alone. For these cases, `smudgy` provides MPI wrappers based on
[`mpi4py`](https://mpi4py.github.io/),
allowing computations to be distributed across many independent MPI processes (often called *ranks*).

These ranks may reside

- on different CPU sockets,
- on multiple machines,
- on different nodes of an HPC cluster.

Each rank executes the same Python program but operates only on a subset of the data. Conceptually, a dataset containing one hundred million particles might be divided as

```text
Rank 0   particles          0 – 12,499,999
Rank 1   particles 12,500,000 – 24,999,999
Rank 2   ...
...
Rank 7   particles 87,500,000 – 99,999,999
```

Each rank then performs the requested operations, where the sub-problems are further parallelized using the Taichi backend as described above. Most distributed operations follow three stages:

1. **Scatter** -- the particle data is divided among all MPI ranks (here R0 to R3), and each process receives only its local subset of particles:

```text
Global particle set
                │
                ▼
┌──────┬──────┬──────┬──────┐
   R0     R1     R2     R3  
└──────┴──────┴──────┴──────┘
```

2. **Local computation** -- each rank independently performs the requested operation. Since the ranks do not interact during this stage, all computations proceed simultaneously. For instance, when depositing particles onto a grid, every rank creates a **local grid** and deposits its own chunk of particles:

```text
Rank 0  → local grid
Rank 1  → local grid
Rank 2  → local grid
Rank 3  → local grid
```

3. **Reduction** -- the local results are combined into the final output. Most operations compute per-point quantities, e.g. smoothing lengths, interpolated values, etc., and thus the local results only need to be appended to one another to form the final output. These operations gather arrays back to a single rank using collective communication such as `MPI.Gather` or `MPI.Gatherv`, depending on the operation and data layout. For additive quantities such as grid deposition, the reduction is performed using one of the MPI operations (e.g. `MPI.Allreduce` or `MPI.Reduce`), which sum the contributions from all local grids from every rank.

```text
Local grid 0
        +
Local grid 1
        +
Local grid 2
        +
Local grid 3
        │
        ▼
Final deposition grid
```

---

## Combining Taichi and MPI

Taichi and MPI parallelize computations at different levels and therefore complement each other naturally. MPI distributes the work across many machines, while Taichi efficiently parallelizes the computation *within* each machine. This hybrid approach enables `smudgy` to scale from a laptop with a few CPU cores to large HPC systems comprising hundreds of nodes, as shown conceptually by the graph below:

```text
HPC cluster
│
├── Node 0
│     ├── MPI Rank 0
│     │      └── Taichi → all CPU cores / GPU
│     └── MPI Rank 1
│            └── Taichi → all CPU cores / GPU
│
├── Node 1
│     ├── MPI Rank 2
│     │      └── Taichi → all CPU cores / GPU
│     └── MPI Rank 3
│            └── Taichi → all CPU cores / GPU
│
└── ...
```

### Running MPI programs

A normal Python script can be launched on multiple ranks using an MPI launcher. For example, on an HPC system using Slurm,

```bash
srun -n 8 python my_script.py
```

starts eight MPI ranks, while a generic MPI installation typically uses

```bash
mpirun -n 8 python my_script.py
# or
mpiexec -n 8 python my_script.py
```
