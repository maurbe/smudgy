# Backends and Parallelization

`smudgy` is designed to make efficient use of modern hardware without changing your workflow. Whether you are working on a laptop, a workstation, or an HPC cluster, the same high-level API can transparently execute on different parallel architectures.

## Choosing a backend and accelerator

`smudgy` currently provides two computational backends:

| Backend | Description |
|---------|-------------|
| `"quadrants"` (default) | High-performance backend supporting multi-core CPUs and GPUs. Supports all operations and is recommended for all workloads. |
| `"numpy"` | Vectorized reference implementation based on NumPy, used for testing and cross-validation. Supports interpolation and fixed-grid deposition (`adaptive=False`) only. |

Quadrants is a required dependency of `smudgy` in either case.

The backend can be selected when constructing a {py:class}`~smudgy.pointcloud.PointCloud`:

```python
import smudgy as sm

pc = sm.PointCloud(
    positions,
    backend="quadrants",
)
```

or changed later at any time:

```python
pc.set_backend("numpy")
```

The default backend is implemented using the
[Quadrants programming language](https://github.com/Genesis-Embodied-AI/quadrants),
allowing `smudgy` to automatically exploit parallel hardware while exposing a simple NumPy-like interface. Unlike explicit parallel programming, no threading or GPU kernels have to be written by the user -- the Quadrants backend takes care of this automatically.

When using the `"quadrants"` backend, an execution target (accelerator) can also be selected via the `arch` keyword: `"cpu"`, `"gpu"`, `"cuda"`, `"vulkan"` or `"metal"`. If no `arch` is given, `smudgy` keeps the already active Quadrants runtime, or initializes it with `"gpu"` on first use (Quadrants falls back to the CPU if no supported GPU is found). Any other keyword argument is forwarded to `ti.init`.

:::{caution}
The Quadrants runtime is shared by all {py:class}`~smudgy.pointcloud.PointCloud` instances of a Python process. Switching the architecture of a running runtime (e.g. `"gpu"` → `"cpu"`) requires `force=True` and recompiles all kernels, so it is best to choose `arch` once, at the start of a script.
:::

### CPU (multi-threaded)

```python
pc = sm.PointCloud(
    positions,
    backend="quadrants",
    arch="cpu",
) 
# or
pc.set_backend("quadrants", arch="cpu", force=True)
```

Quadrants automatically distributes work across all available CPU cores using a shared-memory parallel execution model. If desired, the number of threads can be limited:

```python
pc.set_backend(
    "quadrants",
    arch="cpu",
    cpu_max_num_threads=8,
)
```

### GPU

If a supported GPU is available, simply use

```python
pc.set_backend(
    "quadrants",
    arch="gpu",
)
```

Quadrants automatically selects the best supported GPU backend available on your system (CUDA, Metal, Vulkan, etc.). No code changes are required -- the same `smudgy` API executes on either CPU or GPU.

---

## Parallelization

Most operations in `smudgy` consist of performing the same computation independently for many particles, neighbours or grid cells. For example,

- evaluating kernel weights,
- interpolating particle properties,
- depositing particles onto a grid,

can all be computed independently for large numbers of elements.

Internally, the Quadrants backend expresses these operations as parallel loops. Instead of executing

```text
particle 1
particle 2
particle 3
...
```

sequentially, Quadrants distributes the iterations across all available hardware resources.

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

While Quadrants efficiently utilizes the resources of a *single* machine, many scientific datasets are simply too large to fit into the memory of a single computer or to be processed efficiently on one machine alone. For these cases, `smudgy` distributes the computations across many independent MPI processes (*ranks*) using
[`mpi4py`](https://mpi4py.github.io/).
These ranks may reside on different CPU sockets, on multiple machines, or on different nodes of an HPC cluster.

No changes to the analysis script are required: if `smudgy` detects more than one MPI rank when a {py:class}`~smudgy.pointcloud.PointCloud` is constructed, it decomposes the point cloud and each rank subsequently works on its own share. The same script thus runs unmodified on a laptop, a multi-GPU workstation, or across many nodes of a cluster. A distributed run proceeds in four stages.

### 1. Domain decomposition

The point cloud is partitioned once, at construction. Only rank 0 reads the input arrays (`positions`, `weights`); the arguments are ignored on all other ranks. The particles are ordered along a [Hilbert curve](https://en.wikipedia.org/wiki/Hilbert_curve), which maps spatially close points to nearby indices, and the curve is cut into contiguous, count-balanced intervals, one per rank. Rank 0 sends each rank only its own chunk, so the full dataset is never broadcast:

```text
Global particle set (rank 0)
            │  Hilbert sort + split into equal-count chunks
            ▼
    R0 | R1 | R2 | R3      (spatially compact regions)
```

Afterwards, `pc.positions` and `pc.weights` hold only the rank's own particles (in Hilbert order), and `pc.decomposition` ({py:class}`~smudgy.decomposition.DecompositionInfo`) records where they came from. Per-particle fields must be added via {py:meth}`~smudgy.pointcloud.PointCloud.add_fields`, which (again reading the full array on rank 0) splits them in the same way. A plain attribute assignment such as `pc.temperature = temperature` is *not* split, and `smudgy` warns when it detects one. The decomposition is not updated after construction, i.e. it is not possible to move particles and rebalance an existing point cloud.

### 2. Ghost particles and neighbour search

Since a particle's nearest neighbours may reside on another rank, each rank also imports the required boundary particles from other ranks as *ghost particles*. Starting from a search radius set by the local particle density, it requests all particles in that region and enlarges the radius until each of its particles has found its $N_{\rm ngb}$ neighbours. The neighbour search is therefore exact and identical to a single-rank run. All subsequent neighbour queries use a local $k$-d tree built over the rank's own and ghost particles.

This step runs automatically the first time neighbours are needed, or explicitly via {py:meth}`~smudgy.pointcloud.PointCloud.find_neighbors`. Its `max_iterations` argument limits the number of radius-growing iterations, and `on_max_iterations` decides whether an unconverged search raises an error (`"raise"`, default) or only warns. Since the radius cannot grow beyond half the box size in periodic boxes, `num_neighbors` must be small enough compared to the number of particles per rank.

Query positions passed to {py:meth}`~smudgy.pointcloud.PointCloud.interpolate` or {py:meth}`~smudgy.pointcloud.PointCloud.compute_smoothing` (again only read on rank 0) are routed through the same Hilbert intervals to the rank containing the surrounding region.

### 3. Local computation

Each rank then independently performs the requested operation on its own particles (plus ghosts), parallelized within the rank by the Quadrants backend. For instance, when depositing particles onto a grid, every rank deposits its own particles onto a **local grid**.

### 4. Collecting the results

Per-particle results (smoothing lengths and tensors, densities, interpolated values) **remain distributed**: each rank holds the values of its own particles or query points. They are collected in the original input order on a single rank on demand:

```python
rho = pc.get("density_isotropic")             # looks up pc.smoothing / pc attributes
h   = pc.get("smoothing_lengths")
f_p = pc.gather_particles(pc.interpolate("f"))                          # at the particles
f_q = pc.gather_queries(pc.interpolate("f", query_positions=queries))   # at query positions
```

All three return the full array on the root rank (rank 0 by default) and `None` on all other ranks.

Deposited grids, in contrast, are summed across ranks automatically. By default, the summed grid is returned on every rank (`MPI.Allreduce`); with `gather_to_root=True`, it is only delivered to the `root` rank (`MPI.Reduce`), which saves memory and communication when only one rank needs it:

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
Final deposition grid (every rank, or `root` only)
```

:::{tip}
On a single rank, the decomposition, ghost exchange and gathers are skipped (the particle order is left unchanged), and `get`, `gather_particles` and `gather_queries` simply return the full arrays. Scripts that use them therefore run unmodified with any number of ranks.
:::

---

## Combining Quadrants and MPI

Quadrants and MPI parallelize computations at different levels and therefore complement each other naturally. MPI distributes the work across many machines, while Quadrants efficiently parallelizes the computation *within* each machine. This hybrid approach enables `smudgy` to scale from a laptop with a few CPU cores to large HPC systems comprising hundreds of nodes, as shown conceptually by the graph below:

```text
HPC cluster
│
├── Node 0
│     ├── MPI Rank 0
│     │      └── Quadrants → half the CPU cores / GPU 0
│     └── MPI Rank 1
│            └── Quadrants → half the CPU cores / GPU 1
│
├── Node 1
│     ├── MPI Rank 2
│     │      └── Quadrants → half the CPU cores / GPU 0
│     └── MPI Rank 3
│            └── Quadrants → half the CPU cores / GPU 1
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

### Sharing a node between ranks

Several ranks on one node share its cores and GPUs. `smudgy` reads the node-local rank from the launcher's environment (Open MPI, MPICH/Intel MPI, MVAPICH2, Cray PALS, Slurm) and by default:

- **CPU threads:** splits the node's cores evenly between its ranks, for both the Quadrants thread pool and the $k$-d tree queries. If the launcher already pinned each rank to a set of cores (e.g. `srun --cpus-per-task=8`, `mpirun --map-by ...:PE=8`), that set is used. Pass `cpu_max_num_threads=...` to override the Quadrants part.
- **GPUs:** gives each rank on a node its own GPU (`device="auto"`): node-local rank modulo the number of GPUs. Quadrants itself always opens CUDA device 0, so `smudgy` sets `CUDA_VISIBLE_DEVICES` (Vulkan: `QD_VISIBLE_DEVICE`) before the runtime starts. A launcher that already assigns one GPU per rank (e.g. `srun --gpus-per-task=1`) is left alone. Pass `device=<int>` to choose explicitly, or `device=None` to disable. The variable must be set before anything else in the process initialises CUDA (e.g. importing and using PyTorch or CuPy first).

With more ranks than GPUs per node, several ranks share a GPU and its memory.

### Compiled-kernel cache

Quadrants caches compiled kernels on disk (by default in `~/.cache/quadrants`) and all ranks share that directory through a lock file. On the first run every rank may compile the same kernels and wait on the lock; later runs load them from the cache. On clusters with slow or node-local home directories, point the cache to a shared, fast file system with `QD_OFFLINE_CACHE_FILE_PATH=/path/to/cache`, and warm it up once with a single-rank run of the same script.
