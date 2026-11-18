# Overview

In the following sections we summarize the core class and methods exposed by `smudgy`, the mathematical model for kernels and smoothing, as well as the principal parameters you will pass through the pipeline. Throughout, particles carry an index $i$, positions $\mathbfit{x}_i$, masses (or, more generally, weights) $m_i$, densities $\rho_i$ and field values $f_i$; a query position is denoted by $\mathbfit{x}$ and its neighbourhood of $N_{\rm ngb}$ nearest particles by $\mathcal{N}$.

## The {py:class}`~smudgy.pointcloud.PointCloud` class

The primary user object is {py:class}`~smudgy.pointcloud.PointCloud`, which is initialized with a set of particle positions of shape `(N, d)` with $d \in \{1, 2, 3\}$, weights $m_i$ (optional, e.g. particle masses; uniform if omitted) and, if applicable, boxsize information

```python
pc = sm.PointCloud(positions=positions, weights=weights, boxsize=boxsize)
```

`boxsize` is an important parameter. If **not** provided, the code assumes open boundary conditions. Otherwise, the code uses periodic boundary conditions for all axes and wraps the positions into $[0, \texttt{boxsize})$. Currently, it is not possible to specify per-axis periodicity. If a single number is provided (e.g. `boxsize=1`), it is used for all dimensions, but it can also be specified per axis

```python
pc = sm.PointCloud(..., boxsize=[1.0, 2.0, 0.5]) # for a 3D periodic box
```

Additional keyword arguments configure the backend, e.g. `arch="cpu"` to run the Quadrants kernels on the CPU instead of the default GPU (see {doc}`backends <backend>`).

:::{note}
When running on several MPI ranks, `positions` and `weights` are only read on rank 0, and the point cloud is split across the ranks at construction. `pc.positions` and `pc.weights` then hold only the current rank's share of the particles, and all per-particle results stay distributed until they are explicitly collected (see {ref}`collecting results <collecting-results>`). On a single rank, everything behaves as if the arrays were global, in their original order. See {doc}`backends <backend>` for details.
:::

## Setting basic SPH parameters

The primary class exposes the utility method {py:meth}`~smudgy.pointcloud.PointCloud.global_setup` to easily set three global parameters, `structure`, `kernel_name` and `num_neighbors`, which are shared between most methods.

```python
pc = pc.global_setup(structure='isotropic', kernel_name='cubic_spline', num_neighbors=32)
```

+ `structure` sets how the kernel adapts to the particle distribution: `separable` (axis-aligned boxes, deposition only), `isotropic` (spheres) or `covariant` (ellipsoids).
+ `kernel_name` selects the kernel profile. All radial kernels (`tophat`, `tsc`, `gaussian`, `lucy`, `cubic_spline`, `quintic_spline`, `wendland_c2`, `wendland_c4`, `wendland_c6`) support the `isotropic` and `covariant` structures in 1D, 2D and 3D, while only `tophat`, `tsc` and `gaussian` are available for the `separable` structure. See the {doc}`kernels and structures <kernel_structures>` page and the {doc}`kernel tutorial <../tutorials/tutorial_exploring_kernels>` for a detailed overview.
+ `num_neighbors` is the number of nearest neighbours $N_{\rm ngb}$ used in computations (e.g. smoothing).

:::{tip}
With the {py:meth}`~smudgy.pointcloud.PointCloud.global_setup` method we can set these parameters globally, but they can also be passed individually to all relevant methods as shown below.
:::

## Smoothing the point cloud

In the context of SPH, particles can be thought of as extended objects with an inherent kernel function, whose size (and, for `covariant` structures, shape) is set by the particle's smoothing length or tensor.

### Neighbour search

All smoothing quantities are derived from the $N_{\rm ngb}$ nearest neighbours of every particle. `smudgy` finds them automatically the first time they are needed, using an exact $k$-d tree search (which, on several MPI ranks, also imports the required neighbouring particles from other ranks as *ghost particles*). The search can also be triggered explicitly via {py:meth}`~smudgy.pointcloud.PointCloud.find_neighbors`, which is also the way to change the number of neighbours afterwards:

```python
pc.find_neighbors(num_neighbors=64)
```

### Computing smoothing lengths and tensors

Based on the neighbours, {py:meth}`~smudgy.pointcloud.PointCloud.compute_smoothing` computes the smoothing quantities of the selected structure:

+ For `separable` and `isotropic` structures, the smoothing length $h_i$ is the distance to the $N_{\rm ngb}$-th nearest neighbour, so that every kernel encloses the same number of particles and the resolution follows the local density. For `separable` kernels, $h_i$ is used along every axis, resulting in axis-aligned boxes; in the `isotropic` case, it results in spherically symmetric kernels.

+ For `covariant` structures, the smoothing length is promoted to a **smoothing tensor** $\mathbfss{H}_i$ with elliptical (2D) or ellipsoidal (3D) support. Following Marinho (2021), its shape and orientation follow the mass-weighted covariance $\bmath{\Sigma}$ of the neighbours about the particle, and its size is set such that the kernel is the smallest ellipsoid of that shape enclosing all neighbours, $\mathbfss{H} = \zeta_{\max}\bmath{\Sigma}^{1/2}$. The eigenvalues $h_a$ (sorted in descending order) and eigenvectors of $\mathbfss{H}$ are the principal smoothing lengths and axes of the kernel. See {doc}`kernels and structures <kernel_structures>` for the full derivation. Three options control $\mathbfss{H}$:
    + `ratio_max` (default `100`) caps the axis ratio $h_{\max}/h_{\min}$, which guards against coplanar or collinear neighbour sets.
    + `min_axis` sets a lower bound on every axis, e.g. a softening length.
    + `volume_preserving=True` rescales $\mathbfss{H}$ so that $\det \mathbfss{H} = h^{d}$, keeping only the anisotropic shape. Note that this can shrink the longest axis, so the support may no longer contain every neighbour.

```python
# if `structure` and `num_neighbors` have already been set globally
pc.compute_smoothing()

# else
pc.compute_smoothing(structure='covariant', num_neighbors=32, ratio_max=10.0)
```

The code stores all smoothing results in the `smoothing` attribute (a {py:class}`~smudgy.smooth.SmoothingInfo` instance):

```python
# if structure = `separable` or `isotropic`
h = pc.smoothing.smoothing_lengths         # shape = (N,)

# if structure = `covariant`
H = pc.smoothing.smoothing_tensors         # shape = (N, d, d)
e = pc.smoothing.smoothing_tensors_eigvecs # shape = (N, d, d)
h = pc.smoothing.smoothing_tensors_eigvals # shape = (N, d)
```

:::{caution}
{py:meth}`~smudgy.pointcloud.PointCloud.compute_smoothing` also accepts `query_positions` to evaluate the smoothing quantities at arbitrary positions instead of the particles. In that case, `pc.smoothing` is *replaced* by the values at the query positions, so call `compute_smoothing()` again (without `query_positions`) before computing densities, interpolating or depositing.
:::

In case we already have precomputed smoothing lengths or tensors (e.g. smoothing lengths stored in a simulation snapshot), they can also be set manually via the {py:meth}`~smudgy.pointcloud.PointCloud.set_smoothing` method, e.g. for deposition:

```python
pc.set_smoothing(structure='isotropic', smoothing_lengths=my_h)
# or
pc.set_smoothing(structure='covariant',
                 smoothing_tensors=my_H,
                 smoothing_tensors_eigvecs=my_e,
                 smoothing_tensors_eigvals=my_l)
```

The arrays are indexed like `pc.positions`, i.e. on several ranks they must be the current rank's share.

## Kernels

Kernels in `smudgy` follow the standard SPH definition. For the `isotropic` structure with smoothing length $h$,

$$
W(\mathbfit{x}-\mathbfit{x}', h) = \frac{\alpha_d}{h^d} K(q)
\quad \text{with}\quad
q = \frac{\lVert\mathbfit{x}-\mathbfit{x}'\rVert}{h},
$$

where $K$ is the dimensionless kernel profile, $\alpha_d$ a dimension-dependent normalization constant, and the support radius is $n \times h$ with a kernel-dependent support multiple $n$. For the `covariant` structure, the kernel generalizes to

$$
W(\Delta\mathbfit{x}_i,\mathbfss{H}_i) = \frac{\alpha_d}{\det\mathbfss{H}_i} K(\xi_i), \qquad \xi_i = \lVert \mathbfss{H}_i^{-1}\Delta\mathbfit{x}_i \rVert,
$$

and `separable` kernels are products of one-dimensional kernels along each axis. All available kernels, their profiles and normalizations are listed on the {doc}`kernels and structures <kernel_structures>` page.

## Core methods

Using the {py:class}`~smudgy.pointcloud.PointCloud` instance, we can directly access the core functionalities of the package: continuous field **interpolation** and / or **deposition** onto structured grids. The following sections walk through the required methods and steps. More involved workflows can be found in the tutorial section.

### Density computation

For interpolation purposes, we need the density estimates at the particle positions,

$$
\rho_j = \sum_{i \in \mathcal{N}_j} m_i \, W(\mathbfit{x}_j - \mathbfit{x}_i, h_i).
$$

For this, we can use the {py:meth}`~smudgy.pointcloud.PointCloud.compute_density` method. It requires that {py:meth}`~smudgy.pointcloud.PointCloud.compute_smoothing` has been called with the same structure, otherwise an error is raised. The density arrays are stored per structure:

```python
# either
pc.compute_density()

# or
pc.compute_density(structure='covariant', kernel_name='cubic_spline')

# if structure = `separable` or `isotropic`
density = pc.smoothing.density_isotropic

# if structure = `covariant`
density = pc.smoothing.density_covariant
```

The separate density attributes are by design, so that users can compute the density using different structures and easily compare the results. See e.g. {doc}`this tutorial <../tutorials/deposition/walkthrough>` for a visual comparison.

:::{note}
Interpolation and density estimates only sum over the $N_{\rm ngb}$ nearest neighbours. Kernels with a support multiple $n > 1$ (e.g. `gaussian`, `quintic_spline`, `tsc` and the Wendland kernels) are therefore rescaled so that their support ends at the smoothing length ($h \rightarrow h/n$). Deposition instead uses the full footprint of radius $n \times h$.
:::

### Adding and deleting particle fields

The {py:class}`~smudgy.pointcloud.PointCloud` instance makes it easy to register additional particle fields that can be accessed as class attributes during later operations. The method to do this is {py:meth}`~smudgy.pointcloud.PointCloud.add_fields` and accepts either a single array or a list of names and arrays, given in the original particle order:

```python
temperature = np.ones(N)
pc.add_fields(names='temperature', values=temperature)

velocity = np.ones((N, 3))
pc.add_fields(names=['temperature', 'velocity'], values=[temperature, velocity])
```

Every field is set as a class attribute and can be accessed with its name identifier, e.g.

```python
temperature = pc.temperature
```

:::{important}
Always register per-particle fields via {py:meth}`~smudgy.pointcloud.PointCloud.add_fields` rather than by plain attribute assignment (`pc.temperature = temperature`): on several MPI ranks, `add_fields` splits the field across the ranks in the same way as the positions, while a plain assignment does not (`smudgy` warns if it detects this).
:::

Similarly, using {py:meth}`~smudgy.pointcloud.PointCloud.delete_fields` we can delete fields from the class instance via their attribute names:

```python
pc.delete_fields(names='temperature')               # single field
pc.delete_fields(names=['temperature', 'velocity']) # multiple fields
```

### Interpolation

Interpolating a particle field to a position $\mathbfit{x}$ follows the standard (scatter) SPH summation over the neighbouring particles $i \in \mathcal{N}$, with their masses $m_i$, densities $\rho_i$, smoothing lengths $h_i$ (or tensors $\mathbfss{H}_i$) and specific field values $f_i$:

$$
f(\mathbfit{x}) = \sum_{i \in \mathcal{N}} \frac{m_i}{\rho_i} f_i \, W(\mathbfit{x} - \mathbfit{x}_i, \, h_i).
$$

Since only the kernel depends on $\mathbfit{x}$, derivatives follow by differentiating $W$,

$$
\nabla f (\mathbfit{x}) = \sum_{i \in \mathcal{N}} \frac{m_i}{\rho_i} f_i \, \nabla W(\mathbfit{x} - \mathbfit{x}_i, \, h_i),
$$

and analogously for the divergence and curl of a vector field $\mathbfit{v}$, with $f_i \nabla W$ replaced by $\mathbfit{v}_i \cdot \nabla W$ and $\mathbfit{v}_i \times \nabla W$, respectively.

In `smudgy` these values can be computed using the {py:meth}`~smudgy.pointcloud.PointCloud.interpolate` method by passing the particle field ($f_i$) and the query positions ($\mathbfit{x}$),

```python
pc.add_fields('f', np.random.randn(N))
f_q = pc.interpolate(fields='f', query_positions=query_positions)  # shape = (M, 1)
```

:::{tip}
If we wish to interpolate particle fields (or their gradients) at the particle positions themselves, we can omit the `query_positions` argument, which then defaults to the particle positions.
:::

`smudgy` can also compute gradient-based quantities, i.e. scalar field gradients, and divergence and curl of vector fields, which is controlled by setting the `mode` keyword accordingly

```python
grad_f_q = pc.interpolate(fields='f', query_positions=query_positions,
                          mode='gradient')   # shape = (M, 1, d)

pc.add_fields('v', np.random.randn(N, 3))
div_v_q  = pc.interpolate(fields='v', query_positions=query_positions,
                          mode='divergence') # shape = (M, 1)
curl_v_q = pc.interpolate(fields='v', query_positions=query_positions,
                          mode='curl')       # shape = (M, 1, 3) in 3D
```

For a single mode, the method can also process multiple fields at once, passed by name or (on a single rank) directly as arrays:

```python
multi_f_q = pc.interpolate(fields=['f', 'temperature', 'velocity'])
```

### Deposition to structured grids

Grid deposition is a core operation in many modern analysis workflows that involve particle data. `smudgy` provides a flexible, modular implementation covering both standard *fixed-grid* (i.e. fixed kernel size) mass-assignment schemes and *adaptive* SPH kernels:

+ fixed-grid (`adaptive=False`) → the B-spline stencils `ngp`, `cic`, `tsc`, `bspline4`, `bspline5` and `bspline6` (see {doc}`kernels and structures <kernel_structures>`), whose width is set by the grid
+ `separable` → integrals can be computed analytically and are exact
+ `isotropic` or `covariant` → integrals are evaluated using numerical quadrature or pre-computed integral samples (see {doc}`numerical quadrature <numerical_quadrature>`)

For adaptive kernels, the code integrates each particle's kernel over all cells within the kernel support. For a cell and the set $\mathcal{N}$ of particles whose kernels overlap it,

$$
f_\mathrm{cell} = \sum_{i\in\mathcal N} m_i\, f_i \, I_{\mathrm{cell},i},
\qquad
I_{\mathrm{cell},i} = \int_{\Omega_\mathrm{cell}} W(\mathbfit{x}-\mathbfit{x}_i,\mathbfss{H}_i) \, \mathrm d\mathbfit{x},
$$

where $I_{\mathrm{cell},i}$ is the fraction of the kernel weight of particle $i$ within the cell domain $\Omega_\mathrm{cell}$. Since every kernel is normalized, the total deposited quantity is conserved exactly. Optionally, the result can be normalized to obtain a mass-weighted cell average:

$$
\bar f_\mathrm{cell} = \frac{\sum_{i} m_i\, f_i \, I_{\mathrm{cell},i}}{\sum_{i} m_i\, I_{\mathrm{cell},i}}.
$$

:::{important}
**Mass maps vs. averaged fields.** Every deposited field is multiplied by the particle weights $m_i$ internally, so fields are *specific* quantities (per unit mass/weight).

+ To deposit the mass itself (i.e. $f_i = 1$), pass `fields=None`, which returns the weights grid $\sum_i m_i I_{\mathrm{cell},i}$. Do **not** deposit the weights themselves as a field, which would yield $\sum_i m_i^2 I_{\mathrm{cell},i}$.
+ For an additive quantity such as the internal energy, deposit the specific energy with `averaged=False`.
+ For an intensive quantity such as the temperature, deposit it with `averaged=True` to obtain the mass-weighted cell average.
:::

The figure below illustrates a single particle (black), its kernel (blue), and the grid cells receiving contributions.

```{image} ../src/kernel_over_grid.png
:alt: Kernel illustration
:width: 60%
:align: center
```

The central method for this kind of operation is {py:meth}`~smudgy.pointcloud.PointCloud.deposit`. The most important arguments include:

+ `fields`: the fields to deposit. We can pass a single or a list of fields, by name or (on a single rank) directly as arrays, similar to {py:meth}`~smudgy.pointcloud.PointCloud.interpolate`, or `None` to deposit only the weights (e.g. a mass map).
+ `averaged`: defines which fields represent averaged and which additive quantities. E.g. when depositing temperatures we want the average in a cell, while energy is an additive quantity.
+ `gridnums`: numbers of grid cells to use in each spatial dimension. If a single number is passed, the code uses this number for every dimension, otherwise, a list of numbers for every axis must be passed.
+ `extent`: the domain `[[xmin, xmax], [ymin, ymax], ...]` covered by the grid. If `boxsize` is `None`, this argument must be provided. If provided, only particles within the given extent are deposited, which can be used to quickly generate images of subregions of the cloud.
+ `adaptive`: whether to use the adaptive smoothing data (`True`, default) or a fixed-grid B-spline stencil (`False`, no smoothing data required).

The following arguments are _optional_ but can be useful in certain situations:

+ `plane_projection`: list of axes spanning the projection plane. `[0, 1]` → 'xy', `[2, 1]` → 'zy', etc. Useful for projecting 3D particle clouds to 2D and depositing onto a 2D grid.
+ `return_weights`: whether to also return the weights grid $\sum_i m_i I_{\mathrm{cell},i}$.
+ `integration_method`: one of `'midpoint'`, `'trapezoidal'`, `'simpson'`. Determines which numerical quadrature to use when integrating kernels over grid cells. Only relevant for `isotropic` and `covariant`, since integrals of `separable` kernels are evaluated analytically.
+ `eta_crit`: parameter controlling {ref}`anti-aliasing <anti-aliasing>`.
+ `gather_to_root` and `root`: on several MPI ranks, the partial grids of all ranks are summed and, by default, returned on every rank. With `gather_to_root=True`, the grid is only delivered to rank `root` (other ranks receive `None`), which is cheaper when only one rank needs it.

An example workflow may look like the following example. For more detailed workflows refer to the tutorial sections.

```python
kwargs = dict(
    fields=['f', 'velocity', 'temperature'],
    averaged=[False, True, True],
    kernel_name='gaussian',
    return_weights=True,
)

# shapes: (5, 100, 128, 256), (100, 128, 256)
fields_grid, weights_grid = pc.deposit(
    **kwargs,
    gridnums=[100, 128, 256],
    structure='isotropic')

# projected onto the x-y plane, shapes: (5, 100, 128), (100, 128)
fields_grid, weights_grid = pc.deposit(
    **kwargs,
    gridnums=[100, 128],
    extent=[[0, boxsize / 2], [0, boxsize / 2], [0, boxsize / 2]],
    structure='covariant',
    plane_projection=[0, 1])

# mass map with a fixed-grid cloud-in-cell stencil, shape: (64, 64, 64)
mass_grid = pc.deposit(fields=None, averaged=False, gridnums=64,
                       adaptive=False, kernel_name='cic')
```

(collecting-results)=
## Collecting results

On a single rank, all per-particle results (smoothing lengths and tensors, densities, interpolated values) are ordinary arrays in the original particle (or query) order. On several MPI ranks, each rank only holds the results for its own particles (or query points), and they can be collected in the original order on a single rank:

```python
rho = pc.get('density_isotropic')                      # full array on rank 0, None elsewhere
f_p = pc.gather_particles(pc.interpolate('f'))         # results at the particle positions
f_q = pc.gather_queries(pc.interpolate('f', query_positions=query_positions))  # at query positions
```

These calls also work on a single rank, so scripts written this way run unmodified with any number of ranks. See {doc}`backends <backend>` for details.
