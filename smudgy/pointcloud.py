"""Core PointCloud class for particle-based computations."""

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from . import execution
from .backend.neighbors import (
    build_kdtree,
    coordinate_difference_with_pbc,
    query_kdtree,
)
from .backend.taichi import init as taichi_init
from .decomposition import (
    DecompositionInfo,
    QueryRouting,
    hilbert_partition_and_scatter,
    route_query_positions,
)
from .ghosts import GhostInfo, exchange_ghosts, push_to_ghosts
from .smooth import SmoothingInfo

STRUCTURES = ("separable", "isotropic", "covariant")
Structure = Literal["separable", "isotropic", "covariant"]

INTERPOLATION_MODES = ("field", "gradient", "divergence", "curl")
InterpolationMode = Literal["field", "gradient", "divergence", "curl"]


class PointCloud:
    """Represent a collection of particles for operations."""

    def __init__(
        self,
        positions: npt.NDArray[np.floating],
        weights: npt.NDArray[np.floating] | None = None,
        boxsize: float | Sequence[float] | None = None,
        verbose: bool = True,
        backend: str = "taichi",
        **kwargs,
    ) -> None:
        """Initialize a PointCloud container for particle-based operations.

        Parameters
        ----------
        positions : npt.NDArray[np.floating]
            Particle positions, shape (N, D).
        weights : npt.NDArray[np.floating] | None
            Particle weights (e.g. masses), shape (N,). If None, uniform weights are used.
        boxsize : float or Sequence[float], optional
            Periodic box size(s). If None, no periodicity is used.
        verbose : bool, default True
            Verbosity flag.
        backend : str, default "taichi"
            String to determine backend.
        **kwargs : dict
            Additional keyword arguments for backend initialization.

        """
        # Initialize MPI environment
        self.verbose = verbose
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        if verbose and self.rank == 0:
            print(f'[smudgy] Using {self.size} MPI rank{"s" if self.size > 1 else ""}')

        # Initialize backend (every rank needs its own device/runtime context)
        self.set_backend(backend, **kwargs)

        # Initialize point cloud (rank 0 only; other ranks receive a broadcast copy below)
        if self.rank == 0:
            dim = positions.shape[-1]
            assert dim in (
                1,
                2,
                3,
            ), f"Particle positions must be of shape (N, 1), (N, 2) or (N, 3) but found {positions.shape}"
            positions_resolved = positions.astype(np.float32)

            weights_resolved = (
                np.ones(positions_resolved.shape[0], dtype=np.float32)
                if weights is None
                else weights.astype(np.float32)
            )
            assert (
                weights_resolved.shape[0] == positions_resolved.shape[0]
            ), f"Shape mismatch: length of weights and positions must be the same but found: {weights_resolved.shape} and {positions_resolved.shape}"

            if boxsize is None:
                periodic_resolved = False
                boxsize_resolved = None
            else:
                periodic_resolved = True
                boxsize_arr = np.asarray(boxsize)
                if boxsize_arr.ndim == 0:
                    boxsize_resolved = np.repeat(boxsize_arr, dim)
                else:
                    assert boxsize_arr.shape == (
                        dim,
                    ), f"'boxsize' must be a scalar or have shape ({dim},), got {boxsize_arr.shape}"
                    boxsize_resolved = boxsize_arr

                # Wrap into [0, boxsize) canonically, in float32 (the dtype
                # `self.positions` is stored in): raw input isn't guaranteed
                # pre-wrapped, and even input that IS within [0, boxsize) in
                # float64 can round up to exactly (or past) boxsize once cast
                # to float32 above -- which scipy's periodic cKDTree (used
                # for the smoothing-length neighbor search, see
                # `_check_tree`) rejects outright ("some data is outside of
                # the periodic domain"). `boxsize_resolved` isn't itself
                # guaranteed float32 (e.g. a plain Python float given as
                # `boxsize` stays float64 through `np.asarray`), so it's
                # cast explicitly first -- np.mod would otherwise silently
                # upcast `positions_resolved` to float64. The `np.minimum`
                # step closes one more edge case np.mod alone doesn't fully
                # rule out: for a value extremely close to a multiple of
                # boxsize, the internal subtraction can itself round back up
                # to exactly boxsize, so this clips to the largest float32
                # strictly below it.
                boxsize_f32 = boxsize_resolved.astype(np.float32)
                positions_resolved = np.mod(positions_resolved, boxsize_f32)
                positions_resolved = np.minimum(
                    positions_resolved, np.nextafter(boxsize_f32, np.float32(0))
                ).astype(np.float32)
        else:
            dim = positions_resolved = weights_resolved = None
            periodic_resolved = boxsize_resolved = None

        if self.size > 1:
            dim, periodic_resolved, boxsize_resolved = execution._bcast(
                self.comm, (dim, periodic_resolved, boxsize_resolved)
            )
            positions_resolved = execution._bcast_array(self.comm, positions_resolved)
            weights_resolved = execution._bcast_array(self.comm, weights_resolved)

        self.dim = dim
        self.positions = positions_resolved
        self.weights = weights_resolved
        self.periodic = periodic_resolved
        self.boxsize = boxsize_resolved

        self.smoothing = SmoothingInfo()
        self.decomposition = DecompositionInfo()
        self.ghosts = GhostInfo()
        self.query_routing = QueryRouting()

        # Verbose output after completed initialization
        if self.verbose and self.rank == 0:
            periodic_str = (
                f"in periodic box of size={self.boxsize}"
                if self.periodic
                else "without periodicity"
            )
            print(
                f"[smudgy] Initialized {self.dim}d PointCloud with {self.positions.shape[0]} particles {periodic_str}"
            )

    # =============================================================================
    # Set utilities
    # =============================================================================
    def set_backend(self, backend: str = "taichi", **kwargs) -> None:
        """Set the backend for computations, one of [numpy, taichi]."""
        assert isinstance(
            backend, str
        ), f"'kernel_name' must be a string but found {type(backend)}"
        assert backend in [
            "numpy",
            "taichi",
        ], f"backend must be one of 'numpy' or 'taichi' but found {backend}"
        self.backend = backend
        if backend == "taichi":
            taichi_init(**kwargs)
        if self.verbose and getattr(self, "rank", 0) == 0:
            print(f"[smudgy] Set {backend} backend")

    def decompose(self) -> "PointCloud":
        """Compute a Hilbert-curve, particle-count-balanced spatial decomposition.

        Opt-in: does NOT change `self.positions`/`self.weights` or any
        existing method's behavior. Stores the result in `self.decomposition`
        (see `decomposition.DecompositionInfo`) as a side artifact for future
        use (ghost exchange, local-only compute, gather-to-root) once later
        steps in the domain-decomposition roadmap land. Custom fields added
        via `add_fields` are NOT decomposed by this method.

        Returns
        -------
        PointCloud
            self, for chaining (mirrors `global_setup`).

        """
        domain_min, domain_max = self._resolve_decomposition_domain()
        self.decomposition = hilbert_partition_and_scatter(
            self.comm,
            self.positions if self.rank == 0 else None,
            self.weights if self.rank == 0 else None,
            domain_min=domain_min,
            domain_max=domain_max,
            periodic=self.periodic,
        )
        if self.verbose and self.rank == 0:
            print(
                f"[smudgy] Decomposed {self.positions.shape[0]} particles across "
                f"{self.size} rank{'s' if self.size > 1 else ''} (Hilbert order, "
                "count-balanced)"
            )
        return self

    def _resolve_decomposition_domain(
        self,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Resolve the (domain_min, domain_max) extent for Hilbert quantization.

        Periodic: [0, boxsize] per axis -- `self.boxsize` is already
        identical on every rank (broadcast in `__init__`), so no new
        communication is needed. Non-periodic: the data's own bounding box,
        computed on rank 0 (the only rank required to have authoritative
        data, per `hilbert_partition_and_scatter`'s calling convention) and
        broadcast (a tiny payload: 2*dim floats).
        """
        if self.periodic:
            return np.zeros(self.dim, dtype=np.float32), np.asarray(
                self.boxsize, dtype=np.float32
            )

        domain = (
            (self.positions.min(axis=0), self.positions.max(axis=0))
            if self.rank == 0
            else None
        )
        if self.size > 1:
            domain = execution._bcast(self.comm, domain)
        return domain

    def find_neighbors(
        self,
        num_neighbors: int | None = None,
        max_iterations: int = 20,
        on_max_iterations: str = "raise",
    ) -> "PointCloud":
        """Ghost-particle exchange + iterative true-KNN solve.

        Opt-in: stores the result in `self.ghosts` (see `ghosts.GhostInfo`)
        without touching `self.positions`/`self.weights`/`self.smoothing` or
        any existing method's behavior -- mirrors `decompose()`'s contract
        exactly. `compute_smoothing` is NOT rewired to use this (a later
        domain-decomposition-roadmap step's job).

        Requires `.decompose()` to have already been called.

        Returns
        -------
        PointCloud
            self, for chaining (mirrors `global_setup`/`decompose`).

        """
        self._check_decomposed()
        num_neighbors_temp = self._resolve_num_neighbors(num_neighbors)
        self.ghosts = exchange_ghosts(
            self.comm,
            self.decomposition,
            num_neighbors_temp,
            self.dim,
            self.periodic,
            self.boxsize,
            max_iterations=max_iterations,
            on_max_iterations=on_max_iterations,
        )
        if self.verbose and self.rank == 0:
            print(
                f"[smudgy] Found neighbors via ghost exchange "
                f"({num_neighbors_temp} neighbors per particle)"
            )
        return self

    def gather_particles(
        self, local_array: npt.NDArray[Any], root: int = 0
    ) -> npt.NDArray[Any] | None:
        """Gather a particle-indexed local array back onto `root` only, in
        original particle order (Step 5 of the domain-decomposition roadmap).

        `local_array` must be indexed the way `decomposition.local_*` is
        (row i = this rank's i-th local particle) -- e.g. `smoothing.
        smoothing_lengths`/`smoothing_tensors`, `smoothing.density_isotropic`/
        `density_covariant`, or an `interpolate()` (no `query_positions`)
        result, all produced via the local+ghost path Step 4a wired up.
        Requires `.decompose()` to have already been called. Cheaper than
        gathering the equivalent old-path result: this ships the reassembled
        array to `root` only, not to every rank (see `execution._gather_to_root`).

        Returns
        -------
        np.ndarray, shape (n_particles, *local_array.shape[1:]), on `root`;
        `None` on every other rank.
        """
        self._check_decomposed()
        local_global_indices = self.decomposition.local_global_indices
        if local_array.shape[0] != local_global_indices.shape[0]:
            raise ValueError(
                f"'local_array' has {local_array.shape[0]} rows but this rank "
                f"has {local_global_indices.shape[0]} local particles; "
                "'local_array' must be a local+ghost-path result (indexed the "
                "same way as decomposition.local_positions), not a full-N "
                "array from the old (full-replication) path."
            )
        return execution._gather_to_root(
            self.comm, local_global_indices, local_array,
            self.positions.shape[0], root=root,
        )

    def gather_queries(
        self, local_array: npt.NDArray[Any], root: int = 0
    ) -> npt.NDArray[Any] | None:
        """Gather a query-position-indexed local array back onto `root` only,
        in the original query-array order (Step 5 of the domain-decomposition
        roadmap).

        `local_array` must be an `interpolate(query_positions=...)` result
        produced via Step 4b's local+ghost path -- indexed the way
        `query_routing.local_positions` is (row i = this rank's i-th routed
        query position). Requires that call to have populated
        `self.query_routing` (i.e. `used_ghosts` was True and query
        positions were given).

        Returns
        -------
        np.ndarray, shape (n_query_positions, *local_array.shape[1:]), on
        `root`; `None` on every other rank.
        """
        local_global_indices = self.query_routing.local_global_indices
        if local_global_indices is None:
            raise AttributeError(
                "No query-position routing available yet; call 'interpolate' "
                "with 'query_positions' set (with the local+ghost path active, "
                "i.e. after 'decompose'/'find_neighbors'/'compute_smoothing') "
                "first."
            )
        if local_array.shape[0] != local_global_indices.shape[0]:
            raise ValueError(
                f"'local_array' has {local_array.shape[0]} rows but this rank "
                f"has {local_global_indices.shape[0]} local query positions; "
                "'local_array' must be the result of the most recent "
                "'interpolate(query_positions=...)' call."
            )
        n_total = int(self.query_routing.counts.sum())
        return execution._gather_to_root(
            self.comm, local_global_indices, local_array, n_total, root=root,
        )

    def _set_property(self, name: str, value: Any) -> None:
        """Set a global property with centralized validation."""
        if name == "structure":
            if not isinstance(value, str):
                raise TypeError(f"'structure' must be a string but found {type(value)}")
            if value not in STRUCTURES:
                raise ValueError(
                    f"'structure' must be one of {STRUCTURES}, but found '{value}'"
                )
        elif name == "kernel_name":
            if not isinstance(value, str):
                raise TypeError(
                    f"'kernel_name' must be a string but found {type(value)}"
                )
        elif name == "num_neighbors":
            if not isinstance(value, int):
                raise TypeError(
                    f"'num_neighbors' must be an int but found {type(value)}"
                )
            if value <= 0:
                raise ValueError(
                    f"'num_neighbors' must be a positive integer but found {value}"
                )
        else:
            raise ValueError(
                f"Unsupported property '{name}'; expected one of ('structure', 'kernel_name', 'num_neighbors')"
            )

        setattr(self, name, value)

        if self.verbose:
            message = {
                "structure": f"[smudgy] Set structure to '{value}'",
                "kernel_name": f"[smudgy] Set kernel to '{value}'",
                "num_neighbors": f"[smudgy] Set number of neighbors to {value}",
            }[name]
            print(message)

    # =============================================================================
    # Check utilities
    # =============================================================================
    def _check_property(self, property: str | None = None) -> None:
        """Ensure a global property is set before use."""
        if not hasattr(self, property):
            raise AttributeError(
                f"'{property}' has not been set: either set it via the 'global_setup' method or provide it as a function argument"
            )

    def _check_neighbors(
        self, nn_inds: npt.NDArray[np.integer], n_valid: int | None = None
    ) -> None:
        """Verify that neighbor indices are within valid bounds.

        `n_valid` defaults to `self.positions.shape[0]` (the full-replication
        path's convention); the local+ghost path passes the local+ghost
        combined count instead, since its `nn_inds` values are indices into
        that smaller combined array, not into `self.positions`.
        """
        if n_valid is None:
            n_valid = self.positions.shape[0]
        max_idx = np.max(nn_inds)
        if max_idx >= n_valid:
            raise IndexError(
                f"Neighbor index {max_idx} is out of bounds for {n_valid} particles. This indicates a bug in the neighbor search or input setup."
            )

    def _check_decomposed(self) -> None:
        """Verify that a spatial decomposition has been computed."""
        if self.decomposition.local_positions is None:
            raise AttributeError(
                "No spatial decomposition has been computed yet; call 'decompose' first."
            )

    def _using_decomposition(self) -> bool:
        """Whether `.decompose()` and `.find_neighbors()` have both been
        called, so the local+ghost data path can be used instead of the
        full-replication path."""
        return (
            self.decomposition.local_positions is not None
            and self.ghosts.nn_inds is not None
        )

    def _combined_local_and_ghost(
        self, local_arr: npt.NDArray[Any], ghost_arr: npt.NDArray[Any]
    ) -> npt.NDArray[Any]:
        """Concatenate a rank's local array with its imported ghosts',
        local rows first -- the ordering `GhostInfo.nn_inds` assumes (a
        value < n_local indexes `local_arr`, >= n_local indexes `ghost_arr`
        at `value - n_local`). Named so this invariant lives in one place
        instead of being repeated by hand at every call site.
        """
        return np.concatenate([local_arr, ghost_arr], axis=0)

    def _check_density_computed(self, structure: Structure) -> None:
        """Verify that density has been computed for the requested structure."""
        field = (
            self.smoothing.density_isotropic
            if structure == "isotropic"
            else self.smoothing.density_covariant
        )
        if field is None:
            raise AttributeError(
                f"Particle density has not been computed yet for structure '{structure}'; call 'compute_density' with 'structure={structure}' first."
            )

    def _check_smoothing_computed(self, structure: Structure) -> None:
        """Verify that smoothing lengths/tensors have been computed for the structure."""
        attr_map = {
            "separable": "smoothing_lengths",
            "isotropic": "smoothing_lengths",
            "covariant": "smoothing_tensors",
        }
        if getattr(self.smoothing, attr_map[structure]) is None:
            raise AttributeError(
                f"Smoothing {attr_map[structure]} has not been computed yet for structure '{structure}'; call 'compute_smoothing' with 'structure={structure}' first."
            )

    def _check_shape(self, arr: npt.NDArray[Any], name: str) -> None:
        """Ensure the first dimension of an array matches the number of particles."""
        if arr.shape[0] != self.positions.shape[0]:
            raise ValueError(
                f"Length of '{name}' ({arr.shape[0]}) must match number of points ({self.positions.shape[0]})"
            )

    def _check_field_dimensionality(
        self,
        mode: InterpolationMode,
        field_sizes: list[int],
    ) -> None:

        # map which fields are vector fields
        field_types = np.array([size > 1 for size in field_sizes], dtype=np.bool_)

        """Check that field dimensions match the requested operation.

        Parameters
        ----------
        mode : InterpolationMode
            The interpolation mode ('field', 'gradient', 'divergence', 'curl').
        field_sizes : list[int]
            List of component counts for each field.
        field_types : npt.NDArray[np.bool_]
            Boolean array indicating vector fields (True) vs scalar fields (False).

        Raises
        ------
        ValueError
            If field type is incompatible with the requested operation.

        """
        if mode == "gradient":
            # Gradients work on both scalars and vectors, but we compute per component
            pass
        elif mode == "divergence":
            # Divergence requires vector fields with exactly D components (spatial dimension)
            for i, (is_vec, size) in enumerate(zip(field_types, field_sizes)):
                if not is_vec:
                    raise ValueError(
                        f"divergence requires vector fields (≥2 components), but field {i} has {size} component(s)"
                    )
                if size != self.dim:
                    raise ValueError(
                        f"divergence requires vector fields with {self.dim} components (spatial dimension), but field {i} has {size} component(s)"
                    )
        elif mode == "curl":
            # Curl requires vector fields with exactly D components (spatial dimension)
            for i, (is_vec, size) in enumerate(zip(field_types, field_sizes)):
                if not is_vec:
                    raise ValueError(
                        f"curl requires vector fields (≥2 components), but field {i} has {size} component(s)"
                    )
                if size != self.dim:
                    raise ValueError(
                        f"curl requires vector fields with {self.dim} components (spatial dimension), but field {i} has {size} component(s)"
                    )

    # =============================================================================
    # Resolve utilities
    # =============================================================================
    def _resolve_structure(self, structure: Structure | None) -> Structure:
        """Resolve the smoothing structure from argument or global state."""
        if structure is None:
            self._check_property("structure")
            return self.structure
        if structure not in STRUCTURES:
            raise ValueError(
                f"'structure' must be one of {STRUCTURES}, but found '{structure}'"
            )
        return structure

    def _resolve_kernel(self, kernel_name: str | None = None) -> str:
        """Resolve the kernel name from argument or global state."""
        if kernel_name is None:
            self._check_property("kernel_name")
            return self.kernel_name
        assert isinstance(
            kernel_name, str
        ), f"'kernel_name' must be a string but found {type(kernel_name)}"
        return kernel_name

    def _resolve_num_neighbors(self, num_neighbors: int | None = None) -> int:
        """Resolve the number of neighbors from argument or global state."""
        if num_neighbors is None:
            self._check_property("num_neighbors")
            return self.num_neighbors
        assert (
            isinstance(num_neighbors, int) and num_neighbors > 0
        ), f"'num_neighbors' must be a positive integer but found {num_neighbors}"
        return num_neighbors

    def _resolve_fields(
        self, fields: npt.ArrayLike | str | list[str]
    ) -> tuple[npt.NDArray[np.floating], list[int]]:
        """Resolve fields to a single array and return component counts."""
        if isinstance(fields, (str, np.ndarray)):
            fields = [fields]
        elif not isinstance(fields, list):
            raise ValueError(
                "Invalid 'fields' argument: must be a string, list, or numpy array"
            )

        arrays = []
        field_sizes = []

        for f in fields:
            arr = getattr(self, f) if isinstance(f, str) else np.asarray(f)
            arr = np.atleast_2d(arr).T if arr.ndim == 1 else np.atleast_2d(arr)

            arrays.append(arr)
            field_sizes.append(arr.shape[1])

        for i, arr in enumerate(arrays):
            self._check_shape(arr, f"field {i}")

        return np.concatenate(arrays, axis=-1), field_sizes

    def _resolve_deposition_options(
        self,
        adaptive: bool,
        kernel_name: str | None,
        structure: Structure | None,
    ) -> tuple[str, Structure | None]:
        """Resolve the two mutually exclusive deposition workflows.

        Fixed-grid deposition selects a stencil directly via ``kernel_name``.
        Adaptive deposition selects a smoothing ``structure`` and then uses
        ``kernel_name`` from that structure's kernel family.
        """
        kernel_name = self._resolve_kernel(kernel_name)
        if not adaptive:
            if structure is not None:
                raise ValueError("structure is only valid when adaptive=True.")
            return kernel_name, None

        return kernel_name, self._resolve_structure(structure)

    def _resolve_averaged(
        self,
        averaged: bool | Sequence[bool],
        field_sizes: Sequence[int],
    ) -> npt.NDArray[np.bool_]:
        """Resolve averaged flags to one bool per scalar field component."""
        total_components = sum(field_sizes)

        # Single bool -> broadcast everywhere
        if isinstance(averaged, bool):
            return np.full(total_components, averaged, dtype=np.bool_)

        if not isinstance(averaged, (list, tuple)):
            raise TypeError(
                f"'averaged' must be a bool or sequence of bools, found {type(averaged)}"
            )

        # Case 1:
        # User provided one bool per scalar component already
        if len(averaged) == total_components:
            return np.asarray(averaged, dtype=np.bool_)

        # Case 2:
        # User provided one bool per field -> broadcast components
        if len(averaged) == len(field_sizes):
            resolved = []
            for avg, size in zip(averaged, field_sizes):
                resolved.extend([avg] * size)

            return np.asarray(resolved, dtype=np.bool_)

        raise ValueError(
            f"Length of 'averaged' ({len(averaged)}) must match either "
            f"the number of fields ({len(field_sizes)}) or the total number "
            f"of scalar components ({total_components})"
        )

    def _resolve_gridnums(
        self, gridnums: int | Sequence[int], dim: int
    ) -> npt.NDArray[np.int_]:
        gn = np.atleast_1d(gridnums).astype(np.int32)
        if gn.size == 1:
            gn = np.repeat(gn, dim)
        if gn.size != dim:
            raise ValueError(
                f"Length of 'gridnums' ({gn.size}) must match deposition dimension ({dim})"
            )
        return gn

    # =============================================================================
    # Tree utilities
    # =============================================================================
    def _check_tree(self) -> Any:
        """Ensure a kd-tree exists for neighbor searches.

        Only built on rank 0 (parallel/distributed kd-tree construction is
        deferred); returns ``None`` on non-root ranks. Callers must only
        dereference the returned tree inside an ``if self.rank == 0`` block.
        """
        if self.smoothing.tree is None and self.rank == 0:
            if self.verbose:
                print("[smudgy] Building kd-tree from positions")
            tree = build_kdtree(self.positions, boxsize=self.boxsize)
            self.smoothing.tree = tree
        return self.smoothing.tree

    # =============================================================================
    # Preparation / helper methods
    # =============================================================================
    def global_setup(
        self,
        structure: Structure | None = None,
        kernel_name: str | None = None,
        num_neighbors: int | None = None,
        backend: str | None = None,
        **kwargs,
    ) -> "PointCloud":
        """Set global parameters for computations.

        Parameters
        ----------
        structure : Structure, optional
            Smoothing structure ('separable', 'isotropic', or 'covariant').
        kernel_name : str, optional
            Name of the SPH kernel.
        num_neighbors : int, optional
            Number of neighbors for smoothing length computation.
        backend : str, optional
            Backend to use for computations ('numpy' or 'taichi').
        kwargs : dict
            Additional keyword arguments for backend initialization.

        Returns
        -------
        PointCloud
            The current instance for method chaining.

        """
        if structure:
            self._set_property("structure", structure)

        if kernel_name:
            self._set_property("kernel_name", kernel_name)

        if num_neighbors:
            self._set_property("num_neighbors", num_neighbors)

        if backend:
            self.set_backend(backend, **kwargs)

        return self

    def compute_smoothing(
        self,
        query_positions: npt.ArrayLike | None = None,
        num_neighbors: int | None = None,
        structure: Structure | None = None,
        # backend: str = None,
    ) -> None:
        """Compute smoothing lengths/tensors for SPH calculations.

        Parameters
        ----------
        query_positions : npt.ArrayLike, optional
            Positions where smoothing is evaluated. If None, uses particle positions.
        num_neighbors : int, optional
            Number of neighbors for smoothing length computation.
        structure : Structure, optional
            Smoothing structure for computation.
        backend : {"numpy", "taichi"}, default "numpy"
            Backend used for the smoothing computation.

        Returns
        -------
        None

        Notes
        -----
        Results are stored in the ``smoothing`` attribute.

        """
        num_neighbors_temp = self._resolve_num_neighbors(num_neighbors)
        structure_temp = self._resolve_structure(structure)

        # Local+ghost path: only when a decomposition AND ghost exchange are
        # available, no custom query_positions were given (arbitrary query
        # positions aren't decomposed yet -- deferred), and the requested
        # num_neighbors matches what find_neighbors() actually solved for
        # (self.ghosts.nn_inds has that k baked in; a mismatch can't be
        # silently reused, so falls back to the full path below instead).
        use_ghosts = (
            self._using_decomposition()
            and query_positions is None
            and self.ghosts.nn_inds.shape[1] == num_neighbors_temp
        )
        if use_ghosts:
            if self.verbose and self.rank == 0:
                info_str = "tensors" if structure_temp == "covariant" else "lengths"
                print(
                    f"[smudgy] Computing smoothing {info_str} from "
                    f"{num_neighbors_temp} neighbors (local+ghost)"
                )

            nn_inds_local = self.ghosts.nn_inds
            nn_dists_local = self.ghosts.nn_dists
            combined_positions = self._combined_local_and_ghost(
                self.decomposition.local_positions, self.ghosts.ghost_positions
            )
            combined_weights = self._combined_local_and_ghost(
                self.decomposition.local_weights, self.ghosts.ghost_weights
            )

            if structure_temp in ("separable", "isotropic"):
                self.smoothing.smoothing_lengths = execution._dispatch(
                    "compute_hsml",
                    backend=self.backend,
                    nn_dists=nn_dists_local,
                    reduce=False,
                )
            else:
                (
                    smoothing_tensors,
                    smoothing_tensors_eigvals,
                    smoothing_tensors_eigvecs,
                    nn_dists_vec,
                ) = execution._dispatch(
                    "compute_hmat",
                    backend=self.backend,
                    query_positions=self.decomposition.local_positions,
                    neighbor_positions=combined_positions[nn_inds_local],
                    neighbor_weights=combined_weights[nn_inds_local],
                    boxsize=self.boxsize,
                    reduce=False,
                )
                self.smoothing.smoothing_tensors = smoothing_tensors
                self.smoothing.smoothing_tensors_eigvals = smoothing_tensors_eigvals
                self.smoothing.smoothing_tensors_eigvecs = smoothing_tensors_eigvecs
                self.smoothing.nn_dists_vec = nn_dists_vec

            self.smoothing.nn_inds, self.smoothing.nn_dists = (
                nn_inds_local,
                nn_dists_local,
            )
            self.smoothing.num_neighbors = num_neighbors_temp
            self.smoothing.used_ghosts = True
            self._check_neighbors(nn_inds_local, n_valid=combined_positions.shape[0])
            return

        self.smoothing.used_ghosts = False

        # query_positions is only ever read by rank 0 below (to compute qpos,
        # which then gets broadcast) -- no need to broadcast the raw input.

        # kd-tree build + neighbor search: rank 0 only (deferred: parallel kd-tree)
        if self.rank == 0:
            tree = self._check_tree()
            qpos = (
                self.positions
                if query_positions is None
                else np.asarray(query_positions, dtype=np.float32)
            )
            nn_dists, nn_inds = query_kdtree(tree, qpos, k=num_neighbors_temp)
        else:
            qpos = nn_dists = nn_inds = None

        if self.size > 1:
            qpos = execution._bcast_array(self.comm, qpos)
            nn_dists = execution._bcast_array(self.comm, nn_dists)
            nn_inds = execution._bcast_array(self.comm, nn_inds)

        if self.verbose and self.rank == 0:
            info_str = "tensors" if structure_temp == "covariant" else "lengths"
            print(
                f"[smudgy] Computing smoothing {info_str} from {num_neighbors_temp} neighbors"
            )

        # Slice to this rank's local rows *before* gathering neighbor data,
        # so the expensive fancy-indexing below is O(n/size), not O(n), per
        # rank (see execution._local_slice docstring).
        start, stop = execution._local_slice(qpos.shape[0], self.rank, self.size)
        nn_inds_local = nn_inds[start:stop]

        if structure_temp in ("separable", "isotropic"):
            self.smoothing.smoothing_lengths = execution._dispatch(
                "compute_hsml",
                backend=self.backend,
                nn_dists=nn_dists[start:stop],
            )
        else:
            (
                smoothing_tensors,
                smoothing_tensors_eigvals,
                smoothing_tensors_eigvecs,
                nn_dists_vec,
            ) = execution._dispatch(
                "compute_hmat",
                backend=self.backend,
                query_positions=qpos[start:stop],
                neighbor_positions=self.positions[nn_inds_local],
                neighbor_weights=self.weights[nn_inds_local],
                boxsize=self.boxsize,
            )
            self.smoothing.smoothing_tensors = smoothing_tensors
            self.smoothing.smoothing_tensors_eigvals = smoothing_tensors_eigvals
            self.smoothing.smoothing_tensors_eigvecs = smoothing_tensors_eigvecs
            self.smoothing.nn_dists_vec = nn_dists_vec

        self.smoothing.nn_inds, self.smoothing.nn_dists = nn_inds, nn_dists
        self.smoothing.num_neighbors = num_neighbors_temp
        self._check_neighbors(nn_inds)

    def compute_density(
        self,
        kernel_name: str | None = None,
        structure: Structure | None = None,
    ) -> None:
        """Compute particle densities using SPH kernels.

        Parameters
        ----------
        kernel_name : str, optional
            Name of the SPH kernel to use for density computation.
        structure : Structure, optional
            Smoothing structure specifier.

        Notes
        -----
        Results are stored in the `smoothing` attribute.

        """
        st = self._resolve_structure(structure)
        kn = self._resolve_kernel(kernel_name)
        self._check_smoothing_computed(st)

        if self.verbose and self.rank == 0:
            print(f"[smudgy] Computing density using " f"{st} '{kn}' kernel")

        if self.smoothing.used_ghosts:
            # self.smoothing.nn_inds/smoothing_lengths/smoothing_tensors are
            # already local-sized here (compute_smoothing's local+ghost
            # path), so no _local_slice chunking is needed at all -- an
            # already-local array doesn't need re-chunking.
            nn_inds_local = self.smoothing.nn_inds
            combined_positions = self._combined_local_and_ghost(
                self.decomposition.local_positions, self.ghosts.ghost_positions
            )
            combined_weights = self._combined_local_and_ghost(
                self.decomposition.local_weights, self.ghosts.ghost_weights
            )
            density = execution._dispatch(
                "compute_density",
                backend=self.backend,
                kernel_name=kn,
                dim=self.dim,
                neighbor_weights=combined_weights[nn_inds_local],
                r_ij=self._get_rel_coords(
                    self.decomposition.local_positions,
                    combined_positions[nn_inds_local],
                ),
                h=(
                    self.smoothing.smoothing_tensors
                    if st == "covariant"
                    else self.smoothing.smoothing_lengths
                ),
                structure=st,
                reduce=False,
            )
        else:
            # Slice to this rank's local particles *before* gathering neighbor
            # data, so the expensive fancy-indexing below is O(n/size), not
            # O(n), per rank (see execution._local_slice docstring).
            start, stop = execution._local_slice(
                self.smoothing.nn_inds.shape[0], self.rank, self.size
            )
            nn_inds_local = self.smoothing.nn_inds[start:stop]

            density = execution._dispatch(
                "compute_density",
                backend=self.backend,
                kernel_name=kn,
                dim=self.dim,
                neighbor_weights=self.weights[nn_inds_local],
                r_ij=self._get_rel_coords(
                    self.positions[start:stop], self.positions[nn_inds_local]
                ),
                h=(
                    self.smoothing.smoothing_tensors[start:stop]
                    if st == "covariant"
                    else self.smoothing.smoothing_lengths[start:stop]
                ),
                structure=st,
            )

        self.smoothing.kernel_name = kn

        if st == "covariant":
            self.smoothing.density_covariant = density
        else:
            self.smoothing.density_isotropic = density

    def add_fields(
        self, names: str | list[str], values: npt.ArrayLike | list[npt.ArrayLike]
    ) -> None:
        """Add one or multiple custom fields to the PointCloud instance.

        Parameters
        ----------
        names : str or list of str
            Name(s) of the field(s) to add.
        values : array_like or list of array_like
            Field values. Each array must have shape (N,) or (N, num_components).

        """
        # --- Case 1: multiple fields ---
        if isinstance(names, (list, tuple)) or isinstance(values, (list, tuple)):
            if not (
                isinstance(names, (list, tuple)) and isinstance(values, (list, tuple))
            ):
                raise ValueError(
                    "If passing multiple fields, both 'names' and 'values' must be lists/tuples."
                )

            if len(names) != len(values):
                raise ValueError("'names' and 'values' must have the same length.")

            for name, val in zip(names, values):
                self.add_fields(name, val)  # recursive call

            return

        # --- Case 2: single field ---
        name = names
        # caller-supplied field values are only authoritative on rank 0
        values_arr = (
            np.asarray(values, dtype=np.float32) if self.rank == 0 else None
        )
        if self.size > 1:
            values_arr = execution._bcast_array(self.comm, values_arr)

        if self.rank == 0 and hasattr(self, name):
            print(f"Overwriting existing attribute '{name}' on PointCloud instance.")

        self._check_shape(values_arr, name)
        setattr(self, name, values_arr)

    def delete_fields(self, names: str | list[str]) -> None:
        """Delete one or multiple custom fields from the PointCloud instance.

        Parameters
        ----------
        names : str or list of str
            Name(s) of the field(s) to delete.

        """
        # --- Case 1: multiple fields ---
        if isinstance(names, (list, tuple)):
            for name in names:
                self.delete_fields(name)  # recursive call
            return

        # --- Case 2: single field ---
        name = names

        if hasattr(self, name):
            delattr(self, name)
        else:
            print(
                f"No attribute named '{name}' found on PointCloud instance to delete."
            )

    def set_smoothing(
        self,
        structure: Structure | None = None,
        smoothing_lengths: npt.ArrayLike | None = None,
        smoothing_tensors: npt.ArrayLike | None = None,
        smoothing_tensors_eigvals: npt.ArrayLike | None = None,
        smoothing_tensors_eigvecs: npt.ArrayLike | None = None,
    ) -> None:
        """Manually assign smoothing lengths or tensors to particles.

        Parameters
        ----------
        structure : Structure, optional
            Smoothing structure. Required if setting `smoothing_lengths` or `smoothing_tensors`.
        smoothing_lengths : npt.ArrayLike, optional
            Isotropic smoothing lengths, shape (N,).
        smoothing_tensors : npt.ArrayLike, optional
            Anisotropic smoothing tensors, shape (N, D, D).
        smoothing_tensors_eigvals : npt.ArrayLike, optional
            Eigenvalues of the smoothing tensors, shape (N, D).
        smoothing_tensors_eigvecs : npt.ArrayLike, optional
            Eigenvectors of the smoothing tensors, shape (N, D, D).

        """
        # for smoothing_lengths, structure must be set and either 'separable' or 'isotropic'
        if smoothing_lengths:
            assert (
                structure == "isotropic"
            ), "Structure must be specified when providing 'smoothing_lengths'"
            self._check_shape(np.asarray(smoothing_lengths), "smoothing_lengths")
            self.smoothing.smoothing_lengths = np.asarray(
                smoothing_lengths, dtype=np.float32
            )

        if smoothing_tensors:
            assert (
                structure == "covariant"
            ), "Structure must be specified when providing 'smoothing_tensors'"
            self._check_shape(np.asarray(smoothing_tensors), "smoothing_tensors")
            self.smoothing.smoothing_tensors = np.asarray(
                smoothing_tensors, dtype=np.float32
            )

        if smoothing_tensors_eigvals:
            self._check_shape(
                np.asarray(smoothing_tensors_eigvals), "smoothing_tensors_eigvals"
            )
            self.smoothing.smoothing_tensors_eigvals = np.asarray(
                smoothing_tensors_eigvals, dtype=np.float32
            )

        if smoothing_tensors_eigvecs:
            self._check_shape(
                np.asarray(smoothing_tensors_eigvecs), "smoothing_tensors_eigvecs"
            )
            self.smoothing.smoothing_tensors_eigvecs = np.asarray(
                smoothing_tensors_eigvecs, dtype=np.float32
            )

    def _get_rel_coords(
        self,
        query_positions: npt.NDArray[np.floating],
        positions: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute relative coordinates between particles and query positions, respecting PBC."""
        if self.periodic:
            return coordinate_difference_with_pbc(
                query_positions[:, np.newaxis, :],
                positions,
                self.boxsize,
            )
        return query_positions[:, np.newaxis, :] - positions

    def _detect_field_types(self, field_sizes: list[int]) -> npt.NDArray[np.bool_]:
        """Detect whether each field is scalar (1 component) or vector (≥2 components).

        Parameters
        ----------
        field_sizes : list[int]
            List of component counts for each field.

        Returns
        -------
        npt.NDArray[np.bool_]
            Boolean array where True indicates a vector field.

        """
        return np.array([size > 1 for size in field_sizes], dtype=np.bool_)

    def _prepare_deposition_smoothing(
        self,
        structure: Structure,
        idx: npt.NDArray[np.integer],
        dim: int,
        plane_projection: list[int] | None = None,
    ) -> tuple[
        npt.NDArray[np.float32] | None,
        npt.NDArray[np.float32] | None,
        npt.NDArray[np.float32] | None,
    ]:
        """Prepare smoothing data (h, h_vals, h_vecs) for an adaptive deposition call.

        `idx` is this rank's already-local set of particle indices (see
        `deposit`), not a full-size boolean mask -- keeps this gather at
        O(n/size) instead of O(n) per rank.
        """
        self._check_smoothing_computed(
            "isotropic" if structure == "separable" else structure
        )

        if structure == "separable":
            hsml = self.smoothing.smoothing_lengths[idx]
            return np.repeat(hsml[:, np.newaxis], dim, axis=1), None, None

        if structure == "isotropic":
            return self.smoothing.smoothing_lengths[idx], None, None

        if structure == "covariant":
            if plane_projection is not None:
                # reduce=False: this result is immediately consumed locally
                # by `deposit`'s own dispatch call below, not surfaced to
                # the caller, so there's nothing to gather here.
                _, vals, vecs = execution._dispatch(
                    "project_2d",
                    backend=self.backend,
                    reduce=False,
                    h_tensor=self.smoothing.smoothing_tensors[idx],
                    plane=plane_projection,
                )
            else:
                vals = self.smoothing.smoothing_tensors_eigvals[idx]
                vecs = self.smoothing.smoothing_tensors_eigvecs[idx]
            return None, vals, vecs

        raise ValueError(f"Unsupported deposition structure '{structure}'")

    def _interpolate_local(
        self,
        query_positions_local: npt.NDArray[np.floating],
        ghost_info: GhostInfo,
        fields: npt.NDArray[np.floating],
        fields_sizes: list[int],
        mode: InterpolationMode,
        structure_temp: Structure,
    ) -> npt.NDArray[np.floating]:
        """Shared body of `interpolate()`'s two local+ghost branches.

        SPH-interpolates `fields` at `query_positions_local` using
        `ghost_info.nn_inds`/`nn_dists` (already solved -- either by
        `find_neighbors()` for the particle-position case, or by a fresh
        `ghosts.exchange_ghosts(..., target_positions=...)` call for the
        arbitrary-query-position case, Step 4b) against this rank's own
        local particles plus `ghost_info`'s imported ghosts.
        `ghost_info.export_local_index`/`export_counts` must correspond to
        the exact same exchange that produced `nn_inds` -- used here via
        `push_to_ghosts` to fetch each neighbor's already-computed
        density/smoothing data from whichever rank actually computed it.
        Returns a local-sized (not full-N/M, not gathered) array; the caller
        is responsible for knowing how to reassemble it (see `interpolate`).
        """
        nn_inds_local = ghost_info.nn_inds
        combined_positions = self._combined_local_and_ghost(
            self.decomposition.local_positions, ghost_info.ghost_positions
        )
        combined_weights = self._combined_local_and_ghost(
            self.decomposition.local_weights, ghost_info.ghost_weights
        )

        r_ij = self._get_rel_coords(
            query_positions_local, combined_positions[nn_inds_local]
        )

        fields_local = fields[self.decomposition.local_global_indices]
        ghost_fields = push_to_ghosts(self.comm, ghost_info, fields_local)
        combined_fields = self._combined_local_and_ghost(fields_local, ghost_fields)
        fields_temp = combined_fields[nn_inds_local]

        if mode in ("divergence", "curl"):
            num_fields = len(fields_sizes)
            fields_temp = fields_temp.reshape(
                fields_temp.shape[0], fields_temp.shape[1], num_fields, self.dim
            )

        density_local = (
            self.smoothing.density_covariant
            if structure_temp == "covariant"
            else self.smoothing.density_isotropic
        )
        ghost_density = push_to_ghosts(self.comm, ghost_info, density_local)
        combined_density = self._combined_local_and_ghost(density_local, ghost_density)
        density_temp = combined_density[nn_inds_local]

        weights_temp = combined_weights[nn_inds_local] / (density_temp + 1e-8)

        h_local = (
            self.smoothing.smoothing_tensors
            if structure_temp == "covariant"
            else self.smoothing.smoothing_lengths
        )
        ghost_h = push_to_ghosts(self.comm, ghost_info, h_local)
        combined_h = self._combined_local_and_ghost(h_local, ghost_h)
        h_temp = combined_h[nn_inds_local]

        return execution._dispatch(
            "interpolate",
            backend=self.backend,
            kernel_name=self.smoothing.kernel_name,
            dim=self.dim,
            fields=fields_temp,
            weights=weights_temp,
            r_ij=r_ij,
            h=h_temp,
            mode=mode,
            structure=structure_temp,
            reduce=False,
        )

    # =============================================================================
    # Core methods
    # =============================================================================
    def interpolate(
        self,
        fields: npt.ArrayLike | str | list[str],
        query_positions: npt.ArrayLike = None,
        mode: InterpolationMode = "field",
        structure: Structure | None = None,
    ) -> npt.NDArray[np.floating]:
        r"""Interpolate particle fields to query positions using SPH.

        Compute interpolated field values, gradients, divergence, or curl at query positions
        using smoothed particle hydrodynamics (SPH).

        Parameters
        ----------
        fields : Union[npt.ArrayLike, str, List[str]]
            Field data to interpolate. Can be an array, string name, or list of both.
        query_positions : npt.ArrayLike, optional
            Array of shape (M, D) with positions where quantities are computed.
            If None, uses particle positions.
        mode : InterpolationMode, default 'field'
            Type of quantity to compute:

            - 'field': Return interpolated field values
            - 'gradient': Return field gradients ∇f
            - 'divergence': Return divergence ∇·**f** (vector fields only)
            - 'curl': Return curl ∇×**f** (vector fields only)
        structure : Structure, optional
            Smoothing structure to use for interpolation. If None, uses the globally set structure.

        Returns
        -------
        npt.NDArray[np.floating]
            Interpolated quantities with shape depending on mode:

            - 'field': (M, F) - interpolated field values
            - 'gradient': (M, F, D) - field gradients
            - 'divergence': (M, 1) - divergence of vector field
            - 'curl': (M, 1) in 2D or (M, 3) in 3D - curl of vector field

        Notes
        -----
        **Mathematical formulas (SPH interpolation)**:

        For a scalar field:

        $$f(x) = \\sum_j \\frac{m_j}{\\rho_j} f_j W(x-x_j,h)$$

        Gradient:

        $$\\nabla f(x) = \\sum_j \\frac{m_j}{\\rho_j} f_j \\nabla W(x-x_j,h)$$

        For a vector field **f**:

        **Divergence**:

        $$\\nabla \\cdot \\mathbf{f}(x) = \\sum_j \\frac{m_j}{\\rho_j} \\mathbf{f}_j \\cdot \\nabla W(x-x_j,h)$$

        **Curl (3D)**:

        $$\\nabla \\times \\mathbf{f}(x) = \\sum_j \\frac{m_j}{\\rho_j} \\mathbf{f}_j \\times \\nabla W(x-x_j,h)$$

        **Curl (2D)** (scalar):

        $$(\\nabla \\times \\mathbf{f})_z = \\sum_j \\frac{m_j}{\\rho_j} \\left( f_{x,j}\\frac{\\partial W}{\\partial y} - f_{y,j}\\frac{\\partial W}{\\partial x} \\right)$$

        Examples
        --------
        >>> pc = PointCloud(positions, weights=masses)
        >>> pc.global_setup(kernel_name='cubic_spline', structure='isotropic', num_neighbors=32)
        >>> pc.compute_smoothing()
        >>> pc.compute_density()
        >>> pc.add_fields('velocity', velocity_data)

        Interpolate field values:

        >>> values = pc.interpolate('velocity', query_positions)

        Compute gradients:

        >>> grads = pc.interpolate('velocity', query_positions, mode='gradient')

        Compute divergence (for vector fields):

        >>> div = pc.interpolate('velocity', query_positions, mode='divergence')

        Compute curl (for 3D vector fields):

        >>> curl = pc.interpolate('velocity', query_positions, mode='curl')

        """
        # ----------------------------
        # Setup
        # ----------------------------
        if mode not in INTERPOLATION_MODES:
            raise ValueError(
                f"'mode' must be one of {INTERPOLATION_MODES}, got '{mode}'"
            )

        # check that structure is set either globally or via argument
        structure_temp = self._resolve_structure(structure)
        if structure_temp not in ["isotropic", "covariant"]:
            raise ValueError(
                f"For interpolation, 'structure' must be one of ['isotropic', 'covariant'], got '{structure_temp}'"
            )

        # check that density has been computed for the chosen structure
        self._check_density_computed(structure_temp)

        # cast fields to correct input format and validate their dimensionality for the requested mode
        fields, fields_sizes = self._resolve_fields(fields)
        self._check_field_dimensionality(mode, fields_sizes)

        # whether query_positions was supplied is decided by rank 0 alone
        # (other ranks may pass anything, e.g. None) -- otherwise ranks
        # could take different branches below and hang/diverge in the
        # collective execution._dispatch call at the end of this method.
        has_query_positions = query_positions is not None
        if self.size > 1:
            has_query_positions = execution._bcast(
                self.comm, has_query_positions if self.rank == 0 else None
            )

        # Local+ghost paths: only when compute_smoothing() actually used it
        # (self.smoothing.used_ghosts -- not just "is a decomposition
        # available", since that alone doesn't say which path produced the
        # smoothing/density data currently stored). Two sub-cases, both
        # sharing `_interpolate_local`'s body: evaluate at the particles
        # themselves (nn_inds already solved by find_neighbors(), via
        # self.ghosts), or at arbitrary caller-supplied query positions
        # (Step 4b: route them by the same Hilbert partition, then solve
        # their K-NN with a generalized ghost exchange).
        if self.smoothing.used_ghosts and not has_query_positions:
            if self.verbose and self.rank == 0:
                mode_str = {
                    "field": "fields",
                    "gradient": "gradients of fields",
                    "divergence": "divergence of fields",
                    "curl": "curl of fields",
                }[mode]
                print(
                    f"[smudgy] Interpolating {mode_str} at query positions using "
                    f"{structure_temp} '{self.smoothing.kernel_name}' kernel (local+ghost)"
                )
            return self._interpolate_local(
                self.decomposition.local_positions,
                self.ghosts,
                fields,
                fields_sizes,
                mode,
                structure_temp,
            )

        if self.smoothing.used_ghosts and has_query_positions:
            # Root-authoritative, like hilbert_partition_and_scatter --
            # route_query_positions Scatterv's the routed chunks itself, so
            # (unlike the old path below) the full (M, D) array is never
            # broadcast to every rank.
            query_positions_root = (
                np.asarray(query_positions, dtype=np.float32)
                if self.rank == 0
                else None
            )
            self.query_routing = route_query_positions(
                self.comm, self.decomposition, query_positions_root, self.periodic
            )
            query_ghosts = exchange_ghosts(
                self.comm,
                self.decomposition,
                self.smoothing.num_neighbors,
                self.dim,
                self.periodic,
                self.boxsize,
                target_positions=self.query_routing.local_positions,
            )

            if self.verbose and self.rank == 0:
                mode_str = {
                    "field": "fields",
                    "gradient": "gradients of fields",
                    "divergence": "divergence of fields",
                    "curl": "curl of fields",
                }[mode]
                print(
                    f"[smudgy] Interpolating {mode_str} at routed query positions "
                    f"using {structure_temp} '{self.smoothing.kernel_name}' kernel "
                    "(local+ghost)"
                )
            return self._interpolate_local(
                self.query_routing.local_positions,
                query_ghosts,
                fields,
                fields_sizes,
                mode,
                structure_temp,
            )

        # if query_positions is None, use particle positions
        if not has_query_positions:
            query_positions = self.positions
            nn_inds = self.smoothing.nn_inds
        else:
            # caller-supplied query positions are only authoritative on rank 0
            query_positions = (
                np.asarray(query_positions, dtype=np.float32)
                if self.rank == 0
                else None
            )
            if self.size > 1:
                query_positions = execution._bcast_array(self.comm, query_positions)

            # for new query positions, need to perform a new neighbor search
            # (rank 0 only; deferred: parallel kd-tree / neighbor search)
            if self.rank == 0:
                tree = self._check_tree()
                _, nn_inds = query_kdtree(
                    tree,
                    query_positions,
                    k=self.num_neighbors,
                )
            else:
                nn_inds = None
            if self.size > 1:
                nn_inds = execution._bcast_array(self.comm, nn_inds)

        # Slice to this rank's local query positions *before* gathering
        # neighbor data below, so the expensive fancy-indexing is O(m/size),
        # not O(m), per rank (see execution._local_slice docstring).
        start, stop = execution._local_slice(
            query_positions.shape[0], self.rank, self.size
        )
        query_positions_local = query_positions[start:stop]
        nn_inds_local = nn_inds[start:stop]

        # ----------------------------
        # Input preparation
        # ----------------------------
        # compute relative coordinates
        r_ij = self._get_rel_coords(
            query_positions_local, self.positions[nn_inds_local]
        )

        # prepare interpolation weights and fields
        fields_temp = fields[nn_inds_local]  # Shape: (M_local, K, num_fields)

        # For divergence and curl, reshape fields to (M, K, num_fields, D)
        # This enables clean einsum patterns since all fields are validated to have self.dim components
        if mode in ("divergence", "curl"):
            num_fields = len(fields_sizes)
            fields_temp = fields_temp.reshape(
                fields_temp.shape[0], fields_temp.shape[1], num_fields, self.dim
            )

        density_temp = (
            self.smoothing.density_covariant[nn_inds_local]
            if structure_temp == "covariant"
            else self.smoothing.density_isotropic[nn_inds_local]
        )
        weights_temp = self.weights[nn_inds_local] / (density_temp + 1e-8)
        h_temp = (
            self.smoothing.smoothing_tensors[nn_inds_local]
            if structure_temp == "covariant"
            else self.smoothing.smoothing_lengths[nn_inds_local]
        )

        # ----------------------------
        # Verbose output / computation
        # ----------------------------
        if self.verbose and self.rank == 0:
            mode_str = {
                "field": "fields",
                "gradient": "gradients of fields",
                "divergence": "divergence of fields",
                "curl": "curl of fields",
            }[mode]
            print(
                f"[smudgy] Interpolating {mode_str} at query positions using "
                f"{structure_temp} '{self.smoothing.kernel_name}' kernel"
            )

        return execution._dispatch(
            "interpolate",
            backend=self.backend,
            kernel_name=self.smoothing.kernel_name,
            dim=self.dim,
            fields=fields_temp,
            weights=weights_temp,
            r_ij=r_ij,
            h=h_temp,
            mode=mode,
            structure=structure_temp,
        )

    def deposit(
        self,
        fields: npt.ArrayLike | str | list[str],
        averaged: bool | Sequence[bool],
        gridnums: int | Sequence[int],
        extent: Sequence[Sequence[float]] | None = None,
        adaptive: bool = True,
        kernel_name: str | None = None,
        structure: Structure | None = None,
        plane_projection: list[int] | None = None,
        integration_method: str = "midpoint",
        eta_crit: float = 4.0,
        return_weights: bool = False,
        gather_to_root: bool = False,
        root: int = 0,
    ) -> (
        npt.NDArray[np.floating]
        | tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
    ):
        """Deposit particle fields onto a structured grid.

        Parameters
        ----------
        fields : Union[npt.ArrayLike, str, List[str]]
            Field data to deposit.
        averaged : Union[bool, Sequence[bool]]
            Whether to divide the result by weights for each field component.
        gridnums : Union[int, Sequence[int]]
            Number of cells along each axis.
        extent : Optional[Sequence[Sequence[float]]], optional
            Domain bounds [[xmin, xmax], [ymin, ymax], ...]. If None, uses `boxsize`.
        kernel_name : str, optional
            Fixed-grid stencil (``ngp``, ``cic``, ``tsc``, ``pcs``, or
            ``pqs``) when ``adaptive=False``. For adaptive deposition, a
            rectangular kernel such as ``tsc_rect`` for ``separable``, or a
            spherical kernel for ``isotropic`` and ``covariant``.
        structure : Structure, optional
            Adaptive smoothing structure. Required only when ``adaptive=True``.
        adaptive : bool, default True
            If True, use the instance's smoothing data. If False, use the
            fixed-grid stencil and do not consult smoothing data.
        plane_projection : List[int], optional
            Indices of the axes to project onto for 3D to 2D deposition.
        integration_method : str, default 'midpoint'
            Kernel integration method.
        eta_crit : float, default 4.0
            Anti-aliasing threshold, in units of kernel diameter / cell
            size, to switch a particle from the sampled kernel deposition to
            the per-cell numerical quadrature deposition. The sample-based
            deposition's resolution is derived automatically from eta_crit
            (Nyquist-safe by construction for any value), so this parameter
            only controls the tradeoff between how much of the particle
            population uses the (cheaper) sampled path versus the
            (exact-per-cell) quadrature path.
        return_weights : bool, default False
            If True, returns the weights (density) grid as well.
        gather_to_root : bool, default False
            If True, the summed grid is delivered to `root` only (via
            `comm.reduce`) instead of every rank (via `comm.allreduce`,
            today's default) -- cheaper when only `root` needs the result.
            Non-`root` ranks then return `None` (or `(None, None)` if
            `return_weights=True`).
        root : int, default 0
            Destination rank when `gather_to_root=True`; unused otherwise.
        backend : {"numpy", "taichi"}, optional
            Backend used for this deposition. Defaults to the PointCloud backend.
        accelerator : str, optional
            Accelerator for the Taichi backend. Defaults to GPU when None.
        omp_threads : int, optional
            Number of CPU threads when using Taichi CPU execution.

        Returns
        -------
        Union[npt.NDArray[np.floating], tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]]
            Deposited field grid, and optionally the weights grid.

        """
        kernel_name_temp, structure_temp = self._resolve_deposition_options(
            adaptive, kernel_name, structure
        )
        fields, fields_sizes = self._resolve_fields(fields)
        averaged = self._resolve_averaged(averaged, fields_sizes)

        # construct the mask of particles falling into extent
        if extent is None:
            if self.boxsize is None:
                raise ValueError("Either 'boxsize' must be set or 'extent' provided")

            box = np.atleast_1d(self.boxsize).astype(np.float32)
            if box.size == 1:
                box = np.repeat(box, self.dim)
            domain_min, domain_max, periodic = (
                np.zeros(self.dim, dtype=np.float32),
                box,
                bool(self.periodic),
            )
        else:
            ext = np.asarray(extent, dtype=np.float32)
            domain_min, domain_max, periodic = ext[:, 0], ext[:, 1], False

        d_lens = domain_max - domain_min

        # Local path: only when a decomposition is available AND (either
        # this is a non-adaptive deposit, which never touches smoothing data
        # at all, or compute_smoothing() actually produced local-sized
        # smoothing data -- self.smoothing.used_ghosts). Guarding on
        # used_ghosts (not just "is a decomposition available") matters
        # because _prepare_deposition_smoothing below indexes
        # self.smoothing.smoothing_lengths/tensors with a LOCAL idx; if that
        # array were actually full-N (used_ghosts False), the same idx
        # values would silently select the wrong particles' data rather than
        # raise.
        use_local = self._using_decomposition() and (
            not adaptive or self.smoothing.used_ghosts
        )

        if use_local:
            local_positions = self.decomposition.local_positions
            mask = np.all(
                (local_positions >= domain_min) & (local_positions <= domain_max),
                axis=1,
            )
            idx_local = np.flatnonzero(mask)  # already this rank's own share
            pos_temp = local_positions[idx_local] - domain_min
            weights_temp = self.decomposition.local_weights[idx_local]
            fields_temp = fields[self.decomposition.local_global_indices][idx_local]
        else:
            mask = np.all(
                (self.positions >= domain_min) & (self.positions <= domain_max),
                axis=1,
            )
            # Slice to this rank's local particles *before* gathering per-particle
            # data below, so that gather is O(n/size), not O(n), per rank (see
            # execution._local_slice docstring).
            idx = np.flatnonzero(mask)
            start, stop = execution._local_slice(idx.shape[0], self.rank, self.size)
            idx_local = idx[start:stop]

            pos_temp = self.positions[idx_local] - domain_min
            weights_temp = self.weights[idx_local]
            fields_temp = fields[idx_local]

        # if plane_projection is set, collect relevant axes
        if plane_projection:
            # cast to array and ensure it's 1D
            plane_projection = np.atleast_1d(plane_projection).astype(int)

            # assert that given axes are all different
            assert (
                plane_projection[0] != plane_projection[1]
            ), "Plane projection axes must be unique, but found duplicates in {plane_projection}"
            # assert that given axes are valid indices for the position dimension
            assert np.all(plane_projection < self.dim) and np.all(
                plane_projection >= 0
            ), f"Plane projection axes must be between 0 and {self.dim - 1}, but found {plane_projection}"
            if self.dim != 3:
                raise ValueError(
                    f"Plane projection requires 3D positions, but positions are {self.dim}d"
                )

            # fancy indexing does not work here -> use np.take to select and stack relevant axes
            pos_temp = np.take(pos_temp, plane_projection, axis=1)
            d_lens = np.take(d_lens, plane_projection).astype(np.float32)

        # resolve deposition dimension and gridnums
        dep_dim = pos_temp.shape[1]
        gridnums_temp = self._resolve_gridnums(gridnums, dep_dim)

        # ----------------------------
        # Input preparation
        # ----------------------------
        h = h_vals = h_vecs = None
        if adaptive:
            h, h_vals, h_vecs = self._prepare_deposition_smoothing(
                structure_temp,
                idx_local,
                dep_dim,
                plane_projection,
            )

        deposit_kwargs = {
            "particle_positions": pos_temp,
            "particle_fields": fields_temp,
            "particle_weights": weights_temp,
            "boxsizes": d_lens,
            "gridnums": gridnums_temp,
            "adaptive": adaptive,
            "structure": structure_temp,
            "periodic": periodic,
            "kernel_name": kernel_name_temp,
            "integration_method": integration_method,
            "eta_crit": eta_crit,
        }

        if adaptive:
            if structure_temp in ("separable", "isotropic"):
                deposit_kwargs["particle_hsml"] = h
            else:
                deposit_kwargs["particle_hmat_eigvecs"] = h_vecs
                deposit_kwargs["particle_hmat_eigvals"] = h_vals

        # ----------------------------
        # Verbose output / computation
        # ----------------------------
        if self.verbose and self.rank == 0:
            structure_prefix = f"{structure_temp} " if structure_temp else ""
            print(f"[smudgy] Depositing using {structure_prefix}'{kernel_name_temp}' kernel")

        if gather_to_root:
            # Skip _dispatch's built-in allreduce-to-everyone reduction and
            # do the equivalent sum ourselves, delivered to `root` only (Step
            # 5 of the domain-decomposition roadmap) -- cheaper than
            # replicating the full grid onto every rank when only `root`
            # needs it.
            fields_grid, weights_grid = execution._dispatch(
                "deposit",
                backend=self.backend,
                reduce=False,
                **deposit_kwargs,
            )
            fields_grid, weights_grid = execution._reduce_sum_to_root(
                self.comm, (fields_grid, weights_grid), root=root
            )
        else:
            fields_grid, weights_grid = execution._dispatch(
                "deposit",
                backend=self.backend,
                **deposit_kwargs,
            )

        if fields_grid is not None:
            for i, avg in enumerate(averaged):
                if avg:
                    fields_grid[i] /= weights_grid + 1e-10

        return (fields_grid, weights_grid) if return_weights else fields_grid
