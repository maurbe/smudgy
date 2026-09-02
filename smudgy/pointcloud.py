"""Core PointCloud class for particle-based computations."""

import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from . import execution
from .backend.neighbors import coordinate_difference_with_pbc
from .backend.taichi import init as taichi_init
from .decomposition import (
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

        Every rank ends up holding only its own share of the particles
        (`self.positions`/`self.weights` are local, count-balanced,
        Hilbert-sorted slices, never the full dataset) -- construction reads
        the full arrays on rank 0 only and immediately Hilbert-partitions
        and scatters them (see `decomposition.hilbert_partition_and_scatter`),
        rather than broadcasting a full copy to every rank first. See
        `self.decomposition` (`decomposition.DecompositionInfo`) for
        provenance (e.g. `local_global_indices`, to reassemble a rank-local
        result back into original order via `gather_particles`).

        Parameters
        ----------
        positions : npt.NDArray[np.floating]
            Particle positions, shape (N, D). Only read on rank 0; ignored
            (may be anything, including `None`) on every other rank.
        weights : npt.NDArray[np.floating] | None
            Particle weights (e.g. masses), shape (N,). If None, uniform
            weights are used. Only read on rank 0.
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

        # Resolve the full arrays (rank 0 only) and the small, cheap-to-
        # broadcast metadata (dim, periodic, boxsize, and the domain extent
        # used for Hilbert quantization) -- `positions_full`/`weights_full`
        # are deliberately local variables, never assigned to `self`, and
        # scattered away below rather than ever broadcast in full.
        if self.rank == 0:
            dim = positions.shape[-1]
            assert dim in (
                1,
                2,
                3,
            ), f"Particle positions must be of shape (N, 1), (N, 2) or (N, 3) but found {positions.shape}"
            positions_full = positions.astype(np.float32)

            weights_full = (
                np.ones(positions_full.shape[0], dtype=np.float32)
                if weights is None
                else weights.astype(np.float32)
            )
            assert (
                weights_full.shape[0] == positions_full.shape[0]
            ), f"Shape mismatch: length of weights and positions must be the same but found: {weights_full.shape} and {positions_full.shape}"

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
                # for the smoothing-length neighbor search) rejects outright
                # ("some data is outside of the periodic domain").
                # `boxsize_resolved` isn't itself guaranteed float32 (e.g. a
                # plain Python float given as `boxsize` stays float64
                # through `np.asarray`), so it's cast explicitly first --
                # np.mod would otherwise silently upcast `positions_full` to
                # float64. The `np.minimum` step closes one more edge case
                # np.mod alone doesn't fully rule out: for a value extremely
                # close to a multiple of boxsize, the internal subtraction
                # can itself round back up to exactly boxsize, so this clips
                # to the largest float32 strictly below it.
                boxsize_f32 = boxsize_resolved.astype(np.float32)
                positions_full = np.mod(positions_full, boxsize_f32)
                positions_full = np.minimum(
                    positions_full, np.nextafter(boxsize_f32, np.float32(0))
                ).astype(np.float32)

            # Domain extent for Hilbert quantization, resolved here (against
            # this rank-0-local array, before it's scattered away) rather
            # than by a separate post-construction call: periodic -> [0,
            # boxsize]; non-periodic -> the data's own bounding box.
            if periodic_resolved:
                domain_min = np.zeros(dim, dtype=np.float32)
                domain_max = boxsize_resolved.astype(np.float32)
            else:
                domain_min = positions_full.min(axis=0)
                domain_max = positions_full.max(axis=0)
        else:
            dim = periodic_resolved = boxsize_resolved = None
            positions_full = weights_full = None
            domain_min = domain_max = None

        if self.size > 1:
            (
                dim,
                periodic_resolved,
                boxsize_resolved,
                domain_min,
                domain_max,
            ) = execution._bcast(
                self.comm,
                (dim, periodic_resolved, boxsize_resolved, domain_min, domain_max),
            )

        self.dim = dim
        self.periodic = periodic_resolved
        self.boxsize = boxsize_resolved

        self.decomposition = hilbert_partition_and_scatter(
            self.comm,
            positions_full,
            weights_full,
            domain_min=domain_min,
            domain_max=domain_max,
            periodic=self.periodic,
        )
        # Aliases, not copies: self.positions/self.weights ARE
        # self.decomposition.local_positions/local_weights, so every
        # existing consumer of self.decomposition.local_* (ghosts.py,
        # decomposition.py, every prior test) keeps working unchanged, and
        # self.positions now means "this rank's local slice" everywhere.
        self.positions = self.decomposition.local_positions
        self.weights = self.decomposition.local_weights

        self.smoothing = SmoothingInfo()
        self.ghosts = GhostInfo()
        self.query_routing = QueryRouting()

        # Verbose output after completed initialization
        if self.verbose and self.rank == 0:
            periodic_str = (
                f"in periodic box of size={self.boxsize}"
                if self.periodic
                else "without periodicity"
            )
            n_total = int(self.decomposition.counts.sum())
            rank_str = f"{self.size} rank{'s' if self.size > 1 else ''}"
            print(
                f"[smudgy] Initialized {self.dim}d PointCloud with {n_total} "
                f"particles {periodic_str} (decomposed across {rank_str})"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        """Plain attribute assignment (`pc.a = b`) is treated as per-rank
        metadata -- set as-is on this rank only, never scattered. Warns
        (doesn't block) when that looks like a likely mistake: an `ndarray`
        whose length equals the *global* particle count, on a run with more
        than one rank -- the sanctioned way to add a per-particle field is
        `add_fields`, which validates shape against the global count,
        Hilbert-reorders, and scatters it into local-sized shares.

        Scoped to `self.size > 1` only: at `size == 1`, local count == global
        count, so a bare `pc.a = np.ones(N)` is already a fully valid,
        correctly-ordered assignment there (indistinguishable, by shape
        alone, from what `add_fields` itself would produce) -- nothing to
        warn about.
        """
        if (
            isinstance(value, np.ndarray)
            and getattr(self, "size", 1) > 1
            and name not in ("positions", "weights")
            and getattr(self, "decomposition", None) is not None
            and value.ndim >= 1
            and value.shape[0] == int(self.decomposition.counts.sum())
        ):
            warnings.warn(
                f"Setting 'pc.{name}' directly to an array of length "
                f"{value.shape[0]}, matching the total particle count, looks "
                "like an attempt to add a per-particle field. Plain attribute "
                "assignment is treated as per-rank/meta-information and is "
                "set as-is on this rank only, unscattered. If this is meant "
                f"to be a per-particle field, use 'pc.add_fields({name!r}, "
                f"values)' instead (and 'pc.delete_fields({name!r})' to "
                "remove it) so it's properly chunked across ranks.",
                stacklevel=2,
            )
        object.__setattr__(self, name, value)

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
        """No-op: decomposition now happens automatically in `__init__`.

        Kept only so an existing `PointCloud(...).decompose().find_neighbors()
        ...` call chain doesn't need to be edited -- there's no full array
        left to re-decompose from by the time this could be called anyway
        (see `self.decomposition`, already populated by `__init__`).
        Re-decomposing/rebalancing after construction (e.g. after particle
        positions move during a simulation) is not supported.

        Returns
        -------
        PointCloud
            self, for chaining.

        """
        return self

    def find_neighbors(
        self,
        num_neighbors: int | None = None,
        max_iterations: int = 20,
        on_max_iterations: str = "raise",
    ) -> "PointCloud":
        """Ghost-particle exchange + iterative true-KNN solve.

        Required before any of `compute_smoothing`/`compute_density`/
        `interpolate`/`deposit` -- they all need this call's result
        (`self.ghosts`, see `ghosts.GhostInfo`) to compute anything, and
        raise a clear error if it hasn't been called yet. `self.decomposition`
        (needed here) is already populated by `__init__`, so this can be
        called immediately after construction (once `num_neighbors` is set,
        e.g. via `global_setup`).

        Returns
        -------
        PointCloud
            self, for chaining (mirrors `global_setup`).

        """
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
                f"[smudgy] Found {num_neighbors_temp} neighbors per particle"
            )
        return self

    def gather_particles(
        self, local_array: npt.NDArray[Any], root: int = 0
    ) -> npt.NDArray[Any] | None:
        """Gather a particle-indexed local array back onto `root` only, in
        original particle order.

        `local_array` must be indexed the way `decomposition.local_*` is
        (row i = this rank's i-th local particle) -- e.g. `smoothing.
        smoothing_lengths`/`smoothing_tensors`, `smoothing.density_isotropic`/
        `density_covariant`, or an `interpolate()` (no `query_positions`)
        result. Cheaper than shipping the reassembled array to every rank:
        this delivers it to `root` only (see `execution._gather_to_root`).

        Returns
        -------
        np.ndarray, shape (n_particles, *local_array.shape[1:]), on `root`;
        `None` on every other rank.
        """
        local_global_indices = self.decomposition.local_global_indices
        if local_array.shape[0] != local_global_indices.shape[0]:
            raise ValueError(
                f"'local_array' has {local_array.shape[0]} rows but this rank "
                f"has {local_global_indices.shape[0]} local particles; "
                "'local_array' must be indexed the same way as "
                "decomposition.local_positions (this rank's local particles)."
            )
        return execution._gather_to_root(
            self.comm, local_global_indices, local_array,
            int(self.decomposition.counts.sum()), root=root,
        )

    def gather_queries(
        self, local_array: npt.NDArray[Any], root: int = 0
    ) -> npt.NDArray[Any] | None:
        """Gather a query-position-indexed local array back onto `root` only,
        in the original query-array order (Step 5 of the domain-decomposition
        roadmap).

        `local_array` must be an `interpolate(query_positions=...)` result --
        indexed the way `query_routing.local_positions` is (row i = this
        rank's i-th routed query position). Requires that call to have
        populated `self.query_routing`.

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

    def get(self, name: str, root: int = 0) -> npt.NDArray[Any] | None:
        """Gather a named per-particle field back to `root`, reassembled into
        original input order -- the read-only, results-out-of-the-pipeline
        counterpart to `add_fields`.

        `name` resolves against `self.smoothing` first (e.g.
        `'smoothing_lengths'`, `'density_isotropic'`), then against `self`
        directly (a custom field added via `add_fields`, or `'positions'`/
        `'weights'`). Meant only for handing a final result back to the
        caller -- internal computation should keep reading the per-rank
        attribute directly (e.g. `self.smoothing.density_isotropic`), never
        `get()`, since this triggers a real MPI collective (`gather_particles`)
        and returns `None` on every rank but `root`.

        Returns
        -------
        np.ndarray, shape (n_particles, ...), on `root`; `None` elsewhere.
        """
        if hasattr(self.smoothing, name):
            local_arr = getattr(self.smoothing, name)
        elif hasattr(self, name):
            local_arr = getattr(self, name)
        else:
            raise AttributeError(
                f"No field named {name!r} found (checked 'pc.smoothing.{name}' "
                f"and 'pc.{name}')."
            )
        if local_arr is None:
            raise AttributeError(f"'{name}' has not been computed yet.")
        return self.gather_particles(local_arr, root=root)

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

    def _check_neighbors(self, nn_inds: npt.NDArray[np.integer], n_valid: int) -> None:
        """Verify that neighbor indices are within valid bounds.

        `n_valid` is the size of the (local + ghost) combined array `nn_inds`
        indexes into -- always passed explicitly by the caller, since that
        combined size depends on how many ghosts were fetched. A rank with
        zero local rows (N < P, or a query batch that routed none here) has
        an empty `nn_inds` -- `np.max` on an empty array raises unconditionally,
        so there is vacuously nothing to check in that case.
        """
        if nn_inds.size == 0:
            return
        max_idx = np.max(nn_inds)
        if max_idx >= n_valid:
            raise IndexError(
                f"Neighbor index {max_idx} is out of bounds for {n_valid} particles. This indicates a bug in the neighbor search or input setup."
            )

    def _ensure_neighbors_found(self, num_neighbors: int) -> None:
        """Lazily run `find_neighbors()` on first use, mirroring the pre-
        decomposition-refactor behavior of building the tree and finding
        neighbors automatically at the first call that actually needs them,
        rather than requiring an explicit `find_neighbors()` call first.

        Safe as a collective: `self.ghosts.nn_inds` starts identically `None`
        on every rank (set once, together, either here or by an explicit
        `find_neighbors()` call -- never by rank-dependent branching), so
        every rank reaches the same decision on whether to call
        `find_neighbors()` here. Relies on particle positions never changing
        after construction (no re-decomposition/rebalancing is supported, see
        `decompose()`'s docstring) -- once found, ghosts/neighbors remain
        valid for this `PointCloud`'s lifetime, so this triggers at most once;
        every call after the first sees `nn_inds` already populated and is a
        no-op.
        """
        if self.ghosts.nn_inds is None:
            self.find_neighbors(num_neighbors=num_neighbors)

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
        """Ensure the first dimension of an array matches this rank's local
        particle count (`self.positions` is always local, never full-N)."""
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
        """Resolve fields to a single array and return component counts.

        A field given by name was already decomposed by `add_fields` (local-
        sized already). A field given directly as an array is *not*
        decomposed for the caller -- it must already be this rank's local
        slice (same convention `self.positions` uses), not a full-N array.
        """
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

    def _compute_smoothing_local(
        self,
        query_positions_local: npt.NDArray[np.floating],
        ghost_info: GhostInfo,
        num_neighbors_temp: int,
        structure_temp: Structure,
    ) -> None:
        """Shared body of `compute_smoothing()`'s two branches: particle
        positions (`ghost_info` is `self.ghosts`, from `find_neighbors()`)
        or arbitrary query positions (`ghost_info` is a fresh
        `exchange_ghosts(..., target_positions=...)` result, routed the same
        way `interpolate(query_positions=...)` routes them). Stores
        local-sized results into `self.smoothing.*`.
        """
        nn_inds_local = ghost_info.nn_inds
        nn_dists_local = ghost_info.nn_dists
        combined_positions = self._combined_local_and_ghost(
            self.decomposition.local_positions, ghost_info.ghost_positions
        )
        combined_weights = self._combined_local_and_ghost(
            self.decomposition.local_weights, ghost_info.ghost_weights
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
                query_positions=query_positions_local,
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
        # Aliased (not copied): lets compute_density()/interpolate() rebuild
        # the same combined [local, ghost] array nn_inds indexes into,
        # without assuming it was always self.ghosts (see SmoothingInfo's
        # docstring -- ghost_info here is query_ghosts, not self.ghosts, for
        # the arbitrary-query-position case above).
        self.smoothing.ghost_positions = ghost_info.ghost_positions
        self.smoothing.ghost_weights = ghost_info.ghost_weights
        self._check_neighbors(nn_inds_local, n_valid=combined_positions.shape[0])

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
        self._ensure_neighbors_found(num_neighbors_temp)

        # whether query_positions was supplied is decided by rank 0 alone
        # (other ranks may pass anything, e.g. None) -- otherwise ranks
        # could take different branches below and hang/diverge in the
        # collective calls each branch makes (mirrors interpolate()'s
        # has_query_positions handling).
        has_query_positions = query_positions is not None
        if self.size > 1:
            has_query_positions = execution._bcast(
                self.comm, has_query_positions if self.rank == 0 else None
            )

        if not has_query_positions:
            # self.ghosts.nn_inds has the k find_neighbors() was last called
            # with baked in -- a mismatch can't be silently reused (it would
            # read the wrong number of neighbors), so this must raise rather
            # than guess.
            if self.ghosts.nn_inds.shape[1] != num_neighbors_temp:
                raise ValueError(
                    f"'num_neighbors' ({num_neighbors_temp}) does not match "
                    f"the value 'find_neighbors' was last called with "
                    f"({self.ghosts.nn_inds.shape[1]}); call "
                    f"'find_neighbors(num_neighbors={num_neighbors_temp})' "
                    "again first."
                )
            if self.verbose and self.rank == 0:
                info_str = "tensors" if structure_temp == "covariant" else "lengths"
                print(
                    f"[smudgy] Computing smoothing {info_str} from "
                    f"{num_neighbors_temp} neighbors"
                )
            self._compute_smoothing_local(
                self.decomposition.local_positions,
                self.ghosts,
                num_neighbors_temp,
                structure_temp,
            )
            return

        # Arbitrary query positions: route them by Hilbert code the same way
        # interpolate(query_positions=...) does (Step 4b), then solve their
        # K-NN with a fresh generalized ghost exchange -- num_neighbors here
        # is this call's own, not tied to self.ghosts/find_neighbors().
        query_positions_root = (
            np.asarray(query_positions, dtype=np.float32) if self.rank == 0 else None
        )
        self.query_routing = route_query_positions(
            self.comm, self.decomposition, query_positions_root, self.periodic
        )
        query_ghosts = exchange_ghosts(
            self.comm,
            self.decomposition,
            num_neighbors_temp,
            self.dim,
            self.periodic,
            self.boxsize,
            target_positions=self.query_routing.local_positions,
        )
        if self.verbose and self.rank == 0:
            info_str = "tensors" if structure_temp == "covariant" else "lengths"
            print(
                f"[smudgy] Computing smoothing {info_str} from "
                f"{num_neighbors_temp} neighbors at routed query positions"
            )
        self._compute_smoothing_local(
            self.query_routing.local_positions,
            query_ghosts,
            num_neighbors_temp,
            structure_temp,
        )

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

        # self.smoothing.nn_inds/smoothing_lengths/smoothing_tensors are
        # already local-sized (compute_smoothing's local+ghost result), so
        # no _local_slice chunking is needed -- an already-local array
        # doesn't need re-chunking. Neighbors can be ghosts (imported from
        # other ranks), which is why combined_positions/combined_weights
        # (local + ghost) rather than self.decomposition.local_* alone are
        # needed to index nn_inds -- using self.smoothing.ghost_positions/
        # ghost_weights (not self.ghosts directly), since compute_smoothing()
        # may have used a different ghost set (an arbitrary-query-position
        # call), see SmoothingInfo's docstring.
        nn_inds_local = self.smoothing.nn_inds
        combined_positions = self._combined_local_and_ghost(
            self.decomposition.local_positions, self.smoothing.ghost_positions
        )
        combined_weights = self._combined_local_and_ghost(
            self.decomposition.local_weights, self.smoothing.ghost_weights
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

        self.smoothing.kernel_name = kn

        if st == "covariant":
            self.smoothing.density_covariant = density
        else:
            self.smoothing.density_isotropic = density

    def add_fields(
        self, names: str | list[str], values: npt.ArrayLike | list[npt.ArrayLike]
    ) -> None:
        """Add one or multiple custom fields to the PointCloud instance.

        Split across ranks the same way positions/weights are: `values` is
        only read on rank 0, reordered via `self.decomposition.root_order`
        (the same Hilbert-sort permutation used at construction) and
        scattered out, so the stored attribute is this rank's local slice --
        never a full-N array replicated on every rank.

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
        # caller-supplied field values are only authoritative on rank 0.
        # Shape validated against the GLOBAL count (not self.positions.
        # shape[0], which is local) here, before any scattering -- and the
        # pass/fail decision is broadcast so every rank raises together (or
        # none do) rather than rank 0 raising before calling _scatterv_rows
        # while every other rank is already waiting on that same collective.
        if self.rank == 0:
            values_full = np.asarray(values, dtype=np.float32)
            n_total = int(self.decomposition.counts.sum())
            shape_ok = values_full.shape[0] == n_total
            error_msg = (
                None
                if shape_ok
                else f"Length of '{name}' ({values_full.shape[0]}) must match "
                f"number of points ({n_total})"
            )
        else:
            values_full = None
            shape_ok = error_msg = None

        shape_ok, error_msg = execution._bcast(
            self.comm, (shape_ok, error_msg) if self.rank == 0 else None
        )
        if not shape_ok:
            raise ValueError(error_msg)

        values_sorted = (
            values_full[self.decomposition.root_order] if self.rank == 0 else None
        )
        values_local = execution._scatterv_rows(
            self.comm, values_sorted, self.decomposition.counts, root=0
        )

        if self.rank == 0 and hasattr(self, name):
            print(f"Overwriting existing attribute '{name}' on PointCloud instance.")

        setattr(self, name, values_local)

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

        # fields is already this rank's local slice (add_fields() now
        # decomposes fields the same way positions/weights are, so no
        # local_global_indices slicing is needed here anymore).
        ghost_fields = push_to_ghosts(self.comm, ghost_info, fields)
        combined_fields = self._combined_local_and_ghost(fields, ghost_fields)
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
        self._ensure_neighbors_found(self._resolve_num_neighbors())
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

        # Two cases, both sharing `_interpolate_local`'s body: evaluate at
        # the particles themselves (nn_inds already solved by
        # find_neighbors(), via self.ghosts), or at arbitrary caller-
        # supplied query positions (route them by the same Hilbert
        # partition, then solve their K-NN with a generalized ghost
        # exchange).
        if not has_query_positions:
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

        if has_query_positions:
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

        # idx_local selects this rank's own local particles falling within
        # the requested extent -- already this rank's share (Step 1's
        # Hilbert partition already balanced counts, so no further chunking
        # is needed here). For adaptive=True, this assumes compute_smoothing()
        # was last called at particle positions (not query_positions) --
        # matching this method's own long-standing "use the instance's
        # smoothing data" contract.
        local_positions = self.decomposition.local_positions
        mask = np.all(
            (local_positions >= domain_min) & (local_positions <= domain_max),
            axis=1,
        )
        idx_local = np.flatnonzero(mask)
        pos_temp = local_positions[idx_local] - domain_min
        weights_temp = self.decomposition.local_weights[idx_local]
        # fields is already this rank's local slice (add_fields() decomposes
        # fields the same way positions/weights are), so only idx_local's
        # own within-rank selection is needed here.
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
