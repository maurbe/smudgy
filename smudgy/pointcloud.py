"""The :class:`PointCloud` class: smoothing, density, interpolation and deposition.

MPI model: the constructor distributes the particles over the ranks of
``MPI.COMM_WORLD`` (Hilbert-curve domain decomposition). Every method is a
collective call. Inputs given as full arrays (positions, field values, query
positions) are read on rank 0 only; per-particle results are rank-local and
are collected with :meth:`PointCloud.get`, :meth:`PointCloud.gather_particles`
or :meth:`PointCloud.gather_queries`.
"""

import operator
import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from . import execution
from .backend import check_backend
from .backend._common import ADAPTIVE_STRUCTURES as STRUCTURES
from .backend._common import (  # noqa: F401  (INTERPOLATION_MODES re-exported)
    INTERPOLATION_MODES,
    check_interpolation_mode,
    check_particle_structure,
)
from .backend.numpy.tensor_utils import _fix_eigvec_sign
from .backend.quadrants import init as quadrants_init
from .decomposition import (
    QueryRouting,
    hilbert_partition_and_scatter,
    route_query_positions,
)
from .ghosts import GhostInfo, exchange_ghosts, push_to_ghosts
from .smooth import SmoothingInfo

Structure = Literal["separable", "isotropic", "covariant"]
InterpolationMode = Literal["field", "gradient", "divergence", "curl"]


def _input_error(positions, weights, boxsize) -> str | None:
    """Why the constructor inputs are invalid, or None if they are valid."""
    if positions.ndim != 2 or positions.shape[1] not in (1, 2, 3):
        return (
            "Particle positions must be of shape (N, 1), (N, 2) or (N, 3) but "
            f"found {positions.shape}"
        )
    n, dim = positions.shape
    if weights is not None and np.shape(weights) != (n,):
        return f"'weights' must have shape ({n},), got {np.shape(weights)}"
    if boxsize is not None:
        box = np.asarray(boxsize, dtype=np.float64)
        if box.shape not in ((), (dim,)):
            return f"'boxsize' must be a scalar or have shape ({dim},), got {box.shape}"
        if not np.all(box > 0):
            return f"'boxsize' must be positive, got {box.tolist()}"
    return None


class PointCloud:
    """Particles with positions and weights, the central smudgy object.

    ``positions``, ``weights``, fields added via :meth:`add_fields` and all
    per-particle results hold this rank's share of the particles only, in
    Hilbert order (on a single rank: all particles, in input order).
    """

    def __init__(
        self,
        positions: npt.NDArray[np.floating],
        weights: npt.NDArray[np.floating] | None = None,
        boxsize: float | Sequence[float] | None = None,
        verbose: bool = True,
        backend: str = "quadrants",
        **kwargs,
    ) -> None:
        """Initialize a point cloud.

        On several MPI ranks, the input arrays are only read on rank 0 and the
        particles are distributed across the ranks; ``self.positions`` and
        ``self.weights`` then hold only the current rank's share.

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
        backend : str, default "quadrants"
            Computational backend, one of ``"quadrants"`` or ``"numpy"``. The
            numpy backend supports interpolation and fixed-grid
            (``adaptive=False``) deposition only.
        **kwargs : dict
            Forwarded to the Quadrants runtime initialization
            (:func:`smudgy.backend.quadrants.init`), e.g. ``arch`` (one of
            ``"cpu"``, ``"gpu"``, ``"cuda"``, ``"vulkan"``, ``"metal"``;
            ``"gpu"`` on first initialization if omitted), ``force=True`` to
            re-initialize an already running runtime, or any other
            ``qd.init`` keyword argument (e.g. ``cpu_max_num_threads``).
        """
        self.verbose = verbose
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        if verbose and self.rank == 0:
            print(f'[smudgy] Using {self.size} MPI rank{"s" if self.size > 1 else ""}')

        self.set_backend(backend, **kwargs)

        # Inputs are validated on rank 0, the only rank reading them; the
        # verdict is broadcast so that every rank raises together.
        error_msg = None
        if self.rank == 0:
            positions = np.asarray(positions)
            error_msg = _input_error(positions, weights, boxsize)
        if self.size > 1:
            error_msg = execution._bcast(self.comm, error_msg)
        if error_msg is not None:
            raise ValueError(error_msg)

        if self.rank == 0:
            dim = positions.shape[-1]
            positions_full = positions.astype(np.float32)

            weights_full = (
                np.ones(positions_full.shape[0], dtype=np.float32)
                if weights is None
                else np.asarray(weights, dtype=np.float32)
            )

            if boxsize is None:
                periodic_resolved = False
                boxsize_resolved = None
            else:
                periodic_resolved = True
                boxsize_resolved = np.broadcast_to(np.asarray(boxsize), (dim,)).copy()

                # Wrap into [0, boxsize) in float32: the input need not be
                # wrapped, and values just below boxsize can round up to it
                # in float32, which scipy's periodic cKDTree rejects. Both
                # np.mod and the cast can land exactly on boxsize, hence the
                # clip to the largest float32 below it.
                boxsize_f32 = boxsize_resolved.astype(np.float32)
                positions_full = np.mod(positions_full, boxsize_f32)
                positions_full = np.minimum(
                    positions_full, np.nextafter(boxsize_f32, np.float32(0))
                ).astype(np.float32)

            # domain of the Hilbert quantization: the box, or the bounding box
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
        # aliases of this rank's share, not copies
        self.positions = self.decomposition.local_positions
        self.weights = self.decomposition.local_weights

        self.smoothing = SmoothingInfo()
        self.ghosts = GhostInfo()
        self.query_routing = QueryRouting()
        self._fields = set()  # names added via add_fields
        self._ghost_smoothing_cache = {}  # see _combined_volumes_and_h

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
        """Set an attribute on this rank only, without distributing it.

        On several ranks, warns if the value looks like a full per-particle
        array (length = total particle count): such fields must be added with
        :meth:`add_fields`, which distributes them like the particles.
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

    def set_backend(self, backend: str = "quadrants", **kwargs) -> None:
        """Set the computational backend.

        Parameters
        ----------
        backend : str, default "quadrants"
            One of ``"quadrants"`` or ``"numpy"``.
        **kwargs : dict
            Forwarded to :func:`smudgy.backend.quadrants.init` when
            ``backend="quadrants"`` (e.g. ``arch``, ``force``); see
            :class:`PointCloud`.

        """
        check_backend(backend)
        self.backend = backend
        if backend == "quadrants":
            quadrants_init(**kwargs)
        if self.verbose and getattr(self, "rank", 0) == 0:
            print(f"[smudgy] Set {backend} backend")

    def find_neighbors(
        self,
        num_neighbors: int | None = None,
        max_iterations: int = 20,
        on_max_iterations: str = "raise",
    ) -> "PointCloud":
        """Find the exact ``num_neighbors`` nearest neighbors of every particle.

        Imports ghost particles from neighboring ranks and iteratively grows
        the search radius until every local particle has found its true
        nearest neighbors (identical to a single-rank search). The result is
        stored in ``self.ghosts`` (:class:`smudgy.ghosts.GhostInfo`).

        Calling this explicitly is optional: :meth:`compute_smoothing` and
        :meth:`interpolate` call it automatically on first use. Call it again
        to change the number of neighbors used by later computations.

        Parameters
        ----------
        num_neighbors : int, optional
            Number of nearest neighbors per particle. Defaults to the value
            set via :meth:`global_setup`.
        max_iterations : int, default 20
            Maximum number of radius-growing ghost-exchange iterations.
        on_max_iterations : str, default "raise"
            Behavior if the search has not converged after
            ``max_iterations``: ``"raise"`` raises a ``RuntimeError``; any
            other value only emits a warning (on rank 0).

        Returns
        -------
        PointCloud
            self, for chaining.

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
        """Collect a per-particle array from all ranks, in the original particle order.

        Parameters
        ----------
        local_array : np.ndarray
            Rank-local per-particle array, indexed like ``self.positions`` (e.g. a
            density or an :meth:`interpolate` result without ``query_positions``).
        root : int, default 0
            Rank that receives the result.

        Returns
        -------
        np.ndarray or None
            Full array of shape (N, ...) on `root`, None on all other ranks.
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
        """Collect a per-query-point array from all ranks, in the original query order.

        Parameters
        ----------
        local_array : np.ndarray
            Rank-local result of :meth:`interpolate` with ``query_positions``.
        root : int, default 0
            Rank that receives the result.

        Returns
        -------
        np.ndarray or None
            Full array of shape (M, ...) on `root`, None on all other ranks.
        """
        local_global_indices = self.query_routing.local_global_indices
        if local_global_indices is None:
            raise AttributeError(
                "No query positions have been routed yet; call 'interpolate' "
                "or 'compute_smoothing' with 'query_positions' first."
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
        """Collect a named per-particle quantity from all ranks, in the original order.

        Parameters
        ----------
        name : str
            Attribute of ``self.smoothing`` (e.g. ``"smoothing_lengths"``,
            ``"density_isotropic"``) or a field added via :meth:`add_fields`.
        root : int, default 0
            Rank that receives the result.

        Returns
        -------
        np.ndarray or None
            Full array of shape (N, ...) on `root`, None on all other ranks.
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
        if not isinstance(local_arr, np.ndarray):
            raise TypeError(
                f"'{name}' is a {type(local_arr).__name__}, not a per-particle array."
            )
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
            value = self._as_int(value, "num_neighbors")
            if value <= 0:
                raise ValueError(
                    f"'num_neighbors' must be a positive integer but found {value}"
                )
        else:
            raise ValueError(
                f"Unsupported property '{name}'; expected one of ('structure', 'kernel_name', 'num_neighbors')"
            )

        setattr(self, name, value)

        if self.verbose and self.rank == 0:
            message = {
                "structure": f"[smudgy] Set structure to '{value}'",
                "kernel_name": f"[smudgy] Set kernel to '{value}'",
                "num_neighbors": f"[smudgy] Set number of neighbors to {value}",
            }[name]
            print(message)

    def _check_property(self, property: str | None = None) -> None:
        """Ensure a global property is set before use."""
        if not hasattr(self, property):
            raise AttributeError(
                f"'{property}' has not been set: either set it via the 'global_setup' method or provide it as a function argument"
            )

    def _check_neighbors(self, nn_inds: npt.NDArray[np.integer], n_valid: int) -> None:
        """Raise if `nn_inds` points past the `n_valid` local + ghost particles."""
        if nn_inds.size == 0:
            return
        max_idx = np.max(nn_inds)
        if max_idx >= n_valid:
            raise IndexError(
                f"Neighbor index {max_idx} is out of bounds for {n_valid} particles. This indicates a bug in the neighbor search or input setup."
            )

    def _ensure_neighbors_found(self, num_neighbors: int) -> None:
        """Run :meth:`find_neighbors` if no neighbors have been found yet.

        The condition is identical on every rank, so this is collective-safe.
        """
        if self.ghosts.nn_inds is None:
            self.find_neighbors(num_neighbors=num_neighbors)

    @staticmethod
    def _combined_local_and_ghost(
        local_arr: npt.NDArray[Any], ghost_arr: npt.NDArray[Any]
    ) -> npt.NDArray[Any]:
        """Local rows followed by ghost rows, the order ``nn_inds`` indexes into."""
        return np.concatenate([local_arr, ghost_arr], axis=0)

    def _combined_positions_weights(self, ghost_positions, ghost_weights):
        """Local+ghost positions and weights, cached per ghost set.

        The same ghost set is typically reused by compute_smoothing,
        compute_density and every interpolate call, so the two
        concatenations are done once. The cache holds the ghost arrays
        themselves, so an identity match cannot be a recycled id.
        """
        key = (self.decomposition.local_positions, ghost_positions, ghost_weights)
        cache = getattr(self, "_combined_cache", None)
        if cache is None or any(a is not b for a, b in zip(cache[0], key)):
            combined = (
                self._combined_local_and_ghost(key[0], ghost_positions),
                self._combined_local_and_ghost(
                    self.decomposition.local_weights, ghost_weights
                ),
            )
            cache = (key, combined)
            self._combined_cache = cache
        return cache[1]

    def _reset_densities(self) -> None:
        """Drop density estimates; called whenever the smoothing data changes."""
        self.smoothing.density_isotropic = None
        self.smoothing.density_covariant = None
        self.smoothing.kernel_name = None
        self.smoothing.density_kernel = {}
        self._ghost_smoothing_cache = {}

    def _check_density_computed(self, structure: Structure) -> None:
        """Verify that density has been computed for the requested structure."""
        field = (
            self.smoothing.density_isotropic
            if structure == "isotropic"
            else self.smoothing.density_covariant
        )
        if field is None or structure not in self.smoothing.density_kernel:
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
        if not self.smoothing.at_particles:
            raise ValueError(
                "The current smoothing data belongs to the query positions of "
                "the last 'compute_smoothing(query_positions=...)' call; call "
                "'compute_smoothing()' without query positions first."
            )

    def _check_shape(self, arr: npt.NDArray[Any], name: str) -> None:
        """Raise unless `arr` has one row per local particle."""
        if arr.shape[0] != self.positions.shape[0]:
            raise ValueError(
                f"Length of '{name}' ({arr.shape[0]}) must match number of points ({self.positions.shape[0]})"
            )

    def _check_field_dimensionality(
        self,
        mode: InterpolationMode,
        field_sizes: list[int],
    ) -> None:
        """Check that field dimensions match the requested operation.

        Parameters
        ----------
        mode : InterpolationMode
            The interpolation mode ('field', 'gradient', 'divergence', 'curl').
        field_sizes : list[int]
            List of component counts for each field.

        Raises
        ------
        ValueError
            If field type is incompatible with the requested operation.

        """
        check_interpolation_mode(mode, self.dim)
        if mode in ("divergence", "curl"):
            for i, size in enumerate(field_sizes):
                if size != self.dim:
                    raise ValueError(
                        f"{mode} requires vector fields with {self.dim} components "
                        f"(the spatial dimension), but field {i} has {size} component(s)"
                    )

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
        if not isinstance(kernel_name, str):
            raise TypeError(f"'kernel_name' must be a string but found {type(kernel_name)}")
        return kernel_name

    def _resolve_num_neighbors(self, num_neighbors: int | None = None) -> int:
        """Resolve the number of neighbors from argument or global state."""
        if num_neighbors is None:
            self._check_property("num_neighbors")
            return self.num_neighbors
        num_neighbors = self._as_int(num_neighbors, "num_neighbors")
        if num_neighbors <= 0:
            raise ValueError(
                f"'num_neighbors' must be a positive integer but found {num_neighbors}"
            )
        return num_neighbors

    @staticmethod
    def _as_int(value: Any, name: str) -> int:
        """`value` as a Python int; accepts any integer type (e.g. np.int64)."""
        if isinstance(value, bool):
            raise TypeError(f"'{name}' must be an int but found {type(value)}")
        try:
            return operator.index(value)
        except TypeError:
            raise TypeError(f"'{name}' must be an int but found {type(value)}") from None

    def _resolve_fields(
        self,
        fields: npt.ArrayLike | str | list[str],
        dtype: npt.DTypeLike | None = None,
    ) -> tuple[npt.NDArray[np.floating], list[int]]:
        """Stack fields into one (n_local, C) array; also return each field's C.

        Names refer to attributes (fields from :meth:`add_fields`); arrays
        must already be this rank's share, indexed like ``self.positions``.
        With `dtype`, the result is C-contiguous of that dtype (one copy).
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

        if dtype is not None and len(arrays) == 1:
            return np.ascontiguousarray(arrays[0], dtype=dtype), field_sizes
        return np.concatenate(arrays, axis=-1, dtype=dtype), field_sizes

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

        if isinstance(averaged, bool):
            return np.full(total_components, averaged, dtype=np.bool_)

        if not isinstance(averaged, (list, tuple)):
            raise TypeError(
                f"'averaged' must be a bool or sequence of bools, found {type(averaged)}"
            )

        # one flag per component, or one per field
        if len(averaged) == total_components:
            return np.asarray(averaged, dtype=np.bool_)

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
        """Per-axis cell counts from a scalar or a length-`dim` sequence."""
        gn = np.atleast_1d(gridnums).astype(np.int32)
        if gn.size == 1:
            gn = np.repeat(gn, dim)
        if gn.size != dim:
            raise ValueError(
                f"Length of 'gridnums' ({gn.size}) must match deposition dimension ({dim})"
            )
        if np.any(gn <= 0):
            raise ValueError(f"'gridnums' must be positive, got {gn.tolist()}")
        return gn

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
            Backend to use for computations ('numpy' or 'quadrants').
        kwargs : dict
            Additional keyword arguments for backend initialization.

        Returns
        -------
        PointCloud
            The current instance for method chaining.

        """
        if structure is not None:
            self._set_property("structure", structure)

        if kernel_name is not None:
            self._set_property("kernel_name", kernel_name)

        if num_neighbors is not None:
            self._set_property("num_neighbors", num_neighbors)

        if backend is not None:
            self.set_backend(backend, **kwargs)

        return self

    def _compute_smoothing_local(
        self,
        query_positions_local: npt.NDArray[np.floating],
        ghost_info: GhostInfo,
        num_neighbors_temp: int,
        structure_temp: Structure,
        tensor_options: dict,
        at_particles: bool,
    ) -> None:
        """Compute smoothing at this rank's targets and store it in ``self.smoothing``.

        The targets are the local particles (`ghost_info` = ``self.ghosts``)
        or routed query positions (`ghost_info` from a query ghost exchange).
        """
        nn_inds_local = ghost_info.nn_inds
        nn_dists_local = ghost_info.nn_dists
        combined_positions, combined_weights = self._combined_positions_weights(
            ghost_info.ghost_positions, ghost_info.ghost_weights
        )

        if structure_temp in ("separable", "isotropic"):
            self.smoothing.smoothing_lengths = execution._dispatch(
                "compute_hsml",
                backend=self.backend,
                nn_dists=nn_dists_local,
            )
        else:
            (
                smoothing_tensors,
                smoothing_tensors_eigvals,
                smoothing_tensors_eigvecs,
            ) = execution._dispatch(
                "compute_hmat",
                backend=self.backend,
                query_positions=query_positions_local,
                positions=combined_positions,
                weights=combined_weights,
                nn_inds=nn_inds_local,
                boxsize=self.boxsize,
                hsml=(
                    nn_dists_local[:, -1]
                    if tensor_options["volume_preserving"]
                    else None
                ),
                **tensor_options,
            )
            self.smoothing.smoothing_tensors = smoothing_tensors
            self.smoothing.smoothing_tensors_eigvals = smoothing_tensors_eigvals
            self.smoothing.smoothing_tensors_eigvecs = smoothing_tensors_eigvecs
            self.smoothing.tensor_options = dict(tensor_options)

        self.smoothing.nn_inds, self.smoothing.nn_dists = (
            nn_inds_local,
            nn_dists_local,
        )
        self.smoothing.num_neighbors = num_neighbors_temp
        # the ghost set nn_inds refers to (not necessarily self.ghosts)
        self.smoothing.ghost_positions = ghost_info.ghost_positions
        self.smoothing.ghost_weights = ghost_info.ghost_weights
        self.smoothing.at_particles = at_particles
        self._reset_densities()
        self._check_neighbors(nn_inds_local, n_valid=combined_positions.shape[0])

    def _has_query_positions(self, query_positions) -> bool:
        """Whether query positions were given, as decided on rank 0 for all ranks."""
        has_query_positions = query_positions is not None
        if self.size > 1:
            has_query_positions = execution._bcast(
                self.comm, has_query_positions if self.rank == 0 else None
            )
        return has_query_positions

    def _route_queries(self, query_positions, num_neighbors: int) -> GhostInfo:
        """Route query positions to the ranks owning their region.

        The positions are read on rank 0. Stores the routing in
        ``self.query_routing`` and returns the queries' nearest neighbors.
        """
        query_positions_root = (
            np.asarray(query_positions, dtype=np.float32) if self.rank == 0 else None
        )
        self.query_routing = route_query_positions(
            self.comm, self.decomposition, query_positions_root, self.periodic
        )
        return exchange_ghosts(
            self.comm,
            self.decomposition,
            num_neighbors,
            self.dim,
            self.periodic,
            self.boxsize,
            target_positions=self.query_routing.local_positions,
        )

    def compute_smoothing(
        self,
        query_positions: npt.ArrayLike | None = None,
        num_neighbors: int | None = None,
        structure: Structure | None = None,
        ratio_max: float = 100.0,
        min_axis: float | None = None,
        volume_preserving: bool = False,
    ) -> None:
        r"""Compute smoothing lengths/tensors for SPH calculations.

        Parameters
        ----------
        query_positions : npt.ArrayLike, optional
            Positions where smoothing is evaluated, shape (M, D); only read
            on rank 0. If None, uses the particle positions. Note that
            passing query positions *replaces* ``self.smoothing`` with data
            for those (routed, rank-local) query points; :meth:`compute_density`,
            :meth:`interpolate` and :meth:`deposit` then raise until this
            method is called again without ``query_positions``.
        num_neighbors : int, optional
            Number of neighbors for smoothing length computation. Must match
            the value :meth:`find_neighbors` was last called with.
        structure : Structure, optional
            Smoothing structure, one of ``"separable"``, ``"isotropic"`` or
            ``"covariant"``.
        ratio_max : float, default 100.0
            ``covariant`` only. Maximum axis ratio h_max / h_min of the
            smoothing tensors (>= 1); guards against degenerate (coplanar or
            collinear) neighbor sets.
        min_axis : float, optional
            ``covariant`` only. Floor on every axis of the smoothing tensors,
            e.g. a softening length or cell size. Only ever enlarges axes.
        volume_preserving : bool, default False
            ``covariant`` only. If True, rescale each tensor so that
            det H = h^dim, with h the isotropic smoothing length (distance to
            the `num_neighbors`-th neighbor): anisotropy in shape only. May
            break the property that the kernel support encloses all neighbors.

        Notes
        -----
        Results are stored in the ``smoothing`` attribute and are rank-local
        (use :meth:`get` or :meth:`gather_particles` to collect them).

        Isotropic and separable smoothing lengths are the distance to the
        ``num_neighbors``-th nearest neighbor. Covariant smoothing tensors
        follow Marinho (2021): :math:`H = \zeta_{\max} \Sigma^{1/2}`, with
        :math:`\Sigma` the mass-weighted covariance of the neighbors about
        the query point and :math:`\zeta_{\max}` their largest Mahalanobis
        distance, so the kernel support is the smallest ellipsoid of that
        shape enclosing all neighbors. Like every collective call, all MPI
        ranks must pass the same ``ratio_max``/``min_axis``/
        ``volume_preserving``.

        """
        num_neighbors_temp = self._resolve_num_neighbors(num_neighbors)
        structure_temp = self._resolve_structure(structure)
        tensor_options = {
            "ratio_max": ratio_max,
            "min_axis": min_axis,
            "volume_preserving": volume_preserving,
        }
        self._ensure_neighbors_found(num_neighbors_temp)

        if not self._has_query_positions(query_positions):
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
                tensor_options,
                at_particles=True,
            )
            return

        query_ghosts = self._route_queries(query_positions, num_neighbors_temp)
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
            tensor_options,
            at_particles=False,
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
        Results are stored in ``smoothing.density_isotropic`` or
        ``smoothing.density_covariant`` and are rank-local (use
        :meth:`get`, e.g. ``pc.get("density_isotropic")``, to collect them).
        Requires :meth:`compute_smoothing` at the particle positions with
        the same structure.

        Only the `num_neighbors` nearest neighbors are summed, so kernels
        whose compact support exceeds 1 (in units of h) are scaled such that
        it ends at the smoothing length (the distance to the farthest
        neighbor) or, for covariant smoothing, on the smoothing-tensor
        ellipsoid (h -> h / support, H -> H / support). Deposition instead
        uses a footprint of support * h, so these kernels (e.g. 'gaussian',
        'quintic_spline', 'tsc', the Wendland kernels) have different
        physical widths in the two modes. `min_axis` applies to H before
        this rescaling.

        """
        st = self._resolve_structure(structure)
        check_particle_structure(st, "density")
        kn = self._resolve_kernel(kernel_name)
        self._check_smoothing_computed(st)
        if self.smoothing.nn_inds is None:
            raise ValueError(
                "No neighbor data for the density estimate: smoothing data set "
                "via 'set_smoothing' can only be used for 'deposit'. Call "
                "'compute_smoothing' first."
            )

        if self.verbose and self.rank == 0:
            print(f"[smudgy] Computing density using {st} '{kn}' kernel")

        nn_inds_local = self.smoothing.nn_inds
        combined_positions, combined_weights = self._combined_positions_weights(
            self.smoothing.ghost_positions, self.smoothing.ghost_weights
        )
        density = execution._dispatch(
            "compute_density",
            backend=self.backend,
            kernel_name=kn,
            dim=self.dim,
            query_positions=self.decomposition.local_positions,
            positions=combined_positions,
            weights=combined_weights,
            nn_inds=nn_inds_local,
            h=(
                self.smoothing.smoothing_tensors
                if st == "covariant"
                else self.smoothing.smoothing_lengths
            ),
            boxsize=self.boxsize,
            structure=st,
        )

        self.smoothing.kernel_name = kn
        self.smoothing.density_kernel[st] = kn

        if st == "covariant":
            self.smoothing.density_covariant = density
        else:
            self.smoothing.density_isotropic = density

    def add_fields(
        self, names: str | list[str], values: npt.ArrayLike | list[npt.ArrayLike]
    ) -> None:
        """Add one or multiple per-particle fields to the point cloud.

        The fields are distributed across the MPI ranks in the same way as the
        particles, and stored as rank-local attributes.

        Parameters
        ----------
        names : str or list of str
            Name(s) of the field(s) to add. Must not collide with an existing
            PointCloud attribute; re-adding a field overwrites it.
        values : array_like or list of array_like
            Field values in the original particle order, each of shape (N,)
            or (N, num_components) with N the *global* particle count. Only
            read on rank 0 (other ranks may pass None).
        """
        # Several fields: decided by `names` alone (identical on every rank);
        # `values` is only checked on rank 0, where it is read.
        if isinstance(names, (list, tuple)):
            error_msg = None
            if self.rank == 0 and (
                not isinstance(values, (list, tuple)) or len(values) != len(names)
            ):
                error_msg = (
                    "For several fields, 'values' must be a list/tuple with one "
                    "array per name."
                )
            error_msg = execution._bcast(self.comm, error_msg)
            if error_msg is not None:
                raise ValueError(error_msg)
            for i, name in enumerate(names):
                self.add_fields(name, values[i] if self.rank == 0 else None)
            return

        name = names
        if not isinstance(name, str):
            raise TypeError(f"Field names must be strings, got {type(name).__name__}.")
        if hasattr(self, name) and name not in self._fields:
            raise ValueError(
                f"'{name}' is an existing PointCloud attribute and cannot be "
                "used as a field name."
            )
        # values are read and checked on rank 0; every rank raises together
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
        self._fields.add(name)

    def delete_fields(self, names: str | list[str]) -> None:
        """Delete one or multiple custom fields from the PointCloud instance.

        Parameters
        ----------
        names : str or list of str
            Name(s) of the field(s) to delete.

        """
        if isinstance(names, (list, tuple)):
            for name in names:
                self.delete_fields(name)
            return

        name = names

        if name in self._fields:
            delattr(self, name)
            self._fields.discard(name)
        elif self.rank == 0:
            print(f"No field named '{name}' found on PointCloud instance to delete.")

    def set_smoothing(
        self,
        structure: Structure | None = None,
        smoothing_lengths: npt.ArrayLike | None = None,
        smoothing_tensors: npt.ArrayLike | None = None,
        smoothing_tensors_eigvals: npt.ArrayLike | None = None,
        smoothing_tensors_eigvecs: npt.ArrayLike | None = None,
    ) -> None:
        """Manually assign smoothing lengths or tensors to particles.

        Intended for deposition with externally computed smoothing
        information (e.g. smoothing lengths stored in a simulation
        snapshot). All arrays are indexed like ``self.positions``, i.e. they
        are this rank's local, Hilbert-ordered slice (on a single rank: the
        original particle order).

        Parameters
        ----------
        structure : Structure, optional
            Smoothing structure the values belong to: ``"isotropic"`` or
            ``"separable"`` for ``smoothing_lengths``, ``"covariant"`` for
            ``smoothing_tensors``.
        smoothing_lengths : npt.ArrayLike, optional
            Smoothing lengths, shape (N_local,).
        smoothing_tensors : npt.ArrayLike, optional
            Smoothing tensors, shape (N_local, D, D).
        smoothing_tensors_eigvals : npt.ArrayLike, optional
            Eigenvalues (principal smoothing lengths) of the smoothing
            tensors, shape (N_local, D). Computed from ``smoothing_tensors``
            if omitted.
        smoothing_tensors_eigvecs : npt.ArrayLike, optional
            Eigenvectors (principal axes, as columns) of the smoothing
            tensors, shape (N_local, D, D). Computed from
            ``smoothing_tensors`` if omitted.

        Notes
        -----
        Existing density estimates are discarded. The values carry no
        neighbor lists, so they can be used by :meth:`deposit` but not by
        :meth:`compute_density`.

        """
        if smoothing_lengths is not None:
            if structure not in ("separable", "isotropic"):
                raise ValueError(
                    "'structure' must be 'separable' or 'isotropic' when providing "
                    "'smoothing_lengths'"
                )
            self._check_shape(np.asarray(smoothing_lengths), "smoothing_lengths")
            self.smoothing.smoothing_lengths = np.asarray(
                smoothing_lengths, dtype=np.float32
            )

        if smoothing_tensors is not None:
            if structure != "covariant":
                raise ValueError(
                    "'structure' must be 'covariant' when providing 'smoothing_tensors'"
                )
            tensors = np.asarray(smoothing_tensors, dtype=np.float32)
            self._check_shape(tensors, "smoothing_tensors")
            if tensors.shape[1:] != (self.dim, self.dim):
                raise ValueError(
                    f"'smoothing_tensors' must have shape (N, {self.dim}, {self.dim}), "
                    f"got {tensors.shape}"
                )
            self.smoothing.smoothing_tensors = tensors
            if smoothing_tensors_eigvals is None or smoothing_tensors_eigvecs is None:
                # deposition needs the principal axes; same convention as
                # compute_smoothing (descending, fixed eigenvector sign)
                vals, vecs = np.linalg.eigh(tensors.astype(np.float64))
                smoothing_tensors_eigvals = vals[:, ::-1]
                smoothing_tensors_eigvecs = _fix_eigvec_sign(vecs[:, :, ::-1])

        self.smoothing.at_particles = True
        self._reset_densities()

        if smoothing_tensors_eigvals is not None:
            self._check_shape(
                np.asarray(smoothing_tensors_eigvals), "smoothing_tensors_eigvals"
            )
            self.smoothing.smoothing_tensors_eigvals = np.asarray(
                smoothing_tensors_eigvals, dtype=np.float32
            )

        if smoothing_tensors_eigvecs is not None:
            self._check_shape(
                np.asarray(smoothing_tensors_eigvecs), "smoothing_tensors_eigvecs"
            )
            self.smoothing.smoothing_tensors_eigvecs = np.asarray(
                smoothing_tensors_eigvecs, dtype=np.float32
            )

    def _print_interpolation(self, mode, where, structure, kernel_name):
        """Verbose rank-0 message shared by both interpolate() branches."""
        if self.verbose and self.rank == 0:
            what = {
                "field": "fields",
                "gradient": "gradients of fields",
                "divergence": "divergence of fields",
                "curl": "curl of fields",
            }[mode]
            print(
                f"[smudgy] Interpolating {what} at {where} using {structure} "
                f"'{kernel_name}' kernel"
            )

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
        """Smoothing data ``(h, eigvals, eigvecs)`` of the particles `idx` for deposit.

        `idx` is a slice or index array into this rank's particles.
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
                _, vals, vecs = execution._dispatch(
                    "project_2d",
                    backend=self.backend,
                    h_tensor=self.smoothing.smoothing_tensors[idx],
                    plane=plane_projection,
                )
            else:
                vals = self.smoothing.smoothing_tensors_eigvals[idx]
                vecs = self.smoothing.smoothing_tensors_eigvecs[idx]
            return None, vals, vecs

        raise ValueError(f"Unsupported deposition structure '{structure}'")

    def _combined_volumes_and_h(self, ghost_info, structure, combined_weights):
        """Local + ghost particle volumes m / rho and smoothing h of `structure`.

        Also returns a scratch dict for the backend's values derived from h
        (None if not cached).

        For the particle ghost set (``self.ghosts``) the result is cached per
        structure, density and smoothing: repeated :meth:`interpolate` calls
        at the particle positions skip the two ghost exchanges. These inputs
        are replaced, never modified in place, by the calls that change them,
        so an identity match is a valid hit, and the decision is the same on
        every rank. Query-position ghost sets are new on every call.
        """
        covariant = structure == "covariant"
        density = (
            self.smoothing.density_covariant if covariant else self.smoothing.density_isotropic
        )
        h_local = (
            self.smoothing.smoothing_tensors if covariant else self.smoothing.smoothing_lengths
        )
        key = (ghost_info, density, h_local, combined_weights)
        cacheable = ghost_info is self.ghosts
        cache = self._ghost_smoothing_cache.get(structure)
        if cacheable and cache is not None and all(a is b for a, b in zip(cache[0], key)):
            return cache[1]

        ghost_density = push_to_ghosts(self.comm, ghost_info, density)
        combined_density = self._combined_local_and_ghost(density, ghost_density)
        # per-particle m_j / rho_j; the backends gather it through nn_inds
        weights_over_density = combined_weights / (combined_density + 1e-8)
        ghost_h = push_to_ghosts(self.comm, ghost_info, h_local)
        combined_h = self._combined_local_and_ghost(h_local, ghost_h)
        if not cacheable:
            return weights_over_density, combined_h, None
        result = (weights_over_density, combined_h, {})
        self._ghost_smoothing_cache[structure] = (key, result)
        return result

    def _interpolate_local(
        self,
        query_positions_local: npt.NDArray[np.floating],
        ghost_info: GhostInfo,
        fields: npt.NDArray[np.floating],
        fields_sizes: list[int],
        mode: InterpolationMode,
        structure_temp: Structure,
    ) -> npt.NDArray[np.floating]:
        """Interpolate `fields` at this rank's targets (rank-local result).

        `ghost_info` holds the targets' neighbor lists into the local
        particles followed by its ghosts; the ghosts' fields, densities and
        smoothing are fetched from their owning ranks with `push_to_ghosts`.
        """
        nn_inds_local = ghost_info.nn_inds
        combined_positions, combined_weights = self._combined_positions_weights(
            ghost_info.ghost_positions, ghost_info.ghost_weights
        )

        ghost_fields = push_to_ghosts(self.comm, ghost_info, fields)
        combined_fields = self._combined_local_and_ghost(fields, ghost_fields)

        if mode in ("divergence", "curl"):
            num_fields = len(fields_sizes)
            combined_fields = combined_fields.reshape(
                combined_fields.shape[0], num_fields, self.dim
            )

        weights_over_density, combined_h, backend_cache = self._combined_volumes_and_h(
            ghost_info, structure_temp, combined_weights
        )

        return execution._dispatch(
            "interpolate",
            backend=self.backend,
            kernel_name=self.smoothing.density_kernel[structure_temp],
            dim=self.dim,
            query_positions=query_positions_local,
            positions=combined_positions,
            weights=weights_over_density,
            nn_inds=nn_inds_local,
            fields=combined_fields,
            h=combined_h,
            boxsize=self.boxsize,
            mode=mode,
            structure=structure_temp,
            cache=backend_cache,
        )

    def interpolate(
        self,
        fields: npt.ArrayLike | str | list[str],
        query_positions: npt.ArrayLike = None,
        mode: InterpolationMode = "field",
        structure: Structure | None = None,
    ) -> npt.NDArray[np.floating]:
        r"""Interpolate particle fields to query positions using SPH.

        Compute interpolated field values, gradients, divergence, or curl at
        query positions using the scatter form of the SPH estimate.

        Parameters
        ----------
        fields : Union[npt.ArrayLike, str, List[str]]
            Field data to interpolate: a field name registered via
            :meth:`add_fields`, an array, or a list of both. Arrays passed
            directly must already be this rank's local slice (indexed like
            ``self.positions``).
        query_positions : npt.ArrayLike, optional
            Array of shape (M, D) with positions where quantities are
            computed; only read on rank 0. If None, uses the particle
            positions.
        mode : InterpolationMode, default 'field'
            Type of quantity to compute:

            - ``'field'``: interpolated field values
            - ``'gradient'``: field gradients
            - ``'divergence'``: divergence (vector fields only)
            - ``'curl'``: curl (vector fields only)
        structure : Structure, optional
            Smoothing structure, ``'isotropic'`` or ``'covariant'``. If None,
            uses the globally set structure.

        Returns
        -------
        npt.NDArray[np.floating]
            Interpolated quantities, where M is the number of (rank-local)
            query points, C the total number of field components (summed
            over all fields), F the number of fields and D the spatial
            dimension:

            - ``'field'``: (M, C)
            - ``'gradient'``: (M, C, D)
            - ``'divergence'``: (M, F)
            - ``'curl'``: (M, F) in 2D, (M, F, 3) in 3D

            The result is rank-local: with ``query_positions=None`` its rows
            follow ``self.positions``; otherwise they follow the routed query
            points. Use :meth:`gather_particles` or :meth:`gather_queries`,
            respectively, to collect it in the original order on the root
            rank (on a single rank, both return the input order unchanged).

        Notes
        -----
        For particles :math:`i` in the neighbourhood :math:`\mathcal{N}` of
        :math:`\boldsymbol{x}`, with masses (weights) :math:`m_i`, densities
        :math:`\rho_i` and smoothing lengths :math:`h_i` (or tensors
        :math:`\mathsf{H}_i`):

        .. math::

            f(\boldsymbol{x}) = \sum_{i \in \mathcal{N}} \frac{m_i}{\rho_i}
            f_i \, W(\boldsymbol{x} - \boldsymbol{x}_i, h_i),
            \qquad
            \nabla f(\boldsymbol{x}) = \sum_{i \in \mathcal{N}}
            \frac{m_i}{\rho_i} f_i \, \nabla W(\boldsymbol{x} -
            \boldsymbol{x}_i, h_i),

        and for a vector field :math:`\boldsymbol{v}`, the divergence and
        curl replace :math:`f_i \nabla W` by
        :math:`\boldsymbol{v}_i \cdot \nabla W` and
        :math:`\boldsymbol{v}_i \times \nabla W`, respectively (in 2D the
        curl is the scalar :math:`z`-component).

        The densities must have been computed beforehand with
        :meth:`compute_density` for the same structure. As in
        :meth:`compute_density`, only the nearest neighbors are summed, and
        kernels with support :math:`n > 1` are scaled so their compact
        support ends at each particle's smoothing length / tensor ellipsoid
        (:math:`h \to h / n`), unlike deposition, which uses a footprint of
        :math:`n h`.

        Examples
        --------
        >>> pc = PointCloud(positions, weights=masses)
        >>> pc.global_setup(kernel_name='cubic_spline', structure='isotropic', num_neighbors=32)
        >>> pc.compute_smoothing()
        >>> pc.compute_density()
        >>> pc.add_fields('velocity', velocity_data)
        >>> query_positions = np.random.uniform(0, 1, size=(1000, 3))

        Interpolate field values and collect them on the root rank:

        >>> values = pc.gather_queries(pc.interpolate('velocity', query_positions))

        Compute gradients, divergence and curl (vector fields):

        >>> grads = pc.interpolate('velocity', query_positions, mode='gradient')
        >>> div = pc.interpolate('velocity', query_positions, mode='divergence')
        >>> curl = pc.interpolate('velocity', query_positions, mode='curl')

        """
        check_interpolation_mode(mode, self.dim)
        structure_temp = self._resolve_structure(structure)
        check_particle_structure(structure_temp, "interpolation")
        # a density implies smoothing data at the particles and their neighbors
        self._check_density_computed(structure_temp)
        kernel_name = self.smoothing.density_kernel[structure_temp]
        num_neighbors = self.smoothing.num_neighbors

        fields, fields_sizes = self._resolve_fields(fields)
        self._check_field_dimensionality(mode, fields_sizes)

        if not self._has_query_positions(query_positions):
            if self.ghosts.nn_inds.shape[1] != num_neighbors:
                raise ValueError(
                    f"The neighbor lists ({self.ghosts.nn_inds.shape[1]} neighbors, "
                    "from the last 'find_neighbors' call) do not match the smoothing "
                    f"data ({num_neighbors} neighbors); call 'compute_smoothing' and "
                    "'compute_density' again."
                )
            self._print_interpolation(
                mode, "particle positions", structure_temp, kernel_name
            )
            return self._interpolate_local(
                self.decomposition.local_positions,
                self.ghosts,
                fields,
                fields_sizes,
                mode,
                structure_temp,
            )

        query_ghosts = self._route_queries(query_positions, num_neighbors)
        self._print_interpolation(mode, "query positions", structure_temp, kernel_name)
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
        fields: npt.ArrayLike | str | list[str] | None,
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
        gather_to_root: bool = True,
        root: int = 0,
        brick: bool | None = None,
    ) -> (
        npt.NDArray[np.floating]
        | tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
    ):
        r"""Deposit particle fields onto a structured grid.

        For every grid cell, the deposited value is

        .. math::

            f_{\rm cell} = \sum_{i} m_i \, f_i \, I_{{\rm cell},i},
            \qquad
            I_{{\rm cell},i} = \int_{\Omega_{\rm cell}}
            W(\boldsymbol{x} - \boldsymbol{x}_i, \mathsf{H}_i) \,
            \mathrm{d}\boldsymbol{x},

        i.e. every field is multiplied by the particle weights :math:`m_i`
        internally, and the weights grid is :math:`\sum_i m_i I_{{\rm
        cell},i}` (e.g. the mass in each cell). Fields are therefore
        *specific* quantities (per unit weight): to obtain a mass map, use
        ``fields=None`` (do not deposit the weights themselves as a field,
        which would give :math:`\sum_i m_i^2 I_{{\rm cell},i}`).

        Parameters
        ----------
        fields : Union[npt.ArrayLike, str, List[str], None]
            Field data to deposit: a field name registered via
            :meth:`add_fields`, an array, or a list of both. Arrays passed
            directly must already be this rank's local slice. ``None``
            deposits only the particle weights (e.g. a mass map) and returns
            just the weights grid; ``averaged`` must then be False.
        averaged : Union[bool, Sequence[bool]]
            Whether to divide the result by the weights grid for each field
            (a weight-averaged quantity, e.g. temperature) or not (an
            additive quantity).
        gridnums : Union[int, Sequence[int]]
            Number of cells along each axis.
        extent : Optional[Sequence[Sequence[float]]], optional
            Domain bounds ``[[xmin, xmax], [ymin, ymax], ...]``, one row per
            dimension of the point cloud. If given, the grid spans exactly
            this extent with `gridnums` cells, is non-periodic (whether or
            not `boxsize` was set), and only particles inside are deposited.
            If None, uses ``[0, boxsize]`` with the cloud's periodicity;
            then `boxsize` must be set.
        adaptive : bool, default True
            If True, use the instance's smoothing data (see
            :meth:`compute_smoothing`) with the selected structure and
            kernel. If False, use a fixed-grid B-spline stencil and do not
            consult smoothing data.
        kernel_name : str, optional
            Fixed-grid stencil (``ngp``, ``cic``, ``tsc``, ``bspline4``,
            ``bspline5``, or ``bspline6``) when ``adaptive=False``. For
            adaptive deposition, ``tophat``, ``tsc`` or ``gaussian`` for the
            ``separable`` structure, or any radial kernel for ``isotropic``
            and ``covariant``.
        structure : Structure, optional
            Adaptive smoothing structure. Required only when ``adaptive=True``.
        plane_projection : List[int], optional
            Indices of the two axes spanning the projection plane for 3D to
            2D deposition, e.g. ``[0, 1]`` for the x-y plane. The result is
            the column density: each particle deposits its 3D kernel
            integrated along the remaining axis (for covariant smoothing,
            the shadow of its ellipsoid with the projected radial profile),
            i.e. the 3D deposit summed along that axis, up to cell
            quadrature. Particles are selected by their centre: with
            ``extent``, a kernel straddling the slab boundary along the
            projection axis is counted whole or not at all.
        integration_method : str, default 'midpoint'
            Cell quadrature rule for ``isotropic`` and ``covariant`` kernels:
            ``'midpoint'``, ``'trapezoidal'`` or ``'simpson'``.
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
            If True, returns the weights grid as well.
        gather_to_root : bool, default True
            Only with several MPI ranks. True: the summed grid ends up on
            `root` only; the other ranks return ``None`` (or ``(None, None)``
            if ``return_weights=True``). False: every rank receives the full
            grid (``MPI.Allreduce``) -- slower and more memory; use it only
            when every rank needs the grid.
        root : int, default 0
            Destination rank when ``gather_to_root=True``; unused otherwise.
        brick : bool or None, default None
            Only with several MPI ranks; a single rank always deposits onto
            the full grid and sums nothing. True: every rank deposits onto the
            smallest box of cells its particles touch (a brick) and `root`
            sums the bricks into the full grid (then broadcasts it if
            ``gather_to_root=False``). False: every rank deposits onto a
            full-size grid, summed with ``MPI.Reduce``/``Allreduce``. None
            (default) picks the faster one: bricks with ``gather_to_root=True``
            and the ``"quadrants"`` backend (less memory on non-root ranks),
            full grids otherwise (``Allreduce`` beats bricks plus a broadcast
            when every rank receives the grid). Both give the same grid up to
            the order of floating-point additions across ranks.

        Returns
        -------
        Union[npt.NDArray[np.floating], tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]]
            Deposited field grid of shape ``(C, *gridnums)`` with C the total
            number of field components, and optionally the weights grid of
            shape ``gridnums``. With ``fields=None`` only the weights grid is
            returned.

        Notes
        -----
        The numpy backend supports fixed-grid deposition (``adaptive=False``)
        only.

        """
        kernel_name_temp, structure_temp = self._resolve_deposition_options(
            adaptive, kernel_name, structure
        )
        # fields=None: mass-only deposition (weights grid only, F=0 -- every
        # backend kernel already skips its per-field loop for F=0).
        weights_only = fields is None
        if weights_only:
            if np.any(averaged):
                raise ValueError("'averaged' must be False when fields=None")
            fields = np.empty((self.decomposition.local_positions.shape[0], 0), np.float32)
            fields_sizes = []
        else:
            fields, fields_sizes = self._resolve_fields(fields, dtype=np.float32)
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
            # an explicit extent always wins over boxsize: non-periodic grid
            # spanning exactly [min, max] per axis, whether or not the cloud
            # was built with a (periodic) boxsize
            ext = np.asarray(extent, dtype=np.float32)
            if ext.shape != (self.dim, 2):
                raise ValueError(
                    f"'extent' must have shape ({self.dim}, 2), got {ext.shape}"
                )
            if np.any(ext[:, 1] <= ext[:, 0]):
                raise ValueError(f"'extent' needs max > min on every axis, got {ext.tolist()}")
            domain_min, domain_max, periodic = ext[:, 0], ext[:, 1], False

        d_lens = domain_max - domain_min

        # idx_local selects this rank's particles inside the extent. Without
        # an extent every (wrapped) particle is inside, and slice(None) keeps
        # all gathers below views instead of copies.
        local_positions = self.decomposition.local_positions
        if extent is None:
            idx_local = slice(None)
        else:
            # column-wise: np.all(..., axis=1) over a length-D axis is
            # pathologically slow in numpy
            mask = np.ones(local_positions.shape[0], dtype=bool)
            for d in range(self.dim):
                col = local_positions[:, d]
                mask &= (col >= domain_min[d]) & (col <= domain_max[d])
            idx_local = slice(None) if mask.all() else np.flatnonzero(mask)
        pos_temp = local_positions[idx_local]
        if np.any(domain_min != 0):
            pos_temp = pos_temp - domain_min
        weights_temp = self.decomposition.local_weights[idx_local]
        fields_temp = fields[idx_local]

        # if plane_projection is set, collect relevant axes
        if plane_projection is not None:
            if self.dim != 3:
                raise ValueError(
                    f"Plane projection requires 3D positions, but positions are {self.dim}d"
                )
            plane_projection = np.asarray(plane_projection)
            if (
                plane_projection.shape != (2,)
                or not np.issubdtype(plane_projection.dtype, np.integer)
                or plane_projection[0] == plane_projection[1]
                or np.any((plane_projection < 0) | (plane_projection > 2))
            ):
                raise ValueError(
                    "'plane_projection' must be two distinct axis indices in "
                    f"[0, 2], got {plane_projection.tolist()}"
                )
            plane_projection = [int(a) for a in plane_projection]

            pos_temp = np.take(pos_temp, plane_projection, axis=1)
            d_lens = np.take(d_lens, plane_projection).astype(np.float32)

        # resolve deposition dimension and gridnums
        dep_dim = pos_temp.shape[1]
        gridnums_temp = self._resolve_gridnums(gridnums, dep_dim)

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
        if plane_projection is not None and structure_temp in ("isotropic", "covariant"):
            # the line-of-sight integral of the 3D kernel; separable kernels
            # and fixed-grid stencils project exactly as they are
            deposit_kwargs["projected"] = True

        if adaptive:
            if structure_temp in ("separable", "isotropic"):
                deposit_kwargs["particle_hsml"] = h
            else:
                deposit_kwargs["particle_hmat_eigvecs"] = h_vecs
                deposit_kwargs["particle_hmat_eigvals"] = h_vals

        if self.verbose and self.rank == 0:
            structure_prefix = f"{structure_temp} " if structure_temp else ""
            print(f"[smudgy] Depositing using {structure_prefix}'{kernel_name_temp}' kernel")

        if brick is None:
            brick = gather_to_root and self.backend == "quadrants"
        if self.size == 1:
            grids = execution._dispatch("deposit", backend=self.backend, **deposit_kwargs)
        elif brick:
            # every rank deposits onto its own brick; root sums the bricks
            *grids, offset = execution._dispatch(
                "deposit", backend=self.backend, brick=True, **deposit_kwargs
            )
            grids = execution._sum_bricks(
                self.comm, grids, offset, gridnums_temp,
                root=root if gather_to_root else 0, to_all=not gather_to_root,
            )
        else:
            # every rank deposits onto a full grid; the grids are summed onto
            # `root` only or onto every rank
            grids = execution._dispatch("deposit", backend=self.backend, **deposit_kwargs)
            if gather_to_root:
                grids = execution._reduce_sum_to_root(self.comm, grids, root=root)
            else:
                grids = execution._reduce_sum(self.comm, grids)
        fields_grid, weights_grid = grids

        if weights_only:
            return weights_grid

        if fields_grid is not None:
            for i, avg in enumerate(averaged):
                if avg:
                    fields_grid[i] /= weights_grid + 1e-10

        return (fields_grid, weights_grid) if return_weights else fields_grid
