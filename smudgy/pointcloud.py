"""Core PointCloud class for particle-based computations."""

import warnings
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
        # Initialize backend
        self.verbose = verbose
        self.set_backend(backend, **kwargs)

        # Initialize MPI environment
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        if verbose:
            print(f'[smudgy] Using {size} MPI rank{"s" if size > 1 else ""}')

        # Initialize point cloud
        self.dim = positions.shape[-1]
        assert self.dim in (
            1,
            2,
            3,
        ), f"Particle positions must be of shape (N, 1), (N, 2) or (N, 3) but found {positions.shape}"
        self.positions = positions.astype(np.float32)

        self.weights = (
            np.ones(self.positions.shape[0], dtype=np.float32)
            if weights is None
            else weights.astype(np.float32)
        )
        assert (
            self.weights.shape[0] == self.positions.shape[0]
        ), f"Shape mismatch: length of weights and positions must be the same but found: {self.weights.shape} and {self.positions.shape}"

        if boxsize is None:
            self.periodic = False
            self.boxsize = None
        else:
            self.periodic = True
            boxsize_arr = np.asarray(boxsize)
            if boxsize_arr.ndim == 0:
                self.boxsize = np.repeat(boxsize_arr, self.dim)
            else:
                assert boxsize_arr.shape == (
                    self.dim,
                ), f"'boxsize' must be a scalar or have shape ({self.dim},), got {boxsize_arr.shape}"
                self.boxsize = boxsize_arr

        self.smoothing = SmoothingInfo()

        # Verbose output after completed initialization
        if self.verbose:
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
        if self.verbose:
            print(f"[smudgy] Set {backend} backend")

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

    def _check_neighbors(self, nn_inds: npt.NDArray[np.integer]) -> None:
        """Verify that neighbor indices are within valid bounds."""
        max_idx = np.max(nn_inds)
        if max_idx >= self.positions.shape[0]:
            raise IndexError(
                f"Neighbor index {max_idx} is out of bounds for {self.positions.shape[0]} particles. This indicates a bug in the neighbor search or input setup."
            )

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
        """Ensure a kd-tree exists for neighbor searches."""
        if self.smoothing.tree is None:
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
        tree = self._check_tree()

        qpos = (
            tree.data
            if query_positions is None
            else np.asarray(query_positions, dtype=np.float32)
        )
        nn_dists, nn_inds = query_kdtree(tree, qpos, k=num_neighbors_temp)

        if self.verbose:
            info_str = "tensors" if structure_temp == "covariant" else "lengths"
            print(
                f"[smudgy] Computing smoothing {info_str} from {num_neighbors_temp} neighbors"
            )

        if structure_temp in ("separable", "isotropic"):
            self.smoothing.smoothing_lengths = execution._dispatch(
                "compute_hsml",
                backend=self.backend,
                nn_dists=nn_dists,
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
                query_positions=qpos,
                neighbor_positions=tree.data[nn_inds],
                neighbor_weights=self.weights[nn_inds],
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

        if self.verbose:
            print(f"[smudgy] Computing density using " f"{st} '{kn}' kernel")

        density = execution._dispatch(
            "compute_density",
            backend=self.backend,
            kernel_name=kn,
            dim=self.dim,
            neighbor_weights=self.weights[self.smoothing.nn_inds],
            r_ij=self._get_rel_coords(
                self.positions, self.positions[self.smoothing.nn_inds]
            ),
            h=(
                self.smoothing.smoothing_tensors
                if st == "covariant"
                else self.smoothing.smoothing_lengths
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
        values_arr = np.asarray(values, dtype=np.float32)

        if hasattr(self, name):
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
        mask: npt.NDArray[np.bool_],
        dim: int,
        plane_projection: list[int] | None = None,
    ) -> tuple[
        npt.NDArray[np.float32] | None,
        npt.NDArray[np.float32] | None,
        npt.NDArray[np.float32] | None,
    ]:
        """Prepare smoothing data (h, h_vals, h_vecs) for an adaptive deposition call."""
        self._check_smoothing_computed(
            "isotropic" if structure == "separable" else structure
        )

        if structure == "separable":
            hsml = self.smoothing.smoothing_lengths[mask]
            return np.repeat(hsml[:, np.newaxis], dim, axis=1), None, None

        if structure == "isotropic":
            return self.smoothing.smoothing_lengths[mask], None, None

        if structure == "covariant":
            if plane_projection is not None:
                _, vals, vecs = execution._dispatch(
                    "project_2d",
                    backend=self.backend,
                    h_tensor=self.smoothing.smoothing_tensors[mask],
                    plane=plane_projection,
                )
            else:
                vals = self.smoothing.smoothing_tensors_eigvals[mask]
                vecs = self.smoothing.smoothing_tensors_eigvecs[mask]
            return None, vals, vecs

        raise ValueError(f"Unsupported deposition structure '{structure}'")

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
        if structure_temp not in ['isotropic', 'covariant']:
            raise ValueError(
                f"For interpolation, 'structure' must be one of ['isotropic', 'covariant'], got '{structure_temp}'"
            )

        # check that density has been computed for the chosen structure
        self._check_density_computed(structure_temp)

        # cast fields to correct input format and validate their dimensionality for the requested mode
        fields, fields_sizes = self._resolve_fields(fields)
        self._check_field_dimensionality(mode, fields_sizes)

        # if query_positions is None, use particle positions
        if query_positions is None:
            query_positions = self.positions
            nn_inds = self.smoothing.nn_inds
        else:
            # for new query positions, need to perform a new neighbor search
            tree = self._check_tree()
            _, nn_inds = query_kdtree(
                tree,
                query_positions,
                k=self.num_neighbors,
            )

        # ----------------------------
        # Input preparation
        # ----------------------------
        # compute relative coordinates
        r_ij = self._get_rel_coords(query_positions, self.positions[nn_inds])

        # prepare interpolation weights and fields
        fields_temp = fields[nn_inds]  # Shape: (M, K, num_fields)

        # For divergence and curl, reshape fields to (M, K, num_fields, D)
        # This enables clean einsum patterns since all fields are validated to have self.dim components
        if mode in ("divergence", "curl"):
            num_fields = len(fields_sizes)
            fields_temp = fields_temp.reshape(
                fields_temp.shape[0], fields_temp.shape[1], num_fields, self.dim
            )

        density_temp = (
            self.smoothing.density_covariant[nn_inds]
            if structure_temp == "covariant"
            else self.smoothing.density_isotropic[nn_inds]
        )
        weights_temp = self.weights[nn_inds] / (density_temp + 1e-8)
        h_temp = (
            self.smoothing.smoothing_tensors[nn_inds]
            if structure_temp == "covariant"
            else self.smoothing.smoothing_lengths[nn_inds]
        )

        # ----------------------------
        # Verbose output / computation
        # ----------------------------
        if self.verbose:
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
        num_kernel_evaluations_per_axis: int = 4,
        eta_crit: float = 10.0,
        return_weights: bool = False,
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
        num_kernel_evaluations_per_axis : int, default 4
            Resolution for kernel integration.
        eta_crit : float, default 10.0
            Anti-aliasing threshold, in units of smoothing length / cell
            size, to switch a particle from the sampled kernel deposition to
            the per-cell numerical quadrature deposition. The quadrature
            branch uses a small, fixed number of quadrature points per grid
            cell and cannot reliably resolve kernels whose smoothing length
            is comparable to or smaller than several grid cells; below
            eta_crit ~ 10 it can silently under-sample the kernel and fail
            to conserve mass, in the worst case dropping a particle's
            contribution entirely. Do not lower this below 10 unless you
            have verified mass conservation for your specific grid
            resolution and kernel choice.
        return_weights : bool, default False
            If True, returns the weights (density) grid as well.
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
        if adaptive and eta_crit < 10.0:
            warnings.warn(
                f"eta_crit={eta_crit} < 10: narrow kernels will be under-sampled by "
                "quadrature deposition and mass conservation is not guaranteed.",
                stacklevel=2,
            )

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
        mask = np.all(
            (self.positions >= domain_min) & (self.positions <= domain_max),
            axis=1,
        )
        pos_temp = self.positions[mask] - domain_min
        weights_temp = self.weights[mask]
        fields_temp = fields[mask]

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
                mask,
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
            "num_kernel_evaluations_per_axis": num_kernel_evaluations_per_axis,
            "eta_crit": eta_crit,
        }

        if structure_temp in ("separable", "isotropic"):
            deposit_kwargs["particle_hsml"] = h
        else:
            deposit_kwargs["particle_hmat_eigvecs"] = h_vecs
            deposit_kwargs["particle_hmat_eigvals"] = h_vals

        # ----------------------------
        # Verbose output / computation
        # ----------------------------
        if self.verbose:
            print(
                f"[smudgy] Depositing using "
                f"{structure_temp if structure_temp else ''} '{kernel_name_temp}' kernel"
                )

        fields_grid, weights_grid = execution._dispatch(
            "deposit",
            backend=self.backend,
            **deposit_kwargs,
        )

        for i, avg in enumerate(averaged):
            if avg:
                fields_grid[i] /= weights_grid + 1e-10

        return (fields_grid, weights_grid) if return_weights else fields_grid
