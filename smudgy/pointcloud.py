"""Core SPH operations and PointCloud class for particle-based computations."""

from collections.abc import Sequence
from typing import List, Literal

import numpy as np
import numpy.typing as npt

from .core.kernels import get_kernel
from .core import collection as backend
from .smooth import SmoothingInfo
from .utils import (
    build_kdtree,
    compute_smoLens,
    compute_smoTens,
    coordinate_difference_with_pbc,
    project_smoTens_to_2d,
)

GEOMETRY_NAMES = ("separable", "isotropic", "anisotropic")
Geometry = Literal["separable", "isotropic", "anisotropic"]


class PointCloud:
    """Represent a collection of particles for operations."""

    def __init__(
        self,
        positions: np.ndarray,
        weights: np.ndarray,
        boxsize: float | Sequence[float] | None = None,
        verbose: bool = True,
    ):
        """Represent a collection of particles for operations.

        Parameters
        ----------
        positions : np.ndarray
                Particle positions, shape (N, D).
        weights : np.ndarray
                Particle weights (e.g. masses), shape (N,).
        boxsize : float or array-like, optional
                If provided, all axes are periodic with the given boxsize(s). If None, no periodicity is used.
        verbose : bool
                Verbosity flag.

        """
        self.dim = positions.shape[-1]
        assert (
            self.dim == 2 or self.dim == 3
        ), f"Particle positions must be of shape (N, 2) or (N, 3) but found {positions.shape}"
        self.positions = positions

        self.weights = weights
        assert (
            self.weights.shape[0] == self.positions.shape[0]
        ), f"Shape mismatch: length of weights and positions must be the same but found: {self.weights.shape} and {self.positions.shape}"

        self.verbose = verbose

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

        # initialize empty attributes for smoothing info and density
        self.supported_geometries = GEOMETRY_NAMES
        self.smoothing = {}
        for geom in self.supported_geometries:
            self.smoothing[geom] = SmoothingInfo()

        if self.verbose:
            periodic_str = (
                f"in periodic box of size={self.boxsize}"
                if self.periodic
                else "without periodicity"
            )
            print(
                f"[smudgy] Initialized {self.dim}D PointCloud with {self.positions.shape[0]} particles {periodic_str}"
            )
    

    def _set_geometry(self, geometry: Geometry) -> None:
        if geometry not in self.supported_geometries:
            raise ValueError(f"'geometry' must be one of {self.supported_geometries}, but found '{geometry}'")
        self.geometry = geometry

    def _set_kernel(self, kernel_name: str,) -> None:
        assert isinstance(kernel_name, str), f"'kernel_name' must be a string but found {type(kernel_name)}"
        self.kernel = get_kernel(kernel_name, dim=self.dim)

    def _set_num_neighbors(self, num_neighbors: int) -> None:
        self.num_neighbors = num_neighbors


    def _check_property(self, property: str | None = None):
        """Return the requested property, checking argument and global state."""
        if not hasattr(self, property):
            raise AttributeError(
                f"'{property}' has not been set: either set it via 'global_setup' method or provide it as a function argument"
            )
        
    def _check_valid_neighbors(self, nn_inds):
        max_idx = np.max(nn_inds)
        if max_idx >= self.positions.shape[0]:
            raise IndexError(
                f"Neighbor index {max_idx} is out of bounds for {self.positions.shape[0]} particles. This indicates a bug in the neighbor search or input setup."
            )


    def _resolve_geometry(self, geometry):
        """Return the resolved geometry string, checking argument and global state."""
        if geometry is None:
            self._check_property('geometry')
            return self.geometry
        if geometry not in self.supported_geometries:
            raise ValueError(
                f"'geometry' must be one of {self.supported_geometries}, but found '{geometry}'"
            )
        return geometry

    def _resolve_kernel(self, 
                        kernel_name: str | None = None,
                        dim: int | None = None):
        """Return the resolved kernel object, checking argument and global state."""
        if kernel_name is None:
            self._check_property('kernel')
            return self.kernel
        assert isinstance(kernel_name, str), f"'kernel_name' must be a string but found {type(kernel_name)}"
        return get_kernel(kernel_name, dim=dim)

    def _resolve_num_neighbors(self, num_neighbors: int | None = None):
        """Return the resolved number of neighbors, checking argument and global state."""
        if num_neighbors is None:
            self._check_property('num_neighbors')
            return self.num_neighbors
        assert isinstance(num_neighbors, int) and num_neighbors > 0, f"'num_neighbors' must be a positive integer but found {num_neighbors}"
        return num_neighbors
    
    def _resolve_fields(self, fields: npt.ArrayLike | str | List[str]) -> np.ndarray:
        """
        Resolve the input 'fields' argument to a single (N, total_components) ndarray.
        
        Parameters
        ----------
        fields
                Can be a single string (field name), a list of strings, a single array, or a list of arrays. 
                If strings are provided, they are resolved to attributes on the instance.
        
        Returns
        -------
        np.ndarray
                Resolved field data as a single (N, total_components) ndarray.
        """
        # Normalize input to a list
        if isinstance(fields, (str, np.ndarray)):
            fields = [fields]
        elif not isinstance(fields, list):
            raise ValueError("Invalid 'fields' argument: must be a string, list, or numpy array")

        arrays = []
        for f in fields:
            arr = getattr(self, f) if isinstance(f, str) else f
            arr = np.asarray(arr)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            arrays.append(arr)

        # Check all arrays have the same length
        for i, arr in enumerate(arrays):
            if arr.ndim != 2:
                raise ValueError(
                    f"Field at index {i} must have ndim = 2 after reshaping, got {arr.ndim}"
                )
            if arr.ndim > 2:
                raise ValueError(
                    f"Field at index {i} has ndim={arr.ndim}, but only ndim <= 2 is supported."
                )
            self._assert_shape(arr, f"field {i}")

        # Concatenate along last axis
        return np.concatenate(arrays, axis=-1)

    def _assert_shape(self, arr, name):
            assert (
                arr.shape[0] == self.positions.shape[0]
            ), f"Length of '{name}' ({arr.shape[0]}) must match number of particles ({self.positions.shape[0]})"


    def global_setup(self,
                   geometry: Geometry = None,
                   kernel_name: str = None,
                   num_neighbors: int = None,
                   ) -> None:
        """Convenience method to set all global parameters for SPH computations.
        
        Parameters        
        ----------
        geometry
                Optional geometry specification for smoothing and density computation. If ``None``, uses the globally set geometry or raises an error if not set.
        kernel_name
                Name of the SPH kernel.
        num_neighbors
                Number of neighbors to consider for smoothing length computation.

        Returns
        -------
        None
                An instance of the specified kernel class.
        """
        if geometry:
            self._set_geometry(geometry)
        if kernel_name:
            self._set_kernel(kernel_name)
        if num_neighbors:
            self._set_num_neighbors(num_neighbors)
        return self

    def compute_smoothing(
        self, 
        query_positions: npt.ArrayLike | None = None,
        num_neighbors: int | None = None,
        geometry: Geometry | None = None,
    ) -> None:
        """Compute smoothing lengths  for SPH calculations.

        Parameters
        ----------
        query_positions
            Optional array of shape ``(M, D)`` with positions where smoothing is evaluated.
            If ``None``, uses the particle positions from the instance.
        num_neighbors
            Optional number of neighbors to consider for smoothing length computation. 
            If ``None``, uses the value globally set value or raises an error if not set.
        geometry
            Optional geometry specification for smoothing length computation. If ``None``, uses the globally set geometry

        Returns
        -------
        None
            Results are stored on the instance: ``smoLens`` (isotropic),
            ``smoTens``/``smoTens_eigvals``/``smoTens_eigvecs`` (anisotropic), ``nn_inds``
            (all modes), and ``nn_dists`` (isotropic only).

        """

        geometry_temp = self._resolve_geometry(geometry)
        num_neighbors_temp = self._resolve_num_neighbors(num_neighbors)

        if not hasattr(self, "tree"):
            if self.verbose:
                print("[smudgy] Building kd-tree from particle positions")
            self.tree = build_kdtree(self.positions, boxsize=self.boxsize)

        if self.verbose:
            info_str = "tensors" if geometry_temp == "anisotropic" else "lengths"
            print(f"[smudgy] Computing smoothing {info_str} from {num_neighbors_temp} neighbors")

        # construct kwargs
        kwargs = {
            "tree": self.tree,
            "num_neighbors": num_neighbors_temp,
            "query_positions": query_positions,
        }

        if geometry_temp in ["separable", "isotropic"]:
            (
                smoLens, 
                nn_inds, 
                nn_dists 
            ) = compute_smoLens(**kwargs)

            self.smoothing[geometry_temp].smoLens = smoLens
        
        elif geometry_temp == "anisotropic":
            (
                smoTens,
                smoTens_eigvals,
                smoTens_eigvecs,
                nn_inds,
                nn_dists,
                rel_coords
            ) = compute_smoTens(**kwargs, weights=self.weights)

            self.smoothing[geometry_temp].smoTens = smoTens
            self.smoothing[geometry_temp].smoTens_eigvals = smoTens_eigvals
            self.smoothing[geometry_temp].smoTens_eigvecs = smoTens_eigvecs
            self.smoothing[geometry_temp].rel_coords = rel_coords

        self.smoothing[geometry_temp].nn_inds = nn_inds
        self.smoothing[geometry_temp].nn_dists = nn_dists
        self.smoothing[geometry_temp].num_neighbors = num_neighbors_temp
        
        # Safeguard: check for invalid neighbor indices
        self._check_valid_neighbors(self.smoothing[geometry_temp].nn_inds)

    def set_smoothing(
        self,
        geometry: Geometry | None = None,
        smoLens: npt.ArrayLike | None = None,
        smoTens: npt.ArrayLike | None = None,
        smoTens_eigvals: npt.ArrayLike | None = None,
        smoTens_eigvecs: npt.ArrayLike | None = None,
    ):
        """
        Manually assign smoothing lengths or tensors to particles.

        Parameters
        ----------
        smoLens
            Array of shape ``(N,)`` with isotropic smoothing lengths.
        smoTens
            Array of shape ``(N, D, D)`` with anisotropic smoothing tensors.
        smoTens_eigvals
            Array of shape ``(N, D)`` with eigenvalues of the smoothing tensors.
        smoTens_eigvecs
            Array of shape ``(N, D, D)`` with eigenvectors of the smoothing tensors.

        Returns
        -------
        None
            Results are stored on the instance.
        """

        # for smoLens, geometry must be set and either 'separable' or 'isotropic'
        if smoLens:
            assert geometry in ["separable", "isotropic"], "Geometry must be specified when providing 'smoLens'"
            self._assert_shape(smoLens, "smoLens")
            self.smoothing["isotropic"]["smoLens"] = np.asarray(smoLens, dtype=np.float32)
        
        if smoTens:
            self._assert_shape(smoTens, "smoTens")
            self.smoothing["anisotropic"]["smoTens"] = np.asarray(smoTens, dtype=np.float32)

        if smoTens_eigvals:
            self._assert_shape(smoTens_eigvals, "smoTens_eigvals")
            self.smoothing["anisotropic"]["smoTens_eigvals"] = np.asarray(smoTens_eigvals, dtype=np.float32)
            
        if smoTens_eigvecs:
            self._assert_shape(smoTens_eigvecs, "smoTens_eigvecs")
            self.smoothing["anisotropic"]["smoTens_eigvecs"] = np.asarray(smoTens_eigvecs, dtype=np.float32)

    #def _check_smoothing_set(self):
    #    if self.geometry in ["separable", "isotropic"] and not hasattr(self, "smoLens"):
    #        raise AttributeError(
    #            f"Smoothing lengths have not been computed for geometry '{self.geometry}'. Please call 'compute_smoothing_lengths' first."
    #        )
    #    if self.geometry == "anisotropic" and (
    #        not hasattr(self, "smoTens")
    #        or not hasattr(self, "smoTens_eigvals")
    #        or not hasattr(self, "smoTens_eigvecs")
    #    ):
    #        raise AttributeError(
    #            f"Smoothing tensors have not been computed for anisotropic geometry '{self.geometry}'. Please call 'compute_smoothing_lengths' first."
    #        )

    def add_field(self, name: str, values: npt.ArrayLike):
        """Add a custom field to the PointCloud instance.

        Parameters
        ----------
        name
                Name of the field to add.
        values
                Array of shape (N,) or (N, num_components) with field values for each particle.

        Returns
        -------
        None
                The field is added as an attribute to the instance.
        """
        if hasattr(self, name):
            print(f"Overwriting existing attribute '{name}' on PointCloud instance.")
        values_arr = np.asarray(values, dtype=np.float32)
        self._assert_shape(values_arr, name)
        setattr(self, name, values_arr)

    def delete_field(self, name: str):
        """Delete a custom field from the PointCloud instance.

        Parameters
        ----------
        name
                Name of the field to delete.

        Returns
        -------
        None
                The field is removed from the instance attributes.
        """
        if hasattr(self, name):
            delattr(self, name)
        else:
            print(f"No attribute named '{name}' found on PointCloud instance to delete.")

    def compute_density(
            self,
            geometry: Geometry | None = None,
            kernel_name: str | None = None,
            ) -> None:
        """Compute particle densities using SPH kernels.

        Supports both isotropic and anisotropic smoothing modes. Requires that
        ``compute_smoothing_lengths`` has been called first and a kernel has been set.

        Parameters
        ----------
        geometry
                Optional geometry specification for density computation. If ``None``, uses the globally set geometry or raises an error if not set.
        kernel_name
                Optional name of the kernel to use. If ``None``, uses the globally set kernel or raises an error if not set.

        Returns
        -------
        None
                The result is stored in the instance attribute ``density``.
        """

        # ===== Check that necessary information is available ======
        geometry_temp = self._resolve_geometry(geometry)
        kernel_temp = self._resolve_kernel(kernel_name, dim=self.dim)

        # Build kernel arguments based on method
        kwargs = {}
        if geometry_temp in ["separable", "isotropic"]:
            if self.smoothing[geometry_temp].smoLens is None or self.smoothing[geometry_temp].nn_dists is None:
                raise AttributeError(
                    f"Smoothing lengths and neighbor distances must be computed for isotropic density computation. Please call 'compute_smoothing_lengths' first."
                )
            kwargs["r_ij"] = self.smoothing[geometry_temp].nn_dists
            kwargs["h"] = self.smoothing[geometry_temp].smoLens
            nn_inds_temp = self.smoothing[geometry_temp].nn_inds

        else:
            if self.smoothing[geometry_temp].smoTens is None or self.smoothing[geometry_temp].nn_inds is None:
                raise AttributeError(
                    f"Smoothing tensors must be computed for anisotropic density computation. Please call 'compute_smoothing_tensors' first."
                )
            kwargs["r_ij"] = self.smoothing[geometry_temp].rel_coords
            kwargs["h"] = self.smoothing[geometry_temp].smoTens
            nn_inds_temp = self.smoothing[geometry_temp].nn_inds

        if self.verbose:
            print(f"[smudgy] Computing density using {geometry_temp} '{kernel_temp.name}' kernel")

        # Kernel evaluation and density computation
        w = kernel_temp.evaluate(**kwargs)
        density = np.sum(self.weights[nn_inds_temp] * w, axis=1)
        
        self.smoothing[geometry_temp].density = density
        self.smoothing[geometry_temp].kernel_name = kernel_temp.name

    #def _check_density_computed(self):
    #    if not hasattr(self, "density") or hasattr(self, "density_anisotropic"):
    #        raise AttributeError(
    #            "Particle density has not been computed yet. Please call 'compute_density' first."
    #        )


    def interpolate_fields(
        self,
        fields: npt.ArrayLike | str | List[str],
        query_positions: npt.ArrayLike,
        geometry: Geometry | None = None,
        compute_gradients: bool = False,
    ) -> npt.NDArray[np.floating]:
        """Interpolate particle fields to arbitrary query positions using SPH.

        Requires that ``compute_smoothing_lengths`` and ``compute_density`` have been
        called first. Automatically computes density if not already available.
        No method specification (isotropic / anisotropic) needed, since that only affects how densities are computed.

        Parameters
        ----------
        fields
                Array of shape ``(N, num_fields)`` or ``(N,)`` with particle field data.
        query_positions
                Array of shape ``(M, D)`` with positions where fields are interpolated.
        compute_gradients
                If ``True``, compute field gradients at query positions instead of values.

        Returns
        -------
        numpy.ndarray
                If ``compute_gradients=False``: shape ``(M, num_fields)``.
                If ``compute_gradients=True``: shape ``(M, num_fields, D)``.

        Raises
        ------
        ValueError
                If ``kernel_name`` is not implemented.
        AssertionError
                If ``fields`` length does not match the number of particles.

        """

        # resolve fields
        fields = self._resolve_fields(fields)

        # check that geometry is set either globally or via argument
        geometry_temp = self._resolve_geometry(geometry)

        # load kernel from the smoothing class, since it has been used to compute density already and we want to ensure consistency between density computation and interpolation
        kernel_temp = get_kernel(self.smoothing[geometry_temp].kernel_name, dim=self.dim)
        nn_inds_temp = self.smoothing[geometry_temp].nn_inds
        
        # if query_positions is None, use particle positions for interpolation (i.e. return smoothed fields at particle positions)
        if query_positions is None:
            query_positions = self.positions


        if geometry_temp == "anisotropic" or compute_gradients: # this needs to be recomputed as query positions may differ from those used for smoothing length computation
            if self.periodic:
                rel_coords = coordinate_difference_with_pbc(
                    self.positions[nn_inds_temp],
                    query_positions[:, np.newaxis, :],
                    self.boxsize,
                )
            else:
                rel_coords = self.positions[nn_inds_temp] - query_positions[:, np.newaxis, :]

        nn_dists_temp = self.smoothing[geometry_temp].nn_dists
        smoLens_temp = self.smoothing[geometry_temp].smoLens
        smoTens_temp = self.smoothing[geometry_temp].smoTens

        
        # Unified weight computation and kernel evaluation
        density = self.smoothing[geometry_temp].density
        weights = self.weights[nn_inds_temp] / (density[nn_inds_temp] + 1e-8)
        fields_ = fields[nn_inds_temp]

        if geometry_temp in ["separable", "isotropic"]:
            if compute_gradients:
                kernel_kwargs = {"r_ij_vec": rel_coords, "h": smoLens_temp}
            else:
                kernel_kwargs = {"r_ij": nn_dists_temp, "h": smoLens_temp}
        else:  # anisotropic
            kernel_key = "r_ij_vec" if compute_gradients else "r_ij"
            kernel_kwargs = {kernel_key: rel_coords, "h": smoTens_temp}
        
        if self.verbose:
            grad_str = "gradients" if compute_gradients else "fields"
            print(f"[smudgy] Interpolating {grad_str} at query positions using {geometry_temp} '{kernel_temp.name}' kernel")

        if compute_gradients:
            w = kernel_temp.evaluate_gradient(**kernel_kwargs)
            return np.einsum("mkf,mkd,mk->mfd", fields_, w, weights)
        else:
            w = kernel_temp.evaluate(**kernel_kwargs)
            return np.einsum("mkf,mk,mk->mf", fields_, w, weights)


    def interpolate_gradient_fields(
        self,
        fields: npt.ArrayLike,
        query_positions: npt.ArrayLike,
        geometry: Literal["separable", "isotropic", "anisotropic"] | None = None,
    ) -> npt.NDArray[np.floating]:
        """Compute gradients of particle fields at arbitrary query positions using SPH.

        Convenience wrapper around `interpolate_fields` with `compute_gradients=True`.

        Parameters
        ----------
        fields
                Array of shape ``(N, num_fields)`` or ``(N,)`` with particle field data.
        query_positions
                Array of shape (M, D) with positions where gradients are evaluated.
        geometry
                Geometry type for interpolation (default is None, which uses the globally set geometry).

        Returns
        -------
        numpy.ndarray
            Array of shape (M, num_fields, D) with interpolated field gradients.
        """
        return self.interpolate_fields(
            fields=fields, 
            query_positions=query_positions, 
            geometry=geometry,
            compute_gradients=True
        )


    def deposit_to_grid(
        self,
        fields: npt.ArrayLike | str | List[str],
        averaged: Sequence[bool],
        gridnums: int | Sequence[int],
        geometry: Geometry | None = None,
        extent: Sequence[Sequence[float]] | None = None,
        plane_projection: str | None = None,
        return_weights: bool = False,
        kernel_name: str = None,
        integration: str = "midpoint",
        min_kernel_evaluations_per_axis: int = 128,
        use_python: bool = False,
        use_openmp: bool = True,
        omp_threads: int | None = None,
    ) -> (
        npt.NDArray[np.floating]
        | tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
    ):
        """Deposit particle fields onto a structured grid using the requested scheme.

        Supports multiple deposition methods (CIC, TSC, SPH-based) with optional
        plane projection for 3D→2D anisotropic tensor reduction.

        Parameters
        ----------
        fields
                Particle quantities with shape ``(num_particles, num_fields)``.
        averaged
                Flags indicating which fields should be averaged by the weight grid.
        gridnums
                Number of grid cells per spatial dimension (scalar or per-axis sequence).
        extent
                Optional sequence of ``[min, max]`` pairs per axis defining the region to deposit.
                If ``None``, deposits over ``[0, boxsize]`` for periodic boundaries.
        plane_projection
                Optional plane spec (``"xy"/"01"``, ``"xz"/"02"``, ``"yz"/"12"``) to project
                3D smoothing tensors to 2D before deposition. Only valid for 3D positions.
        return_weights
                Whether to also return the grid of accumulated weights.
        kernel_name
                SPH kernel name used for isotropic/anisotropic methods.
        integration
                Quadrature rule for SPH kernel integration (e.g., ``"midpoint"``).
        min_kernel_evaluations_per_axis
                Minimum number of kernel samples per axis for SPH methods (total samples = min_kernel_evaluations_per_axis ** dim).
        use_python
                Use Python backend instead of C++ backend (mainly for debugging).
        use_openmp
                Enable OpenMP parallelism for the C++ backend. Ignored when ``use_python=True``.
        omp_threads
                Optional positive integer overriding the number of OpenMP threads.
                If ``None``, uses runtime default.

        Returns
        -------
        numpy.ndarray or Tuple[numpy.ndarray, numpy.ndarray]
                If ``return_weights=False``: array of shape ``(gridnums..., num_fields)``.
                If ``return_weights=True``: tuple ``(fields, weights)``.

        Raises
        ------
        TypeError
                If ``omp_threads`` is not an integer when provided.
        ValueError
                If ``omp_threads <= 0`` or extent/boxsize dimensions mismatch.
        AttributeError
                If smoothing lengths/tensors were not computed for SPH methods.

        """
        
        # ====== Check that necessary information is available ======
        geometry_temp = self._resolve_geometry(geometry)

        # check that either smoothing lengths or tensors have been computed if necessary
        assert self.smoothing[geometry_temp].smoLens is not None or self.smoothing[geometry_temp].smoTens is not None, \
            f"Smoothing lengths or tensors must be computed for geometry '{geometry_temp}' before deposition. Please call 'compute_smoothing' first."
        

        # ===== Validate and process input parameters ======
        if omp_threads is not None:
            if not isinstance(omp_threads, (int, np.integer)):
                raise TypeError("'omp_threads' must be an integer if provided")
            if omp_threads <= 0:
                raise ValueError("'omp_threads' must be > 0 when specified")
            omp_threads_value = int(omp_threads)
        else:
            omp_threads_value = 0

        # deposition grid dimension is coordinate dimension unless projecting to 2D plane
        deposition_dim = self.positions.shape[1] if plane_projection is None else 2

        # check that plane_projection is only specified for 3D point clouds
        if plane_projection and self.dim != 3:
            raise ValueError(
                f"Plane projection can only be specified for 3D particle positions, positions are {self.positions.shape[1]}d"
            )
        
        # resolve the fields 
        fields = self._resolve_fields(fields)

        # check and typecast the 'averaged' parameter
        averaged = list(averaged) if isinstance(averaged, (list, tuple)) else [averaged]

        # check and typecast the 'periodic' parameter
        if extent is None:  # assume periodic box and deposit over [0, boxsize]
            if self.boxsize is None:
                raise ValueError(
                    f"Either 'boxsize' must be set on the class or 'extent' must be provided"
                )
            boxsize_array = np.asarray(self.boxsize, dtype=np.float32)
            if boxsize_array.ndim == 0:
                boxsize_array = np.repeat(boxsize_array, deposition_dim)
            if boxsize_array.size != deposition_dim:
                raise ValueError(
                    f"'boxsize' must define {deposition_dim} extents, received {boxsize_array.size}"
                )
            domain_min = np.zeros(deposition_dim, dtype=np.float32)
            domain_max = boxsize_array.astype(np.float32)
            periodic_flag = bool(self.periodic)
        else:
            extent_array = np.asarray(extent, dtype=np.float32)
            if (
                extent_array.ndim != 2
                or extent_array.shape[0] != deposition_dim
                or extent_array.shape[1] != 2
            ):
                raise ValueError(
                    f"Extent must be shaped ({deposition_dim}, 2); received {extent_array.shape}"
                )
            domain_min = extent_array[:, 0]
            domain_max = extent_array[:, 1]
            if np.any(domain_max <= domain_min):
                raise ValueError(f"Each extent axis must have max > min")
            periodic_flag = False

        # check and typecast the 'gridnums' parameter
        gridnums_array = np.asarray(gridnums, dtype=np.int32)
        if gridnums_array.ndim == 0:
            gridnums_array = np.repeat(gridnums_array, deposition_dim)
        if gridnums_array.size != deposition_dim:
            raise ValueError(
                f"Expected {deposition_dim} gridnum numbers, received {gridnums_array.size}"
            )
        gridnums_array = np.ascontiguousarray(gridnums_array, dtype=np.int32)

        # shift the particle positions to be within [0, boxsize_i] for each axis i
        positions = self.positions
        domain_lengths = domain_max - domain_min
        masks = []
        for axis in range(deposition_dim):
            axis_min = domain_min[axis]
            axis_max = domain_max[axis]
            masks.append(
                (positions[:, axis] >= axis_min) & (positions[:, axis] <= axis_max)
            )
        final_mask = (
            np.logical_and.reduce(masks)
            if masks
            else np.ones(positions.shape[0], dtype=bool)
        )

        pos_temp = positions[final_mask]
        pos_temp = pos_temp - domain_min
        fields_temp = fields[final_mask]
        smoLens_temp = self.smoLens[final_mask] if self.smoothing[geometry_temp].smoLens is not None else None


        # apply final mask and check if we need to project smoothing tensors to 2D for deposition and do so if necessary
        if geometry_temp == "anisotropic":
            if deposition_dim == 2 and self.dim == 3:
                _, smoTens_eigvals_temp, smoTens_eigvecs_temp = (
                    project_smoTens_to_2d(self.smoothing[geometry_temp].smoTens[final_mask], 
                                          plane=plane_projection)
                )
            else:
                smoTens_eigvals_temp = self.smoothing[geometry_temp].smoTens_eigvals[final_mask]
                smoTens_eigvecs_temp = self.smoothing[geometry_temp].smoTens_eigvecs[final_mask]
        else:
            smoTens_eigvals_temp = None
            smoTens_eigvecs_temp = None


        if self.verbose and not use_python:
            if use_openmp:
                info_threads = (
                    "runtime default"
                    if omp_threads_value == 0
                    else str(omp_threads_value)
                )
                print(f"[smudgy] OpenMP threads: {info_threads}")
            else:
                print("[smudgy] OpenMP disabled for this deposition call")

    
        # ========== Call correct backend function and perform deposition ==========

        dim_temp = positions.shape[-1]

        # select the deposition function based on the method, dim and use_python
        if use_python and (self.method in ["isotropic", "anisotropic"]):
            raise NotImplementedError(
                "Python backend does not implement adaptive or SPH deposition methods. "
                "Set use_python=False to use the C++ backend."
            )

        # for (adaptive) cic, tsc, (gaussian), use the separable_deposition_nD function
        # gather the appropriate function from the backend module
        func = getattr(backend, f"{self.method}_{dim_temp}d")

        if self.verbose:
            info_backend = "python" if use_python else "c++"
            print(f"[smudgy] Using {info_backend} backend for deposition")
            print(f"[smudgy] Using deposition function: {func.__name__}")

        """
        if "adaptive" in self.method:
            args = (
                pos_temp,
                fields_temp,
                smoLens_temp,
                domain_lengths,
                gridnums,
                periodic_flag,
            )
        """

        if "separable" in self.method:
            args = (
                pos_temp,
                fields_temp,
                smoLens_temp,
                domain_lengths,
                gridnums_array,
                periodic_flag,
                kernel_name,
                integration,
            )

        if "isotropic" == self.method:
            args = (
                pos_temp,
                fields_temp,
                smoLens_temp,
                domain_lengths,
                gridnums_array,
                periodic_flag,
                kernel_name,
                integration,
                min_kernel_evaluations_per_axis,
            )

        if "anisotropic" == self.method:
            args = (
                pos_temp,
                fields_temp,
                smoTens_eigvecs_temp,
                smoTens_eigvals_temp,
                domain_lengths,
                gridnums_array,
                periodic_flag,
                kernel_name,
                integration,
                min_kernel_evaluations_per_axis,
            )

        else:
            args = (
                pos_temp,
                fields_temp,
                domain_lengths,
                gridnums_array,
                periodic_flag,
            )

        threads_arg = 0 if omp_threads is None else int(omp_threads)
        fields, weights = func(
            *args, use_python=use_python, use_openmp=use_openmp, omp_threads=threads_arg
        )

        # divide averaged fields by weight
        for i in range(len(averaged)):
            if averaged[i]:
                fields[..., i] /= weights + 1e-10

        if return_weights:
            return fields, weights
        else:
            return fields

        