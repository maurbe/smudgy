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

STRUCTURES = ("separable", "isotropic", "anisotropic")
Structure = Literal["separable", "isotropic", "anisotropic"]


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
        self.supported_structures = STRUCTURES
        self.smoothing = SmoothingInfo()

        if self.verbose:
            periodic_str = (
                f"in periodic box of size={self.boxsize}"
                if self.periodic
                else "without periodicity"
            )
            print(
                f"[smudgy] Initialized {self.dim}d PointCloud with {self.positions.shape[0]} particles {periodic_str}"
            )

    def _set_structure(self, structure: Structure,) -> None:
        assert isinstance(structure, str), f"'structure' must be a string but found {type(structure)}"
        if structure not in self.supported_structures:
            raise ValueError(f"'structure' must be one of {self.supported_structures}, but found '{structure}'")
        self.structure = structure

    def _set_kernel_name(self, kernel_name: str,) -> None:
        assert isinstance(kernel_name, str), f"'kernel_name' must be a string but found {type(kernel_name)}"
        self.kernel_name = kernel_name

    def _set_num_neighbors(self, num_neighbors: int) -> None:
        self.num_neighbors = num_neighbors

    def _check_property(self, property: str | None = None):
        """Return the requested property, checking argument and global state."""
        if not hasattr(self, property):
            raise AttributeError(
                f"'{property}' has not been set: either set it via 'global_setup' method or provide it as a function argument"
            )
        
    def _validate_neighbors(self, nn_inds):
        max_idx = np.max(nn_inds)
        if max_idx >= self.positions.shape[0]:
            raise IndexError(
                f"Neighbor index {max_idx} is out of bounds for {self.positions.shape[0]} particles. This indicates a bug in the neighbor search or input setup."
            )

    def _check_density_computed(self, structure: Structure):
        field = self.smoothing.density_iso if structure == "isotropic" else self.smoothing.density_aniso
        if field is None:
             raise AttributeError(
                f"Particle density has not been computed yet for structure '{structure}'; call 'compute_density' with 'structure={structure}' first."
            )
    
    def _check_smoothing_computed(self, structure: Structure):
        attr_map = {
            "separable": "smoLens",
            "isotropic": "smoLens",
            "anisotropic": "smoTens",
        }
        if getattr(self.smoothing, attr_map[structure]) is None:
            raise AttributeError(
                f"Smoothing {attr_map[structure]} has not been computed yet for structure '{structure}'; call 'compute_smoothing' with 'structure={structure}' first."
            )

    def _resolve_structure(self, structure: Structure | None):
        """Return the resolved structure string, checking argument and global state."""
        if structure is None:
            self._check_property('structure')
            return self.structure
        if structure not in self.supported_structures:
            raise ValueError(
                f"'structure' must be one of {self.supported_structures}, but found '{structure}'"
            )
        return structure

    def _resolve_kernel_name(self, kernel_name: str | None = None):
        """Return the resolved kernel object, checking argument and global state."""
        if kernel_name is None:
            self._check_property('kernel_name')
            return self.kernel_name
        assert isinstance(kernel_name, str), f"'kernel_name' must be a string but found {type(kernel_name)}"
        return kernel_name

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
                Can be a single string (field name) or array, a list of strings or arrays, or even a mixture of both. 
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
            self._validate_shape(arr, f"field {i}")

        # Concatenate along last axis
        return np.concatenate(arrays, axis=-1)

    def _validate_shape(self, arr, name):
            assert (
                arr.shape[0] == self.positions.shape[0]
            ), f"Length of '{name}' ({arr.shape[0]}) must match number of particles ({self.positions.shape[0]})"


    def global_setup(
            self,
            kernel_name: str = None,
            num_neighbors: int = None,
            structure: Structure = None,
            ) -> None:
        """Convenience method to set all global parameters for SPH computations.
        
        Parameters        
        ----------
        kernel_name
                Name of the SPH kernel.
        num_neighbors
                Number of neighbors to consider for smoothing length computation.
        structure
                Optional structure specification for smoothing and density computation. If ``None``, uses the globally set structure or raises an error if not set.

        Returns
        -------
        None
                An instance of the specified kernel class.
        """
        if kernel_name:
            self._set_kernel_name(kernel_name)
        if num_neighbors:
            self._set_num_neighbors(num_neighbors)
        if structure:
            self._set_structure(structure)
        return self


    def build_tree(
            self,
            positions: np.ndarray,
            boxsize: float = None
            ):
        tree = build_kdtree(positions, boxsize=boxsize)
        self.smoothing.tree = tree
    

    def compute_smoothing(
        self, 
        query_positions: npt.ArrayLike | None = None,
        num_neighbors: int | None = None,
        structure: Structure | None = None,
    ) -> None:
        """Compute smoothing lengths for SPH calculations.

        Parameters
        ----------
        query_positions
            Optional array of shape ``(M, D)`` with positions where smoothing is evaluated.
            If ``None``, uses the particle positions from the instance.
        num_neighbors
            Optional number of neighbors to consider for smoothing length computation. 
            If ``None``, uses the value globally set value or raises an error if not set.
        structure
            Optional structure specification for smoothing length computation. If ``None``, uses the globally set structure.

        Returns
        -------
        None
            Results are stored on the instance: ``smoLens`` (isotropic),
            ``smoTens``/``smoTens_eigvals``/``smoTens_eigvecs`` (anisotropic), ``nn_inds``
            (all modes), and ``nn_dists`` (isotropic only).

        """
        num_neighbors_temp = self._resolve_num_neighbors(num_neighbors)
        structure_temp = self._resolve_structure(structure)

        if self.smoothing.tree is None:
            if self.verbose:
                print("[smudgy] Building kd-tree from particle positions")
            self.build_tree(self.positions, boxsize=self.boxsize)
            tree = self.smoothing.tree

        if self.verbose:
            info_str = "tensors" if structure_temp == "anisotropic" else "lengths"
            print(f"[smudgy] Computing smoothing {info_str} from {num_neighbors_temp} neighbors")

        # construct kwargs
        kwargs = {
            "tree": tree,
            "num_neighbors": num_neighbors_temp,
            "query_positions": query_positions,
        }

        if structure_temp in ["separable", "isotropic"]:
            (
                smoLens, 
                nn_inds, 
                nn_dists 
            ) = compute_smoLens(**kwargs)

            self.smoothing.smoLens = smoLens
        
        elif structure_temp == "anisotropic":
            (
                smoTens,
                smoTens_eigvals,
                smoTens_eigvecs,
                nn_inds,
                nn_dists,
                nn_dists_vec
            ) = compute_smoTens(**kwargs, weights=self.weights)

            self.smoothing.smoTens = smoTens
            self.smoothing.smoTens_eigvals = smoTens_eigvals
            self.smoothing.smoTens_eigvecs = smoTens_eigvecs
            self.smoothing.nn_dists_vec = nn_dists_vec

        self.smoothing.tree = tree
        self.smoothing.nn_inds = nn_inds
        self.smoothing.nn_dists = nn_dists
        self.smoothing.num_neighbors = num_neighbors_temp

        # Safeguard: check for invalid neighbor indices
        self._validate_neighbors(self.smoothing.nn_inds)


    def set_smoothing(
        self,
        structure: Structure | None = None,
        smoLens: npt.ArrayLike | None = None,
        smoTens: npt.ArrayLike | None = None,
        smoTens_eigvals: npt.ArrayLike | None = None,
        smoTens_eigvecs: npt.ArrayLike | None = None,
    ):
        """
        Manually assign smoothing lengths or tensors to particles.

        Parameters
        ----------
        structure
            Smoothing structure. Must be provided (either 'isotropic' or 'anisotropic') if setting smoLens or smoTens.
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

        # for smoLens, structure must be set and either 'separable' or 'isotropic'
        if smoLens:
            assert structure == "isotropic", "Structure must be specified when providing 'smoLens'"
            self._assert_shape(smoLens, "smoLens")
            self.smoothing.smoLens = np.asarray(smoLens, dtype=np.float32)
        
        if smoTens:
            assert structure == "anisotropic", "Structure must be specified when providing 'smoTens'"
            self._assert_shape(smoTens, "smoTens")
            self.smoothing.smoTens = np.asarray(smoTens, dtype=np.float32)

        if smoTens_eigvals:
            self._assert_shape(smoTens_eigvals, "smoTens_eigvals")
            self.smoothing.smoTens_eigvals = np.asarray(smoTens_eigvals, dtype=np.float32)
            
        if smoTens_eigvecs:
            self._assert_shape(smoTens_eigvecs, "smoTens_eigvecs")
            self.smoothing.smoTens_eigvecs = np.asarray(smoTens_eigvecs, dtype=np.float32)


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
            kernel_name: str | None = None,
            structure: Structure | None = None,
            ) -> None:
        """Compute particle densities using SPH kernels.

        Supports both isotropic and anisotropic smoothing modes. Requires that
        ``compute_smoothing_lengths`` has been called first and a kernel has been set.

        Parameters
        ----------
        kernel_name
                Optional name of the kernel to use. If ``None``, uses the globally set kernel or raises an error if not set.
        structure
                Optional smoothing structure. If ``None``, uses the globally set structure or raises an error if not set.

        Returns
        -------
        None
                The result is stored in the instance attribute ``density``.
        """

        # ===== Check that necessary information is available ======
        structure_temp = self._resolve_structure(structure)
        kernel_name_temp = self._resolve_kernel_name(kernel_name)
        kernel_temp = get_kernel(kernel_name_temp, dim=self.dim)

        # check that smoothing information has been computed for the chosen structure
        self._check_smoothing_computed(structure_temp)

        # check that the necessary neighbor information is available for the chosen structure
        if (self.smoothing.nn_dists is None) or (self.smoothing.nn_inds is None):
            raise AttributeError(
                f"List of nearest neighbors has not been computed yet; call 'compute_smoothing_lengths' first.\
                    If you wish to set smoothing lengths/tensors manually, first call 'compute_smoothing_lengths' to compute the kd-tree, then use 'set_smoothing' to override the smoothing lengths/tensors."
            )
        nn_inds_temp = self.smoothing.nn_inds

        # Build kernel arguments based on structure
        kwargs = {
            'r_ij': self.smoothing.nn_dists_vec if structure_temp == "anisotropic" else self.smoothing.nn_dists,
             'h': self.smoothing.smoTens if structure_temp == "anisotropic" else self.smoothing.smoLens
        }

        if self.verbose:
            print(f"[smudgy] Computing density using {structure_temp} '{kernel_temp.name}' kernel")

        # Kernel evaluation and density computation
        w = kernel_temp.evaluate(**kwargs)
        density = np.sum(self.weights[nn_inds_temp] * w, axis=1)
        
        self.smoothing.kernel_name = kernel_name_temp
        if structure_temp == "anisotropic":
            self.smoothing.density_aniso = density
        else:
            self.smoothing.density_iso = density


    def interpolate_fields(
        self,
        fields: npt.ArrayLike | str | List[str],
        query_positions: npt.ArrayLike,
        compute_gradients: bool = False,
        structure: Structure | None = None,
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
        structure
                Optional smoothing structure. If ``None``, uses the globally set structure or raises an error if not set.

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
        # check that structure is set either globally or via argument
        structure_temp = self._resolve_structure(structure)

        # check that density has been computed for the chosen structure
        self._check_density_computed(structure_temp)

        # load kernel from the smoothing class, since it has been used to compute density already and we want to ensure consistency between density computation and interpolation
        kernel_temp = get_kernel(self.smoothing.kernel_name, dim=self.dim)

        # cast fields to correct input format
        fields = self._resolve_fields(fields)

        
        # if query_positions is None, use particle positions for interpolation (i.e. return smoothed fields at particle positions)
        # for compute_gradients = False, this is redundant, since fields is already at particle positions
        # but for compute_gradients = True, this allows to compute smoothed gradients at particle positions
        if query_positions is None:
            query_positions = self.positions
        
        # prepare the relevant attributes
        nn_inds_temp    = self.smoothing.nn_inds
        nn_dists_temp   = self.smoothing.nn_dists
        smoLens_temp    = self.smoothing.smoLens
        smoTens_temp    = self.smoothing.smoTens

        if structure_temp == "anisotropic":
            density_temp = self.smoothing.density_aniso
        else:
            density_temp = self.smoothing.density_iso


        # compute interpolation weights
        weights = self.weights[nn_inds_temp] / (density_temp[nn_inds_temp] + 1e-8)
        fields_ = fields[nn_inds_temp]
        

        if structure_temp == "anisotropic" or compute_gradients: # this needs to be recomputed as query positions may differ from those used for smoothing length computation
            if self.periodic:
                rel_coords = coordinate_difference_with_pbc(
                    self.positions[nn_inds_temp],
                    query_positions[:, np.newaxis, :],
                    self.boxsize,
                )
            else:
                rel_coords = self.positions[nn_inds_temp] - query_positions[:, np.newaxis, :]


        if structure_temp in ["separable", "isotropic"]:
            if compute_gradients:
                kernel_kwargs = {"r_ij_vec": rel_coords, "h": smoLens_temp}
            else:
                kernel_kwargs = {"r_ij": nn_dists_temp, "h": smoLens_temp}
        else:  # anisotropic
            kernel_key = "r_ij_vec" if compute_gradients else "r_ij"
            kernel_kwargs = {kernel_key: rel_coords, "h": smoTens_temp}
        
        if self.verbose:
            grad_str = "gradients" if compute_gradients else "fields"
            print(f"[smudgy] Interpolating {grad_str} at query positions using {structure_temp} '{kernel_temp.name}' kernel")

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
        structure: Structure = None,
    ) -> npt.NDArray[np.floating]:
        """Compute gradients of particle fields at arbitrary query positions using SPH.

        Convenience wrapper around `interpolate_fields` with `compute_gradients=True`.

        Parameters
        ----------
        fields
                Array of shape ``(N, num_fields)`` or ``(N,)`` with particle field data.
        query_positions
                Array of shape (M, D) with positions where gradients are evaluated.
        structure
                Smoothing structure for interpolation (default is None, which uses the globally set structure).

        Returns
        -------
        numpy.ndarray
            Array of shape (M, num_fields, D) with interpolated field gradients.
        """
        return self.interpolate_fields(
            fields=fields, 
            query_positions=query_positions, 
            compute_gradients=True,
            structure=structure,
        )


    def deposit_to_grid(
        self,
        fields: npt.ArrayLike | str | List[str],
        averaged: Sequence[bool],
        gridnums: int | Sequence[int],
        extent: Sequence[Sequence[float]] | None = None,

        kernel_name: str = None,
        structure: Structure = None,
        adaptive: bool = False,

        plane_projection: str | None = None,
        integration: str = "midpoint",
        min_kernel_evaluations_per_axis: int = 5,
        return_weights: bool = False,
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
        kernel_name_temp = self._resolve_kernel_name(kernel_name)
        if kernel_name_temp != 'ngp':
            structure_temp = self._resolve_structure(structure)
        else:
            structure_temp = None

        # 1. NGP Special Case
        if kernel_name_temp == "ngp":
            method_name = "ngp"
        else:
            # 2. Compatibility Matrix & Mapping
            # The following kernels support both separable and isotropic/anisotropic modes.
            # However, spherically symmetric kernels (e.g., Lucy, Splines) do NOT support 'separable' mode.
            dual_mode_kernels = ("tophat", "tsc", "gaussian")
            is_dual_kernel = any(k in kernel_name_temp for k in dual_mode_kernels)
            is_struct_separable = structure_temp == "separable"

            # Error if structure is 'separable' but kernel is NOT a dual-mode kernel.
            # Dual-mode kernels are compatible with BOTH 'separable' and 'isotropic'/'anisotropic' structures.
            if is_struct_separable and not is_dual_kernel:
                raise ValueError(
                    f"Structure '{structure_temp}' is only compatible with dual-mode kernels {dual_mode_kernels}. "
                    f"The kernel '{kernel_name_temp}' is spherically symmetric and requires 'isotropic' or 'anisotropic' structure."
                )
            
            method_name = structure_temp

            # if one of dual kernels specified, but structure == 'separable' -> cast kernel name + "separable"
            if is_dual_kernel and is_struct_separable:
                kernel_name_temp = kernel_name_temp + "_separable"

        # resolve the fields 
        fields = self._resolve_fields(fields)

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
                f"Plane projection can only be specified for 3D particle positions, current positions are {self.positions.shape[1]}d"
            )
        
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
        final_mask = (np.logical_and.reduce(masks) if masks else np.ones(positions.shape[0], dtype=bool))

        pos_temp = positions[final_mask]
        pos_temp = pos_temp - domain_min
        fields_temp = fields[final_mask]

        # 3. Data Preparation (Fixed vs. Adaptive)
        if kernel_name_temp == "ngp":
            smoLens_temp = None
            smoTens_eigvals_temp = None
            smoTens_eigvecs_temp = None
        elif not adaptive:
            # Calculate temporary smoothing as half of the grid cell size
            # This is local and does not modify self.smoothing
            grid_spacing = domain_lengths / gridnums_array
            if structure_temp == "separable":
                # Shape (N, D) where each column is grid_spacing[d] / 2
                smoLens_temp = np.repeat((grid_spacing / 2.0)[np.newaxis, :], pos_temp.shape[0], axis=0).astype(np.float32)
            else:
                # Isotropic scale: average of grid spacing or min grid spacing? Let's use average.
                iso_scale = np.mean(grid_spacing) / 2.0
                smoLens_temp = np.full(pos_temp.shape[0], iso_scale, dtype=np.float32)
            
            smoTens_eigvals_temp = None
            smoTens_eigvecs_temp = None
        else:
            # Adaptive deposition: fetch pre-computed smoothing data
            self._check_smoothing_computed(structure_temp)
            
            if structure_temp == "separable":
                # smoLens is stored as isotropic (N,), broadcast to (N, D) for separable deposition
                h_iso = self.smoothing.smoLens[final_mask]
                smoLens_temp = np.repeat(h_iso[:, np.newaxis], deposition_dim, axis=1)
                smoTens_eigvals_temp = None
                smoTens_eigvecs_temp = None
            elif structure_temp == "isotropic":
                smoLens_temp = self.smoothing.smoLens[final_mask]
                smoTens_eigvals_temp = None
                smoTens_eigvecs_temp = None
            else:  # anisotropic
                smoLens_temp = None
                
                # Handling 3D -> 2D projection locally if necessary
                if deposition_dim == 2 and self.dim == 3:
                    if self.verbose:
                        print(f"[smudgy] Projecting 3D smoothing tensors to 2D plane '{plane_projection}' locally for deposition")
                    _, smoTens_eigvals_temp, smoTens_eigvecs_temp = project_smoTens_to_2d(
                        self.smoothing.smoTens[final_mask], 
                        plane=plane_projection
                    )
                else:
                    smoTens_eigvals_temp = self.smoothing.smoTens_eigvals[final_mask]
                    smoTens_eigvecs_temp = self.smoothing.smoTens_eigvecs[final_mask]

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
        dim_temp = deposition_dim  # using grid dimensions
        func_name = f"{method_name}_{dim_temp}d"
        func = getattr(backend, func_name)

        if self.verbose:
            info_backend = "python" if use_python else "c++"
            print(f"[smudgy] Using {info_backend} backend for {method_name} deposition")
            print(f"[smudgy] Using deposition function: {func.__name__}")


        # Construct specific arguments for each backend type
        if method_name == "ngp":
            args = (
                pos_temp,
                fields_temp,
                domain_lengths,
                gridnums_array,
                periodic_flag,
            )
        elif method_name == "separable":
            args = (
                pos_temp,
                fields_temp,
                smoLens_temp,
                domain_lengths,
                gridnums_array,
                periodic_flag,
                kernel_name_temp,
                integration,
            )
        elif method_name == "isotropic":
            args = (
                pos_temp,
                fields_temp,
                smoLens_temp,
                domain_lengths,
                gridnums_array,
                periodic_flag,
                kernel_name_temp,
                integration,
                min_kernel_evaluations_per_axis,
            )
        elif method_name == "anisotropic":
            args = (
                pos_temp,
                fields_temp,
                smoTens_eigvecs_temp,
                smoTens_eigvals_temp,
                domain_lengths,
                gridnums_array,
                periodic_flag,
                kernel_name_temp,
                integration,
                min_kernel_evaluations_per_axis,
            )
        else:
            raise ValueError(f"Unknown deposition method '{method_name}'")

        threads_arg = 0 if omp_threads is None else int(omp_threads)
        fields, weights = func(
            *args, use_python=use_python, use_openmp=use_openmp, omp_threads=threads_arg
        )

        # divide averaged fields by weight
        for i in range(len(averaged)):
            if i < fields.shape[-1] and averaged[i]:
                fields[..., i] /= (weights + 1e-10)

        if return_weights:
            return fields, weights
        else:
            return fields

        