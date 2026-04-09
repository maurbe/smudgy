"""Core SPH operations and PointCloud class for particle-based computations."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from .core.kernels import get_kernel
from .utils import (
    build_kdtree,
    compute_hsm,
    compute_hsm_tensor,
    coordinate_difference_with_pbc,
    project_hsm_tensor_to_2d,
    _interpolate_fields,
    _deposit_to_grid,
)


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

        if self.verbose:
            periodic_str = (
                f" with boxsize={self.boxsize}"
                if self.periodic
                else " without periodicity"
            )
            print(
                f"[smudgy] Initialized PointCloud with {self.positions.shape[0]} particles in {self.dim}D{periodic_str}"
            )

    def setup(
        self,
        num_neighbors: int = 32,
        method: Literal["isotropic", "anisotropic"] = "isotropic",
        kernel_name: str = None,
    ) -> None:
        """Set all SPH-related properties in one call.

        Parameters
        ----------
        method
                Smoothing method: ``"isotropic"``, or ``"anisotropic"``.
        num_neighbors
                Number of nearest neighbors used for smoothing length estimation.
        kernel_name
                Name of the SPH kernel. Available options are:
                ``"lucy"``, ``"gaussian"``, ``"cubic_spline"``, ``"quintic_spline"``,
                ``"wendland_c2"``, ``"wendland_c4"``, or ``"wendland_c6"``.

        Returns
        -------
        None
                The selected kernel, method, and computed smoothing lengths/tensors are stored on the instance.

        """
        self.num_neighbors = num_neighbors
        self._set_method(method)
        if kernel_name is not None:
            self._set_kernel(kernel_name)

    def _set_kernel(self, kernel_name: str) -> None:
        """Set the SPH kernel.

        Parameters
        ----------
        kernel_name
                Name of the SPH kernel to use (e.g., ``"cubic_spline"``, ``"quintic"``).

        Returns
        -------
        None
                The selected kernel is stored in the instance attribute ``kernel``.
                The selected kernel name is stored in ``kernel_name`` for reference.

        """
        assert isinstance(
            kernel_name, str
        ), f"Kernel name must be a string but found {type(kernel_name)}"
        self.kernel = get_kernel(kernel_name, dim=self.dim)

    def _check_kernel_set(self):
        if not hasattr(self, "kernel"):
            raise AttributeError(
                "No kernel set: use the 'set_sph_parameters' method with a valid 'kernel_name'"
            )

    def _set_method(self, method: Literal["isotropic", "anisotropic"]) -> None:
        """Set the smoothing method for SPH calculations.

        Parameters
        ----------
        method
                Smoothing method: ``"isotropic"``, or ``"anisotropic"``.

        Returns
        -------
        None
                The selected method is stored in the instance attribute ``method`` for reference.

        """
        assert method in [
            "isotropic",
            "anisotropic",
        ], f"Method must be one of 'isotropic' or 'anisotropic' but found {method}"
        self.method = method

    def _check_method_set(self):
        if not hasattr(self, "method"):
            raise AttributeError(
                "'method' not set. Please call 'set_sph_parameters' with a valid 'method'."
            )

    def _check_smoothing_computed(self):
        if self.method == "isotropic" and not hasattr(self, "hsm"):
            raise AttributeError(
                f"Smoothing lengths have not been computed for method '{self.method}'. Please call 'compute_smoothing_lengths' first."
            )
        if self.method == "anisotropic" and (
            not hasattr(self, "h_tensor")
            or not hasattr(self, "h_eigvals")
            or not hasattr(self, "h_eigvecs")
        ):
            raise AttributeError(
                f"Smoothing tensors have not been computed for anisotropic method '{self.method}'. Please call 'compute_smoothing_lengths' first."
            )

    def compute_smoothing_lengths(
        self, query_positions: npt.ArrayLike | None = None
    ) -> None:
        """Compute smoothing lengths or tensors for SPH calculations.

        Parameters
        ----------
        query_positions
            Optional array of shape ``(M, D)`` with positions where smoothing is evaluated.
            If ``None``, uses the particle positions from the instance.

        Returns
        -------
        None
            Results are stored on the instance: ``hsm`` (isotropic),
            ``h_tensor``/``h_eigvals``/``h_eigvecs`` (anisotropic), ``nn_inds``
            (all modes), and ``nn_dists`` (isotropic only).

        """
        assert hasattr(
            self, "method"
        ), "Smoothing method not set. Please call 'set_sph_parameters' before computing smoothing lengths."

        if not hasattr(self, "tree"):
            if self.verbose:
                print("[smudgy] Building kd-tree from particle positions")
            self.tree = build_kdtree(self.positions, boxsize=self.boxsize)

        if self.verbose:
            method_str = f"lengths" if self.method == "isotropic" else "tensors"
            print(
                f"[smudgy] Computing smoothing {method_str} from {self.num_neighbors} neighbors"
            )

        # construct kwargs
        kwargs = {
            "tree": self.tree,
            "num_neighbors": self.num_neighbors,
            "query_positions": query_positions,
        }

        if self.method == "isotropic":
            self.hsm, self.nn_inds, self.nn_dists = compute_hsm(**kwargs)

        elif self.method == "anisotropic":
            (
                self.h_tensor,
                self.h_eigvals,
                self.h_eigvecs,
                self.nn_inds,
                self.nn_dists,
                self.rel_coords,
            ) = compute_hsm_tensor(**kwargs, weights=self.weights)

        else:
            raise AssertionError(
                f"'method' must be either, 'isotropic' or 'anisotropic' but found {self.method}"
            )

        # Safeguard: check for invalid neighbor indices
        if hasattr(self, "nn_inds"):
            max_idx = np.max(self.nn_inds)
            if max_idx >= self.positions.shape[0]:
                raise IndexError(
                    f"Neighbor index {max_idx} is out of bounds for {self.positions.shape[0]} particles. This indicates a bug in the neighbor search or input setup."
                )

    def compute_density(self, kernel_name: str = None) -> None:
        """Compute particle densities using SPH kernels.

        Supports both isotropic and anisotropic smoothing modes. Requires that
        ``compute_smoothing_lengths`` has been called first and a kernel has been set.

        Parameters
        ----------
        kernel_name
                Name of the SPH kernel to use.

        Returns
        -------
        None
                The result is stored in the instance attribute ``density``.
        """
        self._check_method_set()

        # if kernel_name is provided, set the kernel; otherwise check that it has already been set
        if kernel_name is not None:
            self._set_kernel(kernel_name)
        else:
            self._check_kernel_set()

        # check that smoothing lengths/tensors have been computed
        self._check_smoothing_computed()

        # Build kernel arguments based on method
        kwargs = {}
        if self.method == "isotropic":
            kwargs["r_ij"] = self.nn_dists
            kwargs["h"] = self.hsm

        elif self.method == "anisotropic":
            kwargs["r_ij"] = self.rel_coords
            kwargs["h"] = self.h_tensor

        # Kernel evaluation and density computation
        w = self.kernel.evaluate(**kwargs)
        self.density = np.sum(self.weights[self.nn_inds] * w, axis=1)

    def interpolate_fields(
        self,
        fields: npt.ArrayLike,
        query_positions: npt.ArrayLike,
        kernel_name: str = None,
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
        kernel_name
                Name of the SPH kernel to use for interpolation (e.g., ``"cubic_spline"``, ``"quintic_spline"``).
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
        if not isinstance(fields, np.ndarray):
            fields = np.asarray(fields)
        if fields.ndim == 1:
            fields = fields[:, np.newaxis]  # make it (N, 1)
        assert (
            fields.shape[0] == self.positions.shape[0]
        ), f"'fields' array must have the same length as positions array but found position length = {self.positions.shape[0]} and fields length={fields.shape[0]}"

        if not hasattr(self, "density"):
            if self.verbose:
                print(
                    "[smudgy] Particle density has not been computed yet, computing now"
                )
            self.compute_density(kernel_name=kernel_name)

        return _interpolate_fields(
            tree=self.tree,
            positions=self.positions,
            weights=self.weights,
            density=self.density,
            fields=fields,
            kernel_name=kernel_name,
            num_neighbors=self.num_neighbors,
            query_positions=query_positions,
            boxsize=self.boxsize,
            method=self.method,
            compute_gradients=compute_gradients,
        )

    def interpolate_grad_fields(
        self,
        query_positions: npt.ArrayLike,
        kernel_name: str = None,
    ) -> npt.NDArray[np.floating]:
        """Compute gradients of particle fields at arbitrary query positions using SPH.

        Convenience wrapper around `interpolate_fields` with `compute_gradients=True`.

        Parameters
        ----------
        query_positions
            Array of shape (M, D) with positions where gradients are evaluated.

        kernel_name
            Name of the SPH kernel to use for interpolation.

        Returns
        -------
        numpy.ndarray
            Array of shape (M, num_fields, D) with interpolated field gradients.
        """
        return self.interpolate_fields(
            query_positions, kernel_name=kernel_name, compute_gradients=True
        )

    def deposit_to_grid(
        self,
        fields: npt.ArrayLike,
        averaged: Sequence[bool],
        gridnums: int | Sequence[int],
        method: str | None = None,
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
        method
                Deposition method name (e.g., ``"cic"``, ``"tsc"``, ``"isotropic"``).
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
        if omp_threads is not None:
            if not isinstance(omp_threads, (int, np.integer)):
                raise TypeError("'omp_threads' must be an integer if provided")
            if omp_threads <= 0:
                raise ValueError("'omp_threads' must be > 0 when specified")
            omp_threads_value = int(omp_threads)
        else:
            omp_threads_value = 0

        if method is None:
            self._check_method_set()

        # deposition grid dimension is coordinate dimension unless projecting to 2D plane
        deposition_dim = self.positions.shape[1] if plane_projection is None else 2

        # check that plane_projection is only specified for 3D point clouds
        if plane_projection and self.dim != 3:
            raise ValueError(
                f"Plane projection can only be specified for 3D particle positions, but found positions with shape {self.positions.shape}"
            )

        # check and typecast the 'fields' parameter
        fields = np.asarray(fields, dtype=np.float32)
        if fields.ndim == 1:
            fields = fields[:, np.newaxis]  # convert to (N, 1) if given as (N,)
        if fields.shape[0] != self.positions.shape[0]:
            raise ValueError(
                f"'fields' array length ({fields.shape[0]}) must match number of particles ({self.positions.shape[0]})"
            )

        # check that either smoothing lengths or tensors have been computed if necessary
        if (
            method in ["isotropic", "anisotropic"]
            and not hasattr(self, "hsm")
            and not hasattr(self, "h_tensor")
        ):
            raise AttributeError(
                f"No smoothing lengths or tensors computed for method {method}, use the 'compute_smoothing_lengths' method before depositing to grid"
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
                    f"Boxsize must define {deposition_dim} extents, received {boxsize_array.size}"
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
        hsm_temp = self.hsm[final_mask] if hasattr(self, "hsm") else None

        h_eigvals_temp = None
        h_eigvecs_temp = None
        if hasattr(self, "h_tensor"):
            if deposition_dim == 2 and self.dim == 3:
                self.h_tensor_2d, self.h_eigvals_2d, self.h_eigvecs_2d = (
                    project_hsm_tensor_to_2d(self.h_tensor, plane=plane_projection)
                )
                eigvals_source = self.h_eigvals_2d
                eigvecs_source = self.h_eigvecs_2d
            else:
                eigvals_source = self.h_eigvals
                eigvecs_source = self.h_eigvecs
            h_eigvals_temp = eigvals_source[final_mask]
            h_eigvecs_temp = eigvecs_source[final_mask]

        if self.verbose and not use_python:
            if use_openmp:
                thread_desc = (
                    "runtime default"
                    if omp_threads_value == 0
                    else str(omp_threads_value)
                )
                print(f"[smudgy] OpenMP threads: {thread_desc}")
            else:
                print("[smudgy] OpenMP disabled for this deposition call")

        res = _deposit_to_grid(
            positions=pos_temp,
            quantities=fields_temp,
            smoothing_lengths=hsm_temp,
            smoothing_tensor_eigvecs=h_eigvecs_temp,
            smoothing_tensor_eigvals=h_eigvals_temp,
            averaged=averaged,
            gridnums=gridnums_array,
            boxsizes=domain_lengths,
            periodic=periodic_flag,
            return_weights=return_weights,
            method=method,
            use_python=use_python,
            kernel_name=kernel_name,
            integration=integration,
            min_kernel_evaluations_per_axis=min_kernel_evaluations_per_axis,
            use_openmp=use_openmp,
            omp_threads=omp_threads_value,
            verbose=self.verbose,
        )
        return res
