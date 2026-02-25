"""Core SPH operations and PointCloud class for particle-based computations."""

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from . import collection as backend
from .kernels import Kernel
from .utils import (
    build_kdtree,
    compute_hsm,
    compute_hsm_tensor,
    compute_pcellsize_half,
    coordinate_difference_with_pbc,
    project_hsm_tensor_to_2d,
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
        self.pos = positions
        self.weight = weights
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
                ), f"boxsize must be a scalar or have shape ({self.dim},), got {boxsize_arr.shape}"
                self.boxsize = boxsize_arr

    def set_sph_parameters(
        self,
        kernel_name: str = "cubic_spline",
        mode: Literal["adaptive", "isotropic", "anisotropic"] = "isotropic",
        num_neighbors: int = 32,
    ) -> None:
        """Set all SPH-related properties in one call.

        Parameters
        ----------
        kernel_name
                Name of the SPH kernel to use (e.g., ``"cubic_spline"``, ``"quintic"``).
        mode
                Smoothing mode: ``"adaptive"``, ``"isotropic"``, or ``"anisotropic"``.
        num_neighbors
                Number of nearest neighbors used for smoothing length estimation.

        Returns
        -------
        None
                The selected kernel, mode, and computed smoothing lengths/tensors are stored on the instance.

        """
        self._set_kernel(kernel_name)
        self._set_mode(mode)
        self.num_neighbors = num_neighbors

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

        try:
            self.kernel = Kernel(kernel_name, dim=self.dim)
            self.kernel_name = kernel_name
        except AttributeError:
            raise ValueError(f"Kernel '{kernel_name}' is not implemented")

    def _check_kernel_set(self):
        if not hasattr(self, "kernel"):
            raise AttributeError(
                "You must first set a kernel using the 'set_kernel' method before calling this function"
            )

    def _set_mode(self, mode: Literal["adaptive", "isotropic", "anisotropic"]) -> None:
        """Set the smoothing mode for SPH calculations.

        Parameters
        ----------
        mode
                Smoothing mode: ``"adaptive"``, ``"isotropic"``, or ``"anisotropic"``.

        Returns
        -------
        None
                The selected mode is stored in the instance attribute ``mode`` for reference.

        """
        assert mode in [
            "adaptive",
            "isotropic",
            "anisotropic",
        ], f"Mode must be one of 'adaptive', 'isotropic' or 'anisotropic' but found {mode}"
        self.mode = mode

    def compute_smoothing_lengths(self, query_pos: npt.ArrayLike | None = None) -> None:
        """Compute smoothing lengths or tensors for SPH calculations.

        Parameters
        ----------
        query_pos
                Optional array of shape ``(M, D)`` with positions where smoothing is evaluated.
                If ``None``, uses the particle positions from the instance.

        Returns
        -------
        None
                Results are stored on the instance: ``hsm`` (adaptive/isotropic),
                ``h_tensor``/``h_eigvals``/``h_eigvecs`` (anisotropic), ``nn_inds``
                (all modes), and ``nn_dists`` (isotropic only).

        """
        assert hasattr(
            self, "mode"
        ), "Smoothing mode not set. Please call either 'set_sph_parameters' or 'set_mode' before computing smoothing lengths."

        if not hasattr(self, "tree"):
            if self.verbose:
                print("Building kd-tree from particle positions")
            self.tree = build_kdtree(self.pos, boxsize=self.boxsize)

        if self.verbose:
            print(
                f"Computing smoothing lengths/tensors using mode='{self.mode}' with num_neighbors={self.num_neighbors}"
            )

        # construct kwargs
        kwargs = {
            "tree": self.tree,
            "num_neighbors": self.num_neighbors,
            "query_pos": query_pos,
        }

        if self.mode == "adaptive":
            self.hsm, self.nn_inds = compute_pcellsize_half(**kwargs)

        if self.mode == "isotropic":
            self.hsm, self.nn_inds, self.nn_dists = compute_hsm(**kwargs)

        elif self.mode == "anisotropic":
            (
                self.h_tensor,
                self.h_eigvals,
                self.h_eigvecs,
                self.nn_inds,
                self.nn_dists,
                self.rel_coords,
            ) = compute_hsm_tensor(**kwargs, weights=self.weight)

        else:
            raise AssertionError(
                f"'mode' must be either, 'adaptive', 'isotropic' or 'anisotropic' but found {self.mode}"
            )

        # Safeguard: check for invalid neighbor indices
        if hasattr(self, "nn_inds"):
            max_idx = np.max(self.nn_inds)
            if max_idx >= self.pos.shape[0]:
                raise IndexError(
                    f"Neighbor index {max_idx} is out of bounds for {self.pos.shape[0]} particles. This indicates a bug in the neighbor search or input setup."
                )

    def _check_smoothing_computed(self):
        if self.mode in ["adaptive", "isotropic"] and not hasattr(self, "hsm"):
            raise AttributeError(
                f"Smoothing lengths have not been computed for mode '{self.mode}'. Please call 'compute_smoothing_lengths' first."
            )
        if self.mode == "anisotropic" and (
            not hasattr(self, "h_tensor")
            or not hasattr(self, "h_eigvals")
            or not hasattr(self, "h_eigvecs")
        ):
            raise AttributeError(
                "Smoothing tensors have not been computed for anisotropic mode. Please call 'compute_smoothing_lengths' first."
            )

    def compute_density(self) -> None:
        """Compute particle densities using SPH kernels.

        Supports both isotropic and anisotropic smoothing modes. Requires that
        ``compute_smoothing_lengths`` has been called first and a kernel has been set.

        Parameters
        ----------
        None
                This method does not take any arguments.

        Returns
        -------
        None
                The result is stored in the instance attribute ``density``.

        """
        assert self.mode in [
            "isotropic",
            "anisotropic",
        ], "`mode` must be one of ['isotropic', 'anisotropic'] to compute density"
        self._check_kernel_set()
        self._check_smoothing_computed()

        # Build kernel arguments based on mode
        kwargs = {}
        if self.mode == "isotropic":
            kwargs["r_ij"] = self.nn_dists
            kwargs["h"] = self.hsm
        elif self.mode == "anisotropic":
            kwargs["r_ij"] = self.rel_coords
            kwargs["H"] = self.h_tensor

        # Kernel evaluation and density computation
        w = self.kernel.evaluate_kernel(**kwargs)
        print(1, w.shape, self.weight.shape)
        self.density = np.sum(self.weight[self.nn_inds] * w, axis=1)

    def interpolate_fields(
        self,
        fields: npt.ArrayLike,
        query_positions: npt.ArrayLike,
        compute_gradients: bool = False,
    ) -> npt.NDArray[np.floating]:
        """Interpolate particle fields to arbitrary query positions using SPH.

        Requires that ``compute_smoothing_lengths`` and ``compute_density`` have been
        called first. Automatically computes density if not already available.

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
        if not isinstance(fields, np.ndarray):
            fields = np.asarray(fields)
        if fields.ndim == 1:
            fields = fields[:, np.newaxis]  # make it (N, 1)
        assert (
            fields.shape[0] == self.pos.shape[0]
        ), f"'fields' array must have the same length as positions array but found position length = {self.pos.shape[0]} and fields length={fields.shape[0]}"

        if not hasattr(self, "density"):
            if self.verbose:
                print("Particle density has not been computed yet, computing now")
            self.compute_density()

        return self._interpolate_fields(
            self.tree,
            self.pos,
            self.weight,
            self.density,
            fields,
            self.kernel,
            num_neighbors=self.num_neighbors,
            query_positions=query_positions,
            mode=self.mode,
            compute_gradients=compute_gradients,
        )

    def interpolate_grad_fields(
        self,
        query_positions: npt.ArrayLike,
    ) -> npt.NDArray[np.floating]:
        """Compute gradients of particle fields at arbitrary query positions using SPH.

        Convenience wrapper around `interpolate_fields` with `compute_gradients=True`.

        Args:
                query_positions: Array of shape (M, D) with positions where gradients are evaluated.

        Returns:
                Array of shape (M, num_fields, D) with interpolated field gradients.

        """
        return self.interpolate_fields(query_positions, compute_gradients=True)

    def _interpolate_fields(
        self,
        tree: Any,
        positions: npt.NDArray[np.floating],
        weights: npt.NDArray[np.floating],
        density: npt.NDArray[np.floating],
        fields: npt.NDArray[np.floating],
        kernel: Any,
        num_neighbors: int,
        query_positions: npt.ArrayLike,
        mode: Literal["isotropic", "anisotropic"],
        compute_gradients: bool = False,
    ) -> npt.NDArray[np.floating]:
        """Interpolate or compute gradients of particle fields at query positions.

        Internal helper for SPH field interpolation supporting both isotropic and
        anisotropic smoothing kernels.

        Args:
                tree: Spatial index structure (cKDTree) for nearest neighbor queries.
                positions: Particle positions with shape (N, D).
                weights: Particle weights (e.g. masses) with shape (N,).
                density: Particle densities with shape (N,). Used for SPH weighting.
                fields: Particle field values with shape (N, num_fields).
                kernel: Kernel instance with evaluate_kernel() and evaluate_gradient() methods.
                num_neighbors: Number of nearest neighbors to use for interpolation.
                query_positions: Positions where fields are evaluated, shape (M, D).
                mode: Smoothing mode - 'isotropic' uses scalar smoothing lengths,
                        'anisotropic' uses smoothing tensors.
                compute_gradients: If True, compute field gradients via kernel.evaluate_gradient().
                        If False, compute field values via kernel.evaluate_kernel().

        Returns:
                - If compute_gradients=False: array of shape (M, num_fields) with interpolated field values.
                - If compute_gradients=True: array of shape (M, num_fields, D) with interpolated gradients.

        """
        if mode == "isotropic":
            # Compute smoothing lengths at query positions
            hsm, nn_inds, nn_dists = compute_hsm(
                tree,
                num_neighbors=num_neighbors,
                query_pos=query_positions,
            )
            if compute_gradients:
                # For gradients, need relative coordinate vectors
                rel_coords = coordinate_difference_with_pbc(
                    positions[nn_inds],
                    query_positions[:, np.newaxis, :],
                    self.boxsize,
                )
                kernel_kwargs = {"r_ij_vec": rel_coords, "h": hsm}
            else:
                kernel_kwargs = {"r_ij": nn_dists, "h": hsm}

        elif mode == "anisotropic":
            # Compute smoothing tensors at query positions
            h_tensor, _, _, nn_inds, _, rel_coords = compute_hsm_tensor(
                tree,
                weights=weights,
                num_neighbors=num_neighbors,
                query_pos=query_positions,
            )
            kernel_key = "r_ij_vec" if compute_gradients else "r_ij"
            kernel_kwargs = {kernel_key: rel_coords, "H": h_tensor}

        else:
            raise ValueError(f"Unsupported interpolation mode '{mode}'")

        # Unified weight computation and kernel evaluation
        weights = weights[nn_inds] / (density[nn_inds] + 1e-8)
        fields_ = fields[nn_inds]

        if compute_gradients:
            w = kernel.evaluate_gradient(**kernel_kwargs)
            result = np.einsum("mkf,mkd,mk->mfd", fields_, w, weights)
        else:
            w = kernel.evaluate_kernel(**kernel_kwargs)
            result = np.einsum("mkf,mk,mk->mf", fields_, w, weights)

        return result

    def deposit_to_grid(
        self,
        fields: npt.ArrayLike,
        averaged: Sequence[bool],
        gridnums: int | Sequence[int],
        method: str,
        extent: Sequence[Sequence[float]] | None = None,
        plane_projection: str | None = None,
        return_weights: bool = False,
        kernel: str = "quintic",
        integration: str = "midpoint",
        min_kernel_evaluations: int = 128,
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
        kernel
                SPH kernel name used for isotropic/anisotropic methods.
        integration
                Quadrature rule for SPH kernel integration (e.g., ``"midpoint"``).
        min_kernel_evaluations
                Minimum number of kernel samples for SPH methods.
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

        # deposition grid dimension is coordinate dimension unless projecting to 2D plane
        deposition_dim = self.pos.shape[1] if plane_projection is None else 2

        # check that plane_projection is only specified for 3D point clouds
        if plane_projection and self.dim != 3:
            raise ValueError(
                f"Plane projection can only be specified for 3D particle positions, but found positions with shape {self.pos.shape}"
            )

        # check and typecast the 'fields' parameter
        fields = np.asarray(fields, dtype=np.float32)
        if fields.ndim == 1:
            fields = fields[:, np.newaxis]  # convert to (N, 1) if given as (N,)
        if fields.shape[0] != self.pos.shape[0]:
            raise ValueError(
                f"'fields' array length ({fields.shape[0]}) must match number of particles ({self.pos.shape[0]})"
            )

        # check that either smoothing lengths or tensors have been computed if necessary
        if (
            method in ["isotropic", "anisotropic"]
            and not hasattr(self, "hsm")
            and not hasattr(self, "h_tensor")
        ):
            raise AttributeError(
                "You must first compute smoothing lengths or tensors using the 'compute_smoothing_lengths' method before depositing to grid"
            )

        # check and typecast the 'averaged' parameter
        averaged = list(averaged) if isinstance(averaged, (list, tuple)) else [averaged]

        # check and typecast the 'periodic' parameter
        if extent is None:  # assume periodic box and deposit over [0, boxsize]
            if self.boxsize is None:
                raise ValueError(
                    "Either 'boxsize' must be set on the class or 'extent' must be provided"
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
                raise ValueError("Each extent axis must have max > min")
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
        positions = self.pos
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

        res = self._deposit_to_grid(
            positions=pos_temp,
            quantities=fields_temp,
            averaged=averaged,
            gridnums=gridnums_array,
            boxsizes=domain_lengths,
            periodic=periodic_flag,
            hsm=hsm_temp,
            hmat_eigvecs=h_eigvecs_temp,
            hmat_eigvals=h_eigvals_temp,
            return_weights=return_weights,
            method=method,
            use_python=use_python,
            kernel=kernel,
            integration=integration,
            min_kernel_evaluations=min_kernel_evaluations,
            use_openmp=use_openmp,
            omp_threads=omp_threads_value,
        )
        return res

    def _deposit_to_grid(
        self,
        positions: npt.NDArray[np.floating],
        quantities: npt.NDArray[np.floating],
        averaged: Sequence[bool],
        gridnums: npt.NDArray[np.int32],
        boxsizes: npt.NDArray[np.floating],
        periodic: bool,
        *,
        method: str,
        hsm: npt.NDArray[np.floating] | None,
        hmat_eigvecs: npt.NDArray[np.floating] | None,
        hmat_eigvals: npt.NDArray[np.floating] | None,
        return_weights: bool,
        use_python: bool,
        kernel: str,
        integration: str,
        min_kernel_evaluations: int,
        use_openmp: bool,
        omp_threads: int | None,
    ) -> (
        npt.NDArray[np.floating]
        | tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
    ):
        """Dispatch particle-to-grid deposition to the selected backend.

        Internal helper that routes deposition to Python or C++ backend based on
        method and user preferences.

        Args:
                positions: Particle coordinates with shape (N, dim), shifted to domain origin.
                quantities: Fields to deposit with shape (N, num_fields).
                averaged: Flags indicating which fields should be normalized by weights.
                gridnums: Number of grid cells per dimension, shape (dim,).
                boxsizes: Domain extents (max - min) for each axis, shape (dim,).
                periodic: Per-axis periodicity flags, shape (dim,).
                method: Name of the deposition method (e.g., "ngp", "cic", "tsc", "isotropic").
                hsm: Smoothing lengths for isotropic/adaptive methods, shape (N,) or None.
                hmat_eigvecs: Smoothing tensor eigenvectors for anisotropic method, shape (N, D, D) or None.
                hmat_eigvals: Smoothing tensor eigenvalues for anisotropic method, shape (N, D) or None.
                return_weights: Whether to return accumulated weights alongside deposited fields.
                use_python: Use Python backend instead of compiled C++ backend.
                kernel: SPH kernel name for SPH-based methods.
                integration: Quadrature rule name for kernel integration.
                min_kernel_evaluations: Minimum number of kernel samples for SPH methods.
                use_openmp: Enable OpenMP parallelism in C++ backend.
                omp_threads: Number of OpenMP threads (0 = runtime default).

        Returns:
                - If return_weights=False: array of shape (gridnums..., num_fields) with deposited fields.
                - If return_weights=True: tuple of (deposited_fields, weight_grid) arrays.

        Raises:
                AttributeError: If requested deposition function not found in backend module.

        """
        dim = positions.shape[-1]

        # select the deposition function based on the method, dim and use_python
        if use_python and (
            "adaptive" in method or method in ["isotropic", "anisotropic"]
        ):
            raise NotImplementedError(
                "Python backend does not implement adaptive or SPH deposition methods. "
                "Set use_python=False to use the C++ backend."
            )
        func = getattr(backend, f"{method}_{dim}d")

        if self.verbose:
            print(f"Using deposition function: {func.__name__}")

        if "adaptive" in method:
            args = (
                positions,
                quantities,
                boxsizes,
                gridnums,
                periodic,
                hsm,
            )

        elif "isotropic" == method:
            args = (
                positions,
                quantities,
                boxsizes,
                gridnums,
                periodic,
                hsm,
                kernel,
                integration,
                min_kernel_evaluations,
            )

        elif "anisotropic" == method:
            args = (
                positions,
                quantities,
                boxsizes,
                gridnums,
                periodic,
                hmat_eigvecs,
                hmat_eigvals,
                kernel,
                integration,
                min_kernel_evaluations,
            )

        else:
            args = (
                positions,
                quantities,
                boxsizes,
                gridnums,
                periodic,
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
