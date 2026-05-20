"""Core SPH operations and PointCloud class for particle-based computations."""

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from .core.backend import _call_backend
from .core.kernels import get_kernel
from .smooth import SmoothingInfo
from .utils import (
    build_kdtree,
    compute_smoLens,
    compute_smoTens,
    coordinate_difference_with_pbc,
    project_smoTens_to_2d,
    query_kdtree,
)

STRUCTURES = ("separable", "isotropic", "anisotropic")
Structure = Literal["separable", "isotropic", "anisotropic"]


class PointCloud:
    """Represent a collection of particles for operations."""

    def __init__(
        self,
        positions: npt.NDArray[np.floating],
        weights: npt.NDArray[np.floating] | None = None,
        boxsize: float | Sequence[float] | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialize a PointCloud container for particle-based SPH computations.

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

        """
        self.dim = positions.shape[-1]
        assert (
            self.dim == 2 or self.dim == 3
        ), f"Particle positions must be of shape (N, 2) or (N, 3) but found {positions.shape}"
        self.positions = positions.astype(np.float32)

        self.weights = (
            np.ones(self.positions.shape[0], dtype=np.float32)
            if weights is None
            else weights.astype(np.float32)
        )
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

    def _set_structure(self, structure: Structure) -> None:
        """Set the smoothing structure (isotropic, anisotropic, or separable)."""
        assert isinstance(
            structure, str
        ), f"'structure' must be a string but found {type(structure)}"
        if structure not in self.supported_structures:
            raise ValueError(
                f"'structure' must be one of {self.supported_structures}, but found '{structure}'"
            )
        self.structure = structure

    def _set_kernel_name(self, kernel_name: str) -> None:
        """Set the name of the SPH kernel to use."""
        assert isinstance(
            kernel_name, str
        ), f"'kernel_name' must be a string but found {type(kernel_name)}"
        self.kernel_name = kernel_name

    def _set_num_neighbors(self, num_neighbors: int) -> None:
        """Set the number of neighbors for smoothing computations."""
        self.num_neighbors = num_neighbors

    def _check_property(self, property: str | None = None) -> None:
        """Ensure a global property is set before use."""
        if not hasattr(self, property):
            raise AttributeError(
                f"'{property}' has not been set: either set it via 'global_setup' method or provide it as a function argument"
            )

    def _validate_neighbors(self, nn_inds: npt.NDArray[np.integer]) -> None:
        """Verify that neighbor indices are within valid bounds."""
        max_idx = np.max(nn_inds)
        if max_idx >= self.positions.shape[0]:
            raise IndexError(
                f"Neighbor index {max_idx} is out of bounds for {self.positions.shape[0]} particles. This indicates a bug in the neighbor search or input setup."
            )

    def _check_density_computed(self, structure: Structure) -> None:
        """Verify that density has been computed for the requested structure."""
        field = (
            self.smoothing.density_iso
            if structure == "isotropic"
            else self.smoothing.density_aniso
        )
        if field is None:
            raise AttributeError(
                f"Particle density has not been computed yet for structure '{structure}'; call 'compute_density' with 'structure={structure}' first."
            )

    def _check_smoothing_computed(self, structure: Structure) -> None:
        """Verify that smoothing lengths/tensors have been computed for the structure."""
        attr_map = {
            "separable": "smoLens",
            "isotropic": "smoLens",
            "anisotropic": "smoTens",
        }
        if getattr(self.smoothing, attr_map[structure]) is None:
            raise AttributeError(
                f"Smoothing {attr_map[structure]} has not been computed yet for structure '{structure}'; call 'compute_smoothing' with 'structure={structure}' first."
            )

    def _resolve_structure(self, structure: Structure | None) -> Structure:
        """Resolve the smoothing structure from argument or global state."""
        if structure is None:
            self._check_property("structure")
            return self.structure
        if structure not in self.supported_structures:
            raise ValueError(
                f"'structure' must be one of {self.supported_structures}, but found '{structure}'"
            )
        return structure

    def _resolve_kernel_name(self, kernel_name: str | None = None) -> str:
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

    def _resolve(self, val: Any, name: str) -> Any:
        """Resolve global properties (kernel, structure, etc.)."""
        if val is not None:
            return val
        if not hasattr(self, name):
            raise AttributeError(
                f"'{name}' not set; use 'global_setup' or provide as argument"
            )
        return getattr(self, name)

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
            self._validate_shape(arr, f"field {i}")

        return np.concatenate(arrays, axis=-1), field_sizes

    def _validate_shape(self, arr: npt.NDArray[Any], name: str) -> None:
        """Ensure the first dimension of an array matches the number of particles."""
        if arr.shape[0] != self.positions.shape[0]:
            raise ValueError(
                f"Length of '{name}' ({arr.shape[0]}) must match number of particles ({self.positions.shape[0]})"
            )

    def global_setup(
        self,
        structure: Structure | None = None,
        kernel_name: str | None = None,
        num_neighbors: int | None = None,
    ) -> "PointCloud":
        """Set global parameters for SPH computations.

        Parameters
        ----------
        structure : Structure, optional
            Smoothing structure ('separable', 'isotropic', or 'anisotropic').
        kernel_name : str, optional
            Name of the SPH kernel to use globally.
        num_neighbors : int, optional
            Number of neighbors for smoothing length computation.

        Returns
        -------
        PointCloud
            The current instance for method chaining.

        """
        if structure:
            self._set_structure(structure)
            if self.verbose:
                print(f"[smudgy] Set structure globally to '{self.structure}'")
        if kernel_name:
            self._set_kernel_name(kernel_name)
            if self.verbose:
                print(f"[smudgy] Set kernel globally to '{self.kernel_name}'")
        if num_neighbors:
            self._set_num_neighbors(num_neighbors)
            if self.verbose:
                print(
                    f"[smudgy] Set number of neighbors globally to {self.num_neighbors}"
                )
        return self

    def _build_tree(
        self,
        positions: npt.NDArray[np.floating],
        boxsize: float | Sequence[float] | None = None,
    ) -> None:
        """Build a kd-tree for neighbor searches.

        Parameters
        ----------
        positions : npt.NDArray[np.floating]
            Particle positions, shape (N, D).
        boxsize : float or array-like, optional
            Periodic box size for the tree construction.

        """
        tree = build_kdtree(positions, boxsize=boxsize)
        self.smoothing.tree = tree

    def _ensure_tree(self) -> Any:
        """Ensure a kd-tree exists for neighbor searches."""
        if self.smoothing.tree is None:
            if self.verbose:
                print("[smudgy] Building kd-tree from particle positions")
            self._build_tree(self.positions, boxsize=self.boxsize)
        return self.smoothing.tree

    def compute_smoothing(
        self,
        query_positions: npt.ArrayLike | None = None,
        num_neighbors: int | None = None,
        structure: Structure | None = None,
    ) -> None:
        """Compute smoothing lengths for SPH calculations.

        Parameters
        ----------
        query_positions : npt.ArrayLike, optional
            Positions where smoothing is evaluated. If None, uses particle positions.
        num_neighbors : int, optional
            Number of neighbors for smoothing length computation.
        structure : Structure, optional
            Smoothing structure for computation.

        Returns
        -------
        None

        Notes
        -----
        Results are stored in the ``smoothing`` attribute.

        """
        num_neighbors_temp = self._resolve_num_neighbors(num_neighbors)
        structure_temp = self._resolve_structure(structure)
        tree = self._ensure_tree()

        if self.verbose:
            info_str = "tensors" if structure_temp == "anisotropic" else "lengths"
            print(
                f"[smudgy] Computing smoothing {info_str} from {num_neighbors_temp} neighbors"
            )

        kwargs = {
            "tree": tree,
            "num_neighbors": num_neighbors_temp,
            "query_positions": query_positions,
        }

        if structure_temp in ["separable", "isotropic"]:
            smoLens, nn_inds, nn_dists = compute_smoLens(**kwargs)
            self.smoothing.smoLens = smoLens
        else:  # anisotropic
            (
                smoTens,
                smoTens_eigvals,
                smoTens_eigvecs,
                nn_inds,
                nn_dists,
                nn_dists_vec,
            ) = compute_smoTens(**kwargs, weights=self.weights)
            self.smoothing.smoTens, self.smoothing.nn_dists_vec = smoTens, nn_dists_vec
            self.smoothing.smoTens_eigvals, self.smoothing.smoTens_eigvecs = (
                smoTens_eigvals,
                smoTens_eigvecs,
            )

        self.smoothing.nn_inds, self.smoothing.nn_dists = nn_inds, nn_dists
        self.smoothing.num_neighbors = num_neighbors_temp
        self._validate_neighbors(nn_inds)

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

    def _get_active_smoothing(
        self, structure: Structure, mask: npt.NDArray[np.bool_] | None = None
    ) -> npt.NDArray[np.floating]:
        """Retrieve the relevant smoothing data (lengths or tensors) for a given structure."""
        if structure in ["separable", "isotropic"]:
            data = self.smoothing.smoLens
        else:
            data = self.smoothing.smoTens

        return data[mask] if mask is not None else data

    def set_smoothing(
        self,
        structure: Structure | None = None,
        smoLens: npt.ArrayLike | None = None,
        smoTens: npt.ArrayLike | None = None,
        smoTens_eigvals: npt.ArrayLike | None = None,
        smoTens_eigvecs: npt.ArrayLike | None = None,
    ) -> None:
        """Manually assign smoothing lengths or tensors to particles.

        Parameters
        ----------
        structure : Structure, optional
            Smoothing structure. Required if setting `smoLens` or `smoTens`.
        smoLens : npt.ArrayLike, optional
            Isotropic smoothing lengths, shape (N,).
        smoTens : npt.ArrayLike, optional
            Anisotropic smoothing tensors, shape (N, D, D).
        smoTens_eigvals : npt.ArrayLike, optional
            Eigenvalues of the smoothing tensors, shape (N, D).
        smoTens_eigvecs : npt.ArrayLike, optional
            Eigenvectors of the smoothing tensors, shape (N, D, D).

        """
        # for smoLens, structure must be set and either 'separable' or 'isotropic'
        if smoLens:
            assert (
                structure == "isotropic"
            ), "Structure must be specified when providing 'smoLens'"
            self._validate_shape(np.asarray(smoLens), "smoLens")
            self.smoothing.smoLens = np.asarray(smoLens, dtype=np.float32)

        if smoTens:
            assert (
                structure == "anisotropic"
            ), "Structure must be specified when providing 'smoTens'"
            self._validate_shape(np.asarray(smoTens), "smoTens")
            self.smoothing.smoTens = np.asarray(smoTens, dtype=np.float32)

        if smoTens_eigvals:
            self._validate_shape(np.asarray(smoTens_eigvals), "smoTens_eigvals")
            self.smoothing.smoTens_eigvals = np.asarray(
                smoTens_eigvals, dtype=np.float32
            )

        if smoTens_eigvecs:
            self._validate_shape(np.asarray(smoTens_eigvecs), "smoTens_eigvecs")
            self.smoothing.smoTens_eigvecs = np.asarray(
                smoTens_eigvecs, dtype=np.float32
            )

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

        self._validate_shape(values_arr, name)
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
        st = self._resolve(structure, "structure")
        kn = self._resolve(kernel_name, "kernel_name")
        kernel = get_kernel(kn, dim=self.dim)

        self._check_smoothing_computed(st)
        nn_inds = self.smoothing.nn_inds

        kwargs = {
            "r_ij": (
                self.smoothing.nn_dists_vec
                if st == "anisotropic"
                else self.smoothing.nn_dists
            ),
            "h": (
                self.smoothing.smoTens
                if st == "anisotropic"
                else self.smoothing.smoLens
            ),
            "mode": st,
        }

        # r_ij = (N, 8)
        # h = (N,)

        if self.verbose:
            print(f"[smudgy] Computing density using {st} '{kernel.name}' kernel")

        density = np.sum(self.weights[nn_inds] * kernel.evaluate(**kwargs), axis=1)

        self.smoothing.kernel_name = kn
        if st == "anisotropic":
            self.smoothing.density_aniso = density
        else:
            self.smoothing.density_iso = density

    def interpolate_fields(
        self,
        fields: npt.ArrayLike | str | list[str],
        query_positions: npt.ArrayLike = None,
        compute_gradients: bool = False,
        structure: Structure | None = None,
    ) -> npt.NDArray[np.floating]:
        """Interpolate particle fields to query positions using SPH.

        Parameters
        ----------
        fields : Union[npt.ArrayLike, str, List[str]]
            Field data to interpolate. Can be an array, string name, or list of both.
        query_positions : npt.ArrayLike, optional
            Array of shape (M, D) with positions where fields are interpolated.
        compute_gradients : bool, default False
            Whether to compute field gradients instead of values.
        structure : Structure, optional
            Smoothing structure for interpolation.

        Returns
        -------
        npt.NDArray[np.floating]
            Interpolated field values (shape M, F) or gradients (shape M, F, D).

        """
        # check that structure is set either globally or via argument
        structure_temp = self._resolve_structure(structure)

        # check that density has been computed for the chosen structure
        self._check_density_computed(structure_temp)

        # load kernel from the smoothing class, since it has been used to compute density already and we want to ensure consistency between density computation and interpolation
        kernel_temp = get_kernel(self.smoothing.kernel_name, dim=self.dim)

        # cast fields to correct input format
        fields, fields_sizes = self._resolve_fields(fields)

        # if query_positions is None, use particle positions for interpolation (i.e. return smoothed fields at particle positions)
        # for compute_gradients = True, this allows to compute smoothed gradients at particle positions
        if query_positions is None:
            query_positions = self.positions

            # can re-use the neighbor indices and distances from the smoothing computation
            nn_inds = self.smoothing.nn_inds
            nn_dists = self.smoothing.nn_dists

        else:
            # for new query positions, need to perform a new neighbor search
            tree = self._ensure_tree()
            nn_dists, nn_inds = query_kdtree(
                tree,
                query_positions,
                k=self.num_neighbors,
            )

        # prepare interpolation weights and fields
        fields_ = fields[nn_inds]

        if structure_temp == "anisotropic":
            density_temp = self.smoothing.density_aniso
        else:
            density_temp = self.smoothing.density_iso

        weights = self.weights[nn_inds] / (density_temp[nn_inds] + 1e-8)

        # select kernel arguments
        kwargs = dict(mode=structure_temp)

        if structure_temp == "anisotropic" or compute_gradients:
            rel_coords = self._get_rel_coords(
                query_positions,
                self.positions[nn_inds],
            )

        kwargs["r_ij"] = (
            rel_coords
            if (structure_temp == "anisotropic" or compute_gradients)
            else nn_dists
        )

        kwargs["h"] = (
            self.smoothing.smoTens[nn_inds]
            if structure_temp == "anisotropic"
            else self.smoothing.smoLens[nn_inds]
        )

        if self.verbose:
            grad_str = "gradients of " if compute_gradients else ""
            print(
                f"[smudgy] Interpolating {grad_str}fields at query positions using {structure_temp} '{kernel_temp.name}' kernel"
            )

        w = (
            kernel_temp.evaluate_gradient(**kwargs)
            if compute_gradients
            else kernel_temp.evaluate(**kwargs)
        )

        einsum_str = (
            "...kf,...kd,...k->...fd" if compute_gradients else "...kf,...k,...k->...f"
        )

        return np.einsum(einsum_str, fields_, w, weights)

    def interpolate_gradient_fields(
        self,
        fields: npt.ArrayLike,
        query_positions: npt.ArrayLike,
        structure: Structure | None = None,
    ) -> npt.NDArray[np.floating]:
        """Compute gradients of particle fields at query positions using SPH.

        Parameters
        ----------
        fields : npt.ArrayLike
            Field data to differentiate.
        query_positions : npt.ArrayLike
            Positions where gradients are evaluated.
        structure : Structure, optional
            Smoothing structure for interpolation.

        Returns
        -------
        npt.NDArray[np.floating]
            Interpolated field gradients, shape (M, F, D).

        """
        return self.interpolate_fields(
            fields=fields,
            query_positions=query_positions,
            compute_gradients=True,
            structure=structure,
        )

    def _resolve_deposition_method(
        self, kernel_name: str | None, structure: Structure | None
    ) -> tuple[str, str]:
        """Map kernel and structure to a specific deposition method and backend kernel name."""
        kn = self._resolve_kernel_name(kernel_name)
        if kn == "ngp":
            return "ngp", kn

        st = self._resolve_structure(structure)
        dual_kernels = ("tophat", "tsc", "gaussian")
        is_dual = any(k in kn for k in dual_kernels)
        is_separable = st == "separable"

        if is_separable and not is_dual:
            raise ValueError(
                f"Structure 'separable' is incompatible with spherically symmetric kernel '{kn}'"
            )

        method = st
        if is_dual and is_separable:
            kn += "_separable"

        return method, kn

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

    def _prepare_deposition_smoothing(
        self,
        method: str,
        adaptive: bool,
        num_p: int,
        gn: npt.NDArray[np.int32],
        d_lens: npt.NDArray[np.float32],
        mask: npt.NDArray[np.bool_],
        dep_dim: int,
        plane_projection: list[int] | None = None,
        plane_projection_basis: npt.NDArray[np.float32] | None = None,
    ) -> tuple[
        npt.NDArray[np.float32] | None,
        npt.NDArray[np.float32] | None,
        npt.NDArray[np.float32] | None,
    ]:
        """Prepare smoothing data (fixed or adaptive) for the deposition call."""
        # First two are NGP and uniform methods like Cloud-in-Cell and TSC where no smoothing info is needed
        if method == "ngp":
            return None, None, None

        if not adaptive:
            spacing = d_lens / gn
            if method == "separable":
                return (
                    np.repeat((spacing * 1.0)[np.newaxis, :], num_p, axis=0).astype(
                        np.float32
                    ),
                    None,
                    None,
                )
            # for isotropic we average the spacing across all dimensions
            # TODO: can do better here...
            else:
                return (
                    np.full(num_p, np.mean(spacing), dtype=np.float32),
                    None,
                    None,
                )

        # Otherwise we gather smoothing info based on selected structure
        self._check_smoothing_computed(method if method != "separable" else "isotropic")

        if method == "separable":
            return (
                np.repeat(self.smoothing.smoLens[mask][:, np.newaxis], dep_dim, axis=1),
                None,
                None,
            )

        if method == "isotropic":
            return (self.smoothing.smoLens[mask], None, None)

        if method == "anisotropic":
            if plane_projection is not None or plane_projection_basis is not None:
                _, vals, vecs = project_smoTens_to_2d(
                    self.smoothing.smoTens[mask],
                    plane=plane_projection,
                    basis=plane_projection_basis,
                )
                return None, vals, vecs
            else:
                return (
                    None,
                    self.smoothing.smoTens_eigvals[mask],
                    self.smoothing.smoTens_eigvecs[mask],
                )

        else:
            raise ValueError(f"Unsupported deposition method '{method}'")

    def deposit_to_grid(
        self,
        fields: npt.ArrayLike | str | list[str],
        averaged: bool | Sequence[bool],
        gridnums: int | Sequence[int],
        extent: Sequence[Sequence[float]] | None = None,
        kernel_name: str | None = None,
        structure: Structure | None = None,
        adaptive: bool = True,
        plane_projection: list[int] | None = None,
        plane_projection_basis: (
            list[Sequence[float] | npt.NDArray[np.float32]] | None
        ) = None,
        integration_method: str = "midpoint",
        num_kernel_evaluations_per_axis: int = 4,
        eta_crit: float = 1.0,
        return_weights: bool = False,
        use_python: bool = False,
        use_openmp: bool = True,
        omp_threads: int | None = None,
    ) -> (
        npt.NDArray[np.floating]
        | tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
    ):
        """Deposit particle fields onto a structured grid using SPH.

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
            SPH kernel name.
        structure : Structure, optional
            Smoothing structure for deposition.
        adaptive : bool, default False
            Whether to use adaptive smoothing from the instance.
        plane_projection : List[int], optional
            Indices of the axes to project onto for 3D to 2D deposition.
        plane_projection_basis : List[Sequence[float] | npt.NDArray[np.float32]], optional
            This feature is currently not supported. Placeholder for future functionality to specify a custom projection basis.
        integration_method : str, default 'midpoint'
            Kernel integration method.
        num_kernel_evaluations_per_axis : int, default 4
            Resolution for kernel integration.
        eta_crit : float, default 1.0
            Anti-aliasing threshold to switch from sampled to full numerical quadrature.
        return_weights : bool, default False
            If True, returns the weights (density) grid as well.
        use_python : bool, default False
            Whether to force the Python instead of the C++ backend.
        use_openmp : bool, default True
            Whether to use multi-threading in the C++ backend.
        omp_threads : int, optional
            Number of threads for OpenMP.

        Returns
        -------
        Union[npt.NDArray[np.floating], tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]]
            Deposited field grid, and optionally the weights grid.

        """
        method, kn_res = self._resolve_deposition_method(kernel_name, structure)
        fields, fields_sizes = self._resolve_fields(fields)

        # adaptive = False is only supported for structure = separable or isotropic but not anisotropic
        if not adaptive and method == "anisotropic":
            raise ValueError(
                "Adaptive smoothing must be enabled for 'anisotropic' deposition, but found `adaptive`=False"
            )

        if omp_threads is not None and omp_threads < 1:
            raise ValueError(f"omp_threads must be >= 1, got {omp_threads}")

        # validate mutual exclusivity of projection arguments
        if plane_projection_basis is not None:
            raise ValueError(
                "Specifying 'plane_projection_basis' is currently not supported."
            )

        # resolve the averaged argument to a list of bools matching the number of field components
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

        # resolve gridnums and ensure it matches deposition dimension
        dep_dim = pos_temp.shape[1]
        gn = np.atleast_1d(gridnums).astype(np.int32)
        if gn.size == 1:
            gn = np.repeat(gn, dep_dim)
        if gn.size != dep_dim:
            raise ValueError(
                f"Length of 'gridnums' ({gn.size}) must match deposition dimension ({dep_dim})"
            )

        h, h_vals, h_vecs = self._prepare_deposition_smoothing(
            method,
            adaptive,
            pos_temp.shape[0],
            gn,
            d_lens,
            mask,
            dep_dim,
            plane_projection,
        )

        # Backend execution
        threads = 0 if omp_threads is None else int(omp_threads)
        func_name = f"{method}_{dep_dim}d"
        if self.verbose:
            print(
                f"[smudgy] Using {'python' if use_python else 'c++'} backend for {method} deposition ({func_name})"
            )

        fields_grid, weights_grid = _call_backend(
            func_name=func_name,
            use_python=use_python,
            use_openmp=use_openmp,
            omp_threads=threads,
            positions=pos_temp,
            quantities=fields_temp,
            particle_weights=weights_temp,
            smoothing_lengths=h,
            h_vals=h_vals,
            h_vecs=h_vecs,
            boxsizes=d_lens,
            gridnums=gn,
            periodic=periodic,
            kernel_name=kn_res,
            integration_method=integration_method,
            num_kernel_evaluations_per_axis=num_kernel_evaluations_per_axis,
            eta_crit=eta_crit,
        )

        # Post-processing
        for i, avg in enumerate(averaged):
            if i < fields_grid.shape[-1] and avg:
                fields_grid[..., i] /= weights_grid + 1e-10

        return (fields_grid, weights_grid) if return_weights else fields_grid
