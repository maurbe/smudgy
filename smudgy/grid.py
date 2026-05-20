"""Functions for creating regular grids of cell centers in 1D, 2D, and 3D domains."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Float32Array = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int_]
BoxInput = float | Sequence[float] | npt.ArrayLike
CellInput = int | Sequence[int] | npt.ArrayLike


def _resolve_boxsize(boxsize: BoxInput, dim: int) -> npt.NDArray[np.floating]:
    """Return per-dimension box lengths given scalar or array-like input.

    Parameters
    ----------
    boxsize
        Scalar or array-like input specifying box lengths.
    dim
        Number of spatial dimensions.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(dim,)`` with floating-point box lengths.

    """
    box_array = np.asarray(boxsize, dtype=float)
    if box_array.ndim == 0:
        return np.full(dim, float(box_array))
    if box_array.shape != (dim,):
        raise ValueError(f"'boxsize' must have length {dim}")
    return box_array


def _resolve_ncells(n_cells: CellInput, dim: int) -> IntArray:
    """Return per-dimension integer counts given scalar or array-like input.

    Parameters
    ----------
    n_cells
        Scalar or array-like input specifying grid resolution.
    dim
        Number of spatial dimensions.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(dim,)`` with integer counts.

    Raises
    ------
    ValueError
        If ``n_cells`` is not either a single positive integer or an array-like of positive integers with length ``dim``.

    """
    # Convert input to array, allow float but require integer values
    n_cells = np.asarray(n_cells)
    if n_cells.ndim == 0:
        value = int(n_cells)
        if value <= 0:
            raise ValueError("Grid resolution must be strictly positive")
        return np.full(dim, value, dtype=int)

    # Ensure correct shape
    if n_cells.shape != (dim,):
        raise ValueError(
            f"'n_cells' must either be a single value or have length {dim}"
        )

    # Convert to int and check positivity
    n_cells_full = np.array(n_cells, dtype=int)
    if np.any(n_cells_full <= 0):
        raise ValueError("Grid resolution values must be strictly positive")
    return n_cells_full


def _create_grid_nd(n_cells: CellInput, boxsize: BoxInput, dim: int) -> Float32Array:
    """Generate N-dimensional grid cell centers.

    Parameters
    ----------
    n_cells
        Scalar or array-like input specifying grid resolution.
    boxsize
        Scalar or array-like input specifying box lengths.
    dim
        Number of spatial dimensions.

    Returns
    -------
    numpy.ndarray
        Float32 array of shape ``(n_cells[0] * ... * n_cells[N-1], N)`` containing cell centers.

    """
    cells_along_axes = _resolve_ncells(n_cells, dim)
    box_lengths = _resolve_boxsize(boxsize, dim)
    deltas = box_lengths / cells_along_axes

    axes = [
        np.linspace(delta / 2.0, length - delta / 2.0, count)
        for delta, length, count in zip(deltas, box_lengths, cells_along_axes)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    grid_positions = np.stack([m.ravel() for m in mesh], axis=-1).astype("float32")
    return grid_positions


def create_grid_1d(n_cells: int, boxsize: BoxInput) -> Float32Array:
    """Generate 1D grid cell centers. Calls ``create_grid_nd`` with 1D parameters.

    Parameters
    ----------
    n_cells
        Number of cells along the axis.
    boxsize
        Physical size of the domain (scalar).

    Returns
    -------
    numpy.ndarray
        Float32 array of shape ``(n_cells, 1)`` with cell-center coordinates.

    """
    return _create_grid_nd(n_cells, boxsize, dim=1)


def create_grid_2d(n_cells: CellInput, boxsize: BoxInput) -> Float32Array:
    """Generate 2D grid cell centers. Calls ``create_grid_nd`` with 2D parameters.

    Parameters
    ----------
    n_cells
        Scalar or ``(2,)`` iterable with counts per axis.
    boxsize
        Scalar or ``(2,)`` iterable with domain lengths.

    Returns
    -------
    numpy.ndarray
        Float32 array of shape ``(n_cells[0] * n_cells[1], 2)`` containing cell centers.

    """
    return _create_grid_nd(n_cells, boxsize, dim=2)


def create_grid_3d(n_cells: CellInput, boxsize: BoxInput) -> Float32Array:
    """Generate 3D grid cell centers. Calls ``create_grid_nd`` with 3D parameters.

    Parameters
    ----------
    n_cells
        Scalar or ``(3,)`` iterable with counts per axis.
    boxsize
        Scalar or ``(3,)`` iterable with domain lengths.

    Returns
    -------
    numpy.ndarray
        Float32 array of shape ``(n_cells[0] * n_cells[1] * n_cells[2], 3)`` containing cell centers.

    """
    return _create_grid_nd(n_cells, boxsize, dim=3)


def grid_lines(edges, nx, ny=None, include_edges=True):
    """Create 2D grid lines for plotting.

    Parameters
    ----------
    edges
        Tuple of (xmin, xmax, ymin, ymax) defining the rectangular domain.
    nx
        Number of vertical lines to draw.
    ny
        Number of horizontal lines to draw. If None, defaults to nx.
    include_edges
        If True, lines include the edges (using np.linspace); if False, lines are interior-only (exclude endpoints).

    Returns
    -------
    tuple
        (vlines, hlines, (xmin, xmax, ymin, ymax)) where vlines and hlines are 1D arrays of line positions.

    """
    if ny is None:
        ny = nx
    xmin, xmax, ymin, ymax = edges
    nx = int(nx)
    ny = int(ny)
    if nx < 0 or ny < 0:
        raise ValueError("nx/ny must be non-negative")

    def _gen(a, b, M):
        if M <= 0:
            return np.array([], dtype=float)
        if include_edges:
            return np.linspace(a, b, M + 1)
        # interior-only: place M lines strictly inside (exclude endpoints)
        if M == 1:
            return np.array([(a + b) / 2.0])
        return np.linspace(a, b, M + 1)[1:-1]

    vlines = _gen(xmin, xmax, nx)
    hlines = _gen(ymin, ymax, ny)

    return (
        vlines,
        hlines,
        (ymin, ymax, xmin, xmax),
    )  # switched order here is correct for plotting
