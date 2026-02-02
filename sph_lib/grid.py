from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int_]
BoxInput = float | Sequence[float] | npt.ArrayLike
CountInput = int | Sequence[int] | npt.ArrayLike


def _normalize_boxsize(boxsize: BoxInput, 
                       dims: int) -> FloatArray:
    """Return per-dimension box lengths given scalar or array-like input."""
    box_array = np.asarray(boxsize, dtype=float)
    if box_array.ndim == 0:
        return np.full(dims, float(box_array))
    if box_array.shape != (dims,):
        raise ValueError(f"'boxsize' must have length {dims}")
    return box_array


def _normalize_counts(counts: CountInput, 
                      dims: int) -> IntArray:
    """Return per-dimension integer counts given scalar or array-like input."""
    count_array = np.asarray(counts, dtype=int)
    if count_array.ndim == 0:
        value = int(count_array)
        if value <= 0:
            raise ValueError("Grid resolution must be positive")
        return np.full(dims, value, dtype=int)
    if count_array.shape != (dims,):
        raise ValueError(f"'counts' must either be a single value or have length {dims}")
    if np.any(count_array <= 0):
        raise ValueError("Grid resolution values must be positive")
    return count_array.astype(int, copy=False)


def create_grid_1d(nx: int, 
                   boxsize: float) -> Float32Array:
    """Generate 1D grid cell centers.

    Args:
        nx: Number of cells along the axis.
        boxsize: Physical size of the domain (scalar).

    Returns:
        np.ndarray of shape (nx, 1) with cell-center coordinates.
    """

    dx = boxsize / nx
    x = np.linspace(dx / 2.0, boxsize - dx / 2.0, nx)
    x = x[:, np.newaxis]
    return x.astype("float32")


def create_grid_2d(n_cells: CountInput, 
                   boxsize: BoxInput) -> Float32Array:
    """Generate 2D grid cell centers.

    Args:
        n_cells: Scalar or (2,) iterable with counts per axis.
        boxsize: Scalar or (2,) iterable with domain lengths.

    Returns:
        np.ndarray of shape (nx * ny, 2) containing cell centers.
    """

    counts = _normalize_counts(n_cells, 2)
    box_lengths = _normalize_boxsize(boxsize, 2)
    dx = box_lengths[0] / counts[0]
    dy = box_lengths[1] / counts[1]

    x = np.linspace(dx / 2.0, box_lengths[0] - dx / 2.0, counts[0])
    y = np.linspace(dy / 2.0, box_lengths[1] - dy / 2.0, counts[1])

    xx, yy = np.meshgrid(x, y, indexing="ij")
    grid_positions = np.stack((xx.ravel(), yy.ravel()), axis=-1).astype("float32")
    return grid_positions


def create_grid_3d(n_cells: CountInput, 
                   boxsize: BoxInput) -> Float32Array:
    """Generate 3D grid cell centers.

    Args:
        n_cells: Scalar or (3,) iterable with counts per axis.
        boxsize: Scalar or (3,) iterable with domain lengths.

    Returns:
        np.ndarray of shape (nx * ny * nz, 3) containing cell centers.
    """

    counts = _normalize_counts(n_cells, 3)
    box_lengths = _normalize_boxsize(boxsize, 3)
    dx = box_lengths[0] / counts[0]
    dy = box_lengths[1] / counts[1]
    dz = box_lengths[2] / counts[2]

    x = np.linspace(dx / 2.0, box_lengths[0] - dx / 2.0, counts[0])
    y = np.linspace(dy / 2.0, box_lengths[1] - dy / 2.0, counts[1])
    z = np.linspace(dz / 2.0, box_lengths[2] - dz / 2.0, counts[2])

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    grid_positions = np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=-1).astype(
        "float32"
    )
    return grid_positions
