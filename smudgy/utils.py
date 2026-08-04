"""Visualization utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

# =============================================================================
# Plotting helpers
# =============================================================================
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


# =============================================================================
# Video utils
# =============================================================================
"""
import cv2
# Step 1: Generate the 3D grid for the original array
def create_3d_grid(n):
    x = np.linspace(-n / 2, n / 2, n)
    y = np.linspace(-n / 2, n / 2, n)
    z = np.linspace(-n / 2, n / 2, n)
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing="ij")
    return x_grid, y_grid, z_grid


# Step 2: Create small grid with sidelength n * sqrt(2)/2, centered around [0, 0, 0]
def create_small_grid(n):
    side_length_small = n * np.sqrt(2) / 2
    x = np.linspace(-side_length_small / 2, side_length_small / 2, n)
    y = np.linspace(-side_length_small / 2, side_length_small / 2, n)
    z = np.linspace(-side_length_small / 2, side_length_small / 2, n)
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing="ij")
    return x_grid, y_grid, z_grid


# Step 3: Rotate grid according to the given angle phi (around the z-axis)
def rotate_grid(x_grid, y_grid, z_grid, phi_deg):
    phi = np.radians(phi_deg)
    rotation_matrix = np.array(
        [[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]]
    )

    coords = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])
    rotated_coords = np.dot(rotation_matrix, coords)

    x_rot = rotated_coords[0].reshape(x_grid.shape)
    y_rot = rotated_coords[1].reshape(y_grid.shape)
    z_rot = rotated_coords[2].reshape(z_grid.shape)

    return x_rot, y_rot, z_rot


# Step 4: Interpolate the original 3D array onto the rotated small grid
def interpolate_to_small_grid(original_array, x_rot, y_rot, z_rot, n):
    x_grid, y_grid, z_grid = create_3d_grid(n)

    interpolator = RegularGridInterpolator(
        (x_grid[:, 0, 0], y_grid[0, :, 0], z_grid[0, 0, :]), original_array
    )

    new_coords = np.vstack([x_rot.ravel(), y_rot.ravel(), z_rot.ravel()]).T
    interpolated_values = interpolator(new_coords)

    return interpolated_values.reshape(x_rot.shape)


# Step 5: Main function to create projections for each camera angle
def create_projections(original_array, num_angles):
    n, _, _ = original_array.shape
    phi_values = np.linspace(0, 360, num_angles)
    projections = []

    for phi in tqdm(phi_values, desc="creating frames"):
        x_small, y_small, z_small = create_small_grid(n)
        x_rot, y_rot, z_rot = rotate_grid(x_small, y_small, z_small, phi)

        interpolated_values = interpolate_to_small_grid(
            original_array, x_rot, y_rot, z_rot, n
        )
        projection = np.sum(interpolated_values, axis=1)
        projection = np.rot90(projection)
        projections.append(projection)

    return np.asarray(projections)


# Function to create video from projections
def array_to_video(array, cmaps, fps):

    movie_path = "video_projection.mp4"

    vmin, vmax = array.min(), array.max()

    writer = None

    for k in trange(len(array), desc="creating video"):

        fig, ax = plt.subplots(dpi=500)
        ax.imshow(array[k], vmin=vmin, vmax=vmax, cmap=cmaps[0])
        ax.axis("off")
        plt.tight_layout()
        fig.canvas.draw()

        frame = np.asarray(fig.canvas.buffer_rgba())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                movie_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )

        writer.write(frame)
        plt.close(fig)

    writer.release()

    return movie_path


# Function to display the video
def movie_grid(filenames):
    from IPython.display import display
    from ipywidgets import GridspecLayout, Video

    filenames = np.asarray(filenames)

    try:
        num_x, num_y = filenames.shape
    except:
        num_x, num_y = 1, len(filenames)
        filenames = filenames[np.newaxis, ...]

    grid = GridspecLayout(num_x, num_y)
    for i in range(num_y):
        for j in range(num_x):
            grid[j, i] = Video.from_file(filenames[j, i], play=True)

    display(grid)
    return grid


# New simplified function to generate projections and display video
def generate_projection_video(
    array, num_angles=360, duration=10, cmaps=None, transform=None, size=None
):
    if cmaps is None:
        cmaps = ["viridis"]

    assert (
        len(array.shape) == 3
    ), f"Expect input array to be of shape (n, n, n), found {array.shape}"

    if size is not None:
        from scipy.ndimage import zoom

        zoom_factors = tuple(size / s for s in array.shape)
        array = zoom(
            array, zoom_factors, order=1
        )  # order=1 is linear (trilinear in 3D)

    projections = create_projections(array, num_angles)
    print("projections shape:", projections.shape)

    # here we can apply a transform, e.g. log to represent the whole dynamic range
    if transform is not None:
        projections = transform(projections)

    fps = num_angles / duration
    movie_path = array_to_video(projections, cmaps, fps=fps)

    # show movies in cell
    # movie_grid([movie_path])


# -----------


# Step 5: Main function to create perspectives for each camera angle
def create_perspectives(original_array, num_angles):

    n, _, _ = original_array.shape
    phi_values = np.linspace(0, 360, num_angles)
    perspectives = []

    for phi in tqdm(phi_values, desc="creating frames"):
        x_small, y_small, z_small = create_small_grid(n)
        x_rot, y_rot, z_rot = rotate_grid(x_small, y_small, z_small, phi)

        interpolated_values = interpolate_to_small_grid(
            original_array, x_rot, y_rot, z_rot, n
        )
        perspectives.append(interpolated_values)

    return np.asarray(perspectives)
"""
