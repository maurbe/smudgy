"""Utility functions for SPH operations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy import spatial

FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.int_]
BoxInput = float | Sequence[float] | npt.ArrayLike


def build_kdtree(
    points: npt.ArrayLike, boxsize: BoxInput | None = None
) -> spatial.cKDTree:
    """Construct a cKDTree with optional periodic box sizes.

    If ``boxsize`` is provided, the tree will use periodic boundary conditions.

    Parameters
    ----------
    points
            Array of shape ``(N, D)`` with particle coordinates, where ``N`` is the
            number of particles and ``D`` the dimension.
    boxsize
            Scalar or ``(D,)`` array defining periodic box lengths, or ``None``.

    Returns
    -------
    scipy.spatial.cKDTree
            Tree built from ``points``.

    """
    return spatial.cKDTree(points, boxsize=boxsize)


def query_kdtree(
    tree: spatial.cKDTree, points: npt.ArrayLike, k: int
) -> tuple[FloatArray, IntArray]:
    """Query a cKDTree for the k nearest neighbors of given points.

    Parameters
    ----------
    tree
            cKDTree instance to query.
    points
            Array of shape ``(M, D)`` with query coordinates, where ``M`` is the
            number of query positions.
    k
            Number of nearest neighbors to return.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
            Tuple of ``(distances, indices)`` from ``cKDTree.query``.

    """
    return tree.query(points, k=k, workers=-1)


def shift_coordinates(coordinates: npt.ArrayLike) -> FloatArray:
    """Shift coordinates so the minimum per-axis value is at zero.

    Parameters
    ----------
    coordinates
            Array of shape ``(N, D)`` with particle coordinates.

    Returns
    -------
    numpy.ndarray
            Shifted coordinates with the same shape as ``coordinates``.

    Raises
    ------
    AssertionError
            If the input array is not 2D.

    """
    coordinates_array = np.asarray(coordinates)
    assert (
        coordinates_array.ndim == 2
    ), "Input coordinates must be a 2D array of shape (N, D)"
    return coordinates_array - coordinates_array.min(axis=0)


def coordinate_difference_with_pbc(
    x: npt.ArrayLike, y: npt.ArrayLike, boxsize: BoxInput
) -> FloatArray:
    """Compute coordinate differences with periodic boundary conditions.

    Parameters
    ----------
    x
            Array-like coordinates.
    y
            Array-like coordinates to subtract from ``x``.
    boxsize
            Scalar or ``(D,)`` array defining periodic box lengths.

    Returns
    -------
    numpy.ndarray
            Coordinate differences wrapped into the range
            $[-0.5 * boxsize, 0.5 * boxsize)$ per dimension.

    Raises
    ------
    AssertionError
            If the input coordinate arrays do not have the same spatial dimension.
            If ``boxsize`` is an array, if it is not 1D or if its length does not match the coordinate dimension.

    """
    shape_x = x.shape
    shape_y = y.shape
    assert (
        shape_x[-1] == shape_y[-1]
    ), "Input coordinate arrays must have the same spatial dimension"

    # compute differences
    coordinate_differences = np.asarray(x) - np.asarray(y)

    # early exit if boxsize is None (non-periodic case)
    if boxsize is None:
        return coordinate_differences

    # prepare boxsize array for broadcasting
    if np.isscalar(boxsize):
        box_arr = np.array([boxsize] * shape_x[-1])
    else:
        assert np.asarray(boxsize).ndim == 1, "'boxsize' must be a scalar or 1D array"
        assert (
            len(boxsize) == shape_x[-1]
        ), "Length of 'boxsize' must match coordinate dimension"
        box_arr = np.asarray(boxsize)

    half_box = 0.5 * box_arr
    return (coordinate_differences + half_box) % box_arr - half_box


def compute_smoLens(
    tree: spatial.cKDTree,
    num_neighbors: int,
    query_positions: npt.ArrayLike | None = None,
) -> tuple[FloatArray, IntArray, FloatArray]:
    """Estimate smoothing length as half the distance to the Nth neighbor.

    Parameters
    ----------
    tree
            cKDTree built from particle positions.
    num_neighbors
            Number of neighbors used for the estimate.
    query_positions
            Array of shape ``(M, D)`` with positions where smoothing length is evaluated.
            If ``None``, uses particle positions from the tree.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Tuple of ``(hsm, nn_inds, nn_dists)`` where ``hsm`` has shape ``(M,)``,
            ``nn_dists`` has shape ``(M, num_neighbors)``, and ``nn_inds`` has shape
            ``(M, num_neighbors)``.

    """
    if query_positions is None:
        query_positions = tree.data

    nn_dists, nn_inds = query_kdtree(tree, query_positions, k=num_neighbors)
    hsm = nn_dists[:, -1]  # * 0.5
    return hsm, nn_inds, nn_dists


def compute_smoTens(
    tree: spatial.cKDTree,
    weights: npt.ArrayLike,
    num_neighbors: int,
    query_positions: npt.ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, IntArray, FloatArray]:
    """Compute anisotropic smoothing tensor using a covariance-based method.

    Implements the method from Marinho (2021), generalized for 2D and 3D.

    Parameters
    ----------
    tree
            cKDTree built from particle positions.
    weights
            Array of shape ``(N,)`` with particle weights.
    num_neighbors
            Number of neighbors used for covariance estimation.
    query_positions
            Array of shape ``(M, D)`` where the tensor is evaluated.
            If ``None``, uses particle positions from the tree.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Tuple ``(H, eigvals, eigvecs, nn_inds, nn_dists, rel_coords)`` where
            ``H`` has shape ``(M, D, D)``, ``eigvals`` has shape ``(M, D)``,
            ``eigvecs`` has shape ``(M, D, D)``, ``nn_inds`` has shape
            ``(M, num_neighbors)``, ``nn_dists`` has shape ``(M, num_neighbors)``,
            and ``rel_coords`` has shape ``(M, num_neighbors, D)``.

    """
    pos = tree.data
    boxsize = tree.boxsize
    spatial_dim = pos.shape[-1]

    if spatial_dim not in (2, 3):
        raise ValueError(
            "[smudgy] Only 2D and 3D positions are supported for anisotropic smoothing tensors."
        )

    if query_positions is None:
        query_positions = pos
    weights = weights.flatten() if weights.ndim > 1 else weights

    # Find nearest neighbors
    nn_dists, nn_inds = query_kdtree(tree, query_positions, k=num_neighbors)
    neighbor_coords = pos[nn_inds]
    neighbor_weights = weights[nn_inds]

    # Account for periodic boundary conditions
    rel_coords = coordinate_difference_with_pbc(
        neighbor_coords, query_positions[:, np.newaxis, :], boxsize
    )

    # Compute mass-weighted covariance matrix
    outer = np.einsum("...i, ...j -> ...ij", rel_coords, rel_coords)
    outer = outer * neighbor_weights[..., np.newaxis, np.newaxis]
    Sigma = (
        np.sum(outer, axis=1)
        / np.sum(neighbor_weights, axis=1)[..., np.newaxis, np.newaxis]
    )

    # Compute eigendecomposition and construct smoothing tensor H = VΛV^T
    # Relative regularization of the Sigma matrix, to make sure H, eigvals, and vecs are all finite
    eps = 1e-6
    trace = np.trace(Sigma, axis1=-2, axis2=-1)[..., None, None]
    Sigma_reg = Sigma + eps * trace * np.eye(spatial_dim)[None, :, :]

    eigvals, eigvecs = np.linalg.eigh(Sigma_reg)
    eigvals = np.sqrt(np.clip(eigvals, 0, None))
    Λ = eigvals[..., np.newaxis] * np.eye(spatial_dim)
    H = np.matmul(np.matmul(eigvecs, Λ), np.transpose(eigvecs, axes=(0, 2, 1)))

    return H, eigvals, eigvecs, nn_inds, nn_dists, rel_coords


def project_smoTens_to_2d(
    h_tensor: npt.ArrayLike,
    plane: list[int] | None = None,
    basis: list[Sequence[float], Sequence[float]] | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Project 3D smoothing tensors onto a 2D plane.

    Parameters
    ----------
    h_tensor
            Array of shape ``(N, 3, 3)`` with 3D smoothing tensors.
    plane
            Indices of the axes to project onto for 3D to 2D deposition.
            Mutually exclusive with ``basis``.
    basis
            List of basis vectors ``(e1, e2)`` spanning the projection plane.
            Each vector should be array-like of length 3.
            Mutually exclusive with ``plane``.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Tuple ``(h_tensor_2d, eigvals, eigvecs)`` where ``h_tensor_2d`` has
            shape ``(N, 2, 2)``, ``eigvals`` has shape ``(N, 2)``, and ``eigvecs``
            has shape ``(N, 2, 2)``.

    Raises
    ------
    ValueError
            If neither or both of ``plane`` and ``basis`` are provided.
            If ``plane`` is not one of the allowed values or types or does not have length 2.
            If ``basis`` is not a 2-tuple of 3D vectors.

    """
    # Validate inputs
    if plane is None and basis is None:
        raise ValueError("Either 'plane' or 'basis' must be provided")
    if plane is not None and basis is not None:
        raise ValueError(
            "'plane' and 'basis' are mutually exclusive, only provide one of the two"
        )

    # Validate that plane is either list or array of length 2 with valid indices
    if plane is not None:
        if not (isinstance(plane, (list, np.ndarray)) and len(plane) == 2):
            raise ValueError("'plane' must be a list or array of length 2")
        if any(p not in [0, 1, 2] for p in plane):
            raise ValueError("'plane' indices must be in the range [0, 2]")

    # Define projection basis vectors
    if basis is not None:
        if len(basis) != 2:
            raise ValueError("'basis' must be a 2-tuple of vectors")
        e1, e2 = basis
    else:
        unit_vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        e1 = unit_vectors[plane[0]]
        e2 = unit_vectors[plane[1]]

    # Compute projected tensors: (P @ H^-1 @ P^T)^-1
    projection_matrix = np.array([e1, e2], dtype="float32")  # (2, 3)
    h_tensor_inv = np.linalg.inv(h_tensor)  # (N, 3, 3)

    # Vectorized computation: P @ H_inv @ P^T for all particles
    temp = np.einsum(
        "ij,njk,lk->nil", projection_matrix, h_tensor_inv, projection_matrix
    )
    h_tensor_2d = np.linalg.inv(temp)  # (N, 2, 2)

    # Compute eigendecomposition of 2D tensors
    eigvals, eigvecs = np.linalg.eigh(h_tensor_2d)
    return h_tensor_2d, eigvals, eigvecs









# Video utils
# no transfer function so far, only projected (interpolated) views

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from tqdm.notebook import tqdm, trange


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


import cv2
import numpy as np
from tqdm import trange


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

"""
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
