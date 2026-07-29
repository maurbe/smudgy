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
