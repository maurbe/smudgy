"""Test the total integrals of the kernels (all expected = 1.0)."""

import pytest
import numpy as np
from smudgy import PointCloud

DIMS = [2, 3]
METHODS = ["isotropic", "anisotropic"]
NUM_NEIGHBORS = [3, 4, 5]
MIN_KERNEL_EVALUATIONS_PER_AXIS = [2, 3, 4]
INTEGRAL_METHODS = ["midpoint", "trapezoidal", "simpson"]
KERNEL_NAMES = [
    "lucy",
    "gaussian",
    "cubic_spline",
    "quintic_spline",
    "wendland_c2",
    "wendland_c4",
    "wendland_c6",
]


# function that creates a point cloud within [0, 1]^dim with uniform weights
def create_uniform_point_cloud(dim: int) -> PointCloud:

    boxsize = 1.0
    num_points = int(1e4)
    positions = np.random.uniform(0, boxsize - 1e-6, size=(num_points, dim)).astype(
        np.float32
    )
    weights = np.ones(num_points, dtype=np.float32)
    return PointCloud(positions=positions, weights=weights, boxsize=boxsize)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("num_neighbors", NUM_NEIGHBORS)
@pytest.mark.parametrize(
    "min_kernel_evaluations_per_axis", MIN_KERNEL_EVALUATIONS_PER_AXIS
)
@pytest.mark.parametrize("integral_method", INTEGRAL_METHODS)
@pytest.mark.parametrize("kernel_name", KERNEL_NAMES)
def test_weight_conservation(
    dim: int,
    method: str,
    num_neighbors: int,
    min_kernel_evaluations_per_axis: int,
    integral_method: str,
    kernel_name: str,
):
    """Test that the deposited weights are conserved (i.e., sum to 1.0)."""

    gridnums = 32

    # set up the point cloud and kernel
    pc = create_uniform_point_cloud(dim)
    pc.setup(
        num_neighbors=num_neighbors,
        method=method,
    )
    pc.compute_smoothing_lengths()

    _, weights_sph = pc.deposit_to_grid(
        fields=pc.weights,
        averaged=False,
        gridnums=gridnums,
        method=method,
        kernel_name=kernel_name,
        min_kernel_evaluations_per_axis=min_kernel_evaluations_per_axis,
        integration=integral_method,
        return_weights=True,
    )

    weight_deposited = np.sum(weights_sph)
    weight_true = np.sum(pc.weights)
    ratio = weight_deposited / weight_true
    assert np.isclose(
        ratio, 1.0, atol=1e-3
    ), f"Weight conservation failed for {kernel_name} in {dim}D"
