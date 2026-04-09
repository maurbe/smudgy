"""Test the analytical integrals of the kernels (all expected = 1.0)."""

import pytest
import numpy as np
from smudgy import check_kernel_integral

DIMS = [1, 2, 3]
MIN_KERNEL_EVALUATIONS_PER_AXIS = [1, 17, 23, 100]
KERNEL_NAMES = [
    "lucy",
    "gaussian",
    "cubic_spline",
    "quintic_spline",
    "wendland_c2",
    "wendland_c4",
    "wendland_c6",
    "tophat",
    "tsc",
    #"tophat_rect", # is defined on cartesian grid, so cannot use spherical function
    #"tsc_rect", # is defined on cartesian grid, so cannot use spherical function
]


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("kernel_name", KERNEL_NAMES)
@pytest.mark.parametrize(
    "min_kernel_evaluations_per_axis", MIN_KERNEL_EVALUATIONS_PER_AXIS
)
def test_kernel_integrals(
    kernel_name: str, dim: int, min_kernel_evaluations_per_axis: int
):
    """Test that the kernel integrals are correct."""
    integral = check_kernel_integral(
        kernel_name,
        dim,
        min_kernel_evaluations_per_axis=min_kernel_evaluations_per_axis,
    )
    print("Integral", integral)
    assert np.allclose(
        integral, 1.0, atol=1e-2
    ), f"Kernel integral failed for {kernel_name} in {dim}D"
