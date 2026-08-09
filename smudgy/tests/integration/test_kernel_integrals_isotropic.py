"""Test the analytical integrals of the kernels (all expected = 1.0)."""

import numpy as np
import pytest

from smudgy import compute_total_integral_spherical

DIMS = [1, 2, 3]
NUM_KERNEL_EVALUATIONS_PER_AXIS = [1, 17, 23]
KERNEL_NAMES = [
    "tophat",
    "tsc",
    "lucy",
    "gaussian",
    "cubic_spline",
    "quintic_spline",
    "wendland_c2",
    "wendland_c4",
    "wendland_c6",
]


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("kernel_name", KERNEL_NAMES)
@pytest.mark.parametrize(
    "num_kernel_evaluations_per_axis", NUM_KERNEL_EVALUATIONS_PER_AXIS
)
def test_kernel_integrals(
    kernel_name: str, dim: int, num_kernel_evaluations_per_axis: int
):
    """Test that the integrals of the kernels are close to 1.0."""
    integral = compute_total_integral_spherical(
        kernel_name,
        dim,
        num_kernel_evaluations_per_axis=num_kernel_evaluations_per_axis,
    )
    assert np.allclose(
        # 0.1% accuracy constraint
        integral,
        1.0,
        atol=1e-3,
    ), f"Kernel integral failed for {kernel_name} in {dim}D -- integral = {integral}"
