"""Test the analytical integrals of the kernels (all expected = 1.0)."""

import numpy as np
import pytest

from smudgy import compute_total_integral_separable

DIMS = [1, 2, 3]
KERNEL_NAMES = [
    "tophat",
    "tsc",
    "gaussian",
]


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("kernel_name", KERNEL_NAMES)
def test_kernel_integrals(
    kernel_name: str,
    dim: int,
):
    """Test that the integrals of the kernels are close to 1.0."""
    integral = compute_total_integral_separable(
        kernel_name,
        dim,
    )
    assert np.allclose(
        # 0.1% accuracy constraint
        integral,
        1.0,
        atol=1e-4,
    ), f"Kernel integral failed for {kernel_name} in {dim}D -- integral = {integral}"
