"""Test the analytical integrals of the kernels (all expected = 1.0)."""

import numpy as np
import pytest

from smudgy import compute_total_integral_spherical

DIMS = [1, 2, 3]
ETA_CRIT = [1, 4, 10]
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
@pytest.mark.parametrize("eta_crit", ETA_CRIT)
def test_kernel_integrals(kernel_name: str, dim: int, eta_crit: float):
    """Test that the integrals of the kernels are close to 1.0."""
    integral = compute_total_integral_spherical(
        kernel_name,
        dim,
        eta_crit=eta_crit,
    )
    assert np.allclose(
        # 0.1% accuracy constraint
        integral,
        1.0,
        atol=1e-3,
    ), f"Kernel integral failed for {kernel_name} in {dim}D -- integral = {integral}"
