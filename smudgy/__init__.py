"""Point Cloud utilities."""

from __future__ import annotations

from .core._cpp_functions import (
    compute_total_integral_separable,
    compute_total_integral_spherical,
    get_separable_kernel_values_1D,
    get_spherical_kernel_values_1D,
)
from .pointcloud import PointCloud


def check_openmp() -> bool:
    """Return whether the C++ backend was built with OpenMP support."""
    try:
        from .core import _cpp_functions as cppfunc
    except Exception:
        return False
    return bool(getattr(cppfunc, "has_openmp", False))


def compute_kernel_integral(
    kernel_name: str, dim: int, min_kernel_evaluations_per_axis: int = None
) -> bool:
    """Check whether the kernel integral sums to 1."""
    from .core import _cpp_functions as cppfunc

    if "separable" in kernel_name:
        return cppfunc.compute_total_integral_separable(kernel_name, dim)
    else:
        return cppfunc.compute_total_integral_spherical(
            kernel_name, dim, min_kernel_evaluations_per_axis
        )


def get_kernel_shapes_1D(kernel_name: str) -> tuple[list[float], list[float]]:
    """Get the 1D kernel shapes (q values and kernel values)."""
    from .core import _cpp_functions as cppfunc

    if "separable" in kernel_name:
        return cppfunc.get_separable_kernel_values_1D(kernel_name)
    else:
        return cppfunc.get_spherical_kernel_values_1D(kernel_name)


__all__ = [
    "PointCloud",
    "compute_total_integral_separable",
    "compute_total_integral_spherical",
    "get_separable_kernel_values_1D",
    "get_spherical_kernel_values_1D",
    "check_openmp",
    "compute_kernel_integral",
    "get_kernel_shapes_1D",
]
