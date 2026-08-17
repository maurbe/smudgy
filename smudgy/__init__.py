"""Point Cloud utilities."""

from __future__ import annotations

from .backend.taichi.kernels import (
    compute_total_integral_separable,
    compute_total_integral_spherical,
    get_separable_kernel_values_1D,
    get_spherical_kernel_values_1D,
)
from .backend.numpy.kernels import get_kernel
from .pointcloud import PointCloud


def get_kernel_shapes_1D(kernel_name: str) -> tuple[list[float], list[float]]:
    """Get the 1D kernel shapes (q values and kernel values)."""
    if "separable" in kernel_name:
        return get_separable_kernel_values_1D(kernel_name)
    else:
        return get_spherical_kernel_values_1D(kernel_name)


__all__ = [
    "PointCloud",
    "compute_total_integral_separable",
    "compute_total_integral_spherical",
    "get_kernel_shapes_1D",
    "get_kernel"
]
