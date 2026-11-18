"""Point Cloud utilities."""

from __future__ import annotations

from .backend.quadrants.kernels import (
    compute_total_integral_separable,
    compute_total_integral_spherical,
    get_separable_kernel_values_1D,
    get_spherical_kernel_values_1D,
)
from .backend.numpy.kernels import get_kernel
from .pointcloud import PointCloud


def get_kernel_shapes_1D(
    kernel_name: str, structure: str = "isotropic"
) -> tuple[list[float], list[float]]:
    """Get the 1D kernel profile (q values and kernel values).

    Parameters
    ----------
    kernel_name : str
        Name of the kernel, e.g. ``"cubic_spline"``.
    structure : str, default "isotropic"
        ``"separable"`` returns the 1D profile of the separable (tensor
        product) kernel, available for ``"tophat"``, ``"tsc"`` and
        ``"gaussian"``; ``"isotropic"`` or ``"covariant"`` return the radial
        profile.

    Returns
    -------
    tuple of array_like
        Normalized coordinates q and kernel values W(q).

    """
    if structure == "separable":
        return get_separable_kernel_values_1D(kernel_name)
    return get_spherical_kernel_values_1D(kernel_name)


__all__ = [
    "PointCloud",
    "compute_total_integral_separable",
    "compute_total_integral_spherical",
    "get_kernel_shapes_1D",
    "get_kernel"
]
