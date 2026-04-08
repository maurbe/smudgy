"""Point Cloud utilities."""

from __future__ import annotations

from .main import *
from .utils import *
from .grid import *


def check_openmp() -> bool:
    """Return whether the C++ backend was built with OpenMP support."""
    try:
        from .core import _cpp_functions as cppfunc
    except Exception:
        return False
    return bool(getattr(cppfunc, "has_openmp", False))


def check_kernel_integral(
    kernel_name: str, dim: int, min_kernel_evaluations_per_axis: int
) -> bool:
    """Check whether the kernel integrals are correct."""
    from .core import _cpp_functions as cppfunc

    return cppfunc.compute_total_kernel_integral(
        kernel_name, dim, min_kernel_evaluations_per_axis
    )


__all__ = ["PointCloud", "check_openmp", "check_kernel_integral"]
