from __future__ import annotations

from .ops import PointCloud


def check_openmp() -> bool:
	"""Return whether the C++ backend was built with OpenMP support."""
	try:
		from .core import _cpp_functions as cppfunc
	except Exception:
		return False
	return bool(getattr(cppfunc, "has_openmp", False))


__all__ = ["PointCloud", "check_openmp"]

