"""Shim for the compiled C++ backend extension."""

from importlib import import_module

_ext = import_module("smudgy.core._cpp_functions_ext")

globals().update(_ext.__dict__)

# Aliases for backend-agnostic function names (expected by collection.py)
if hasattr(_ext, "_ngp_2d_cpp"):
    _ngp_2d = _ext._ngp_2d_cpp
if hasattr(_ext, "_ngp_3d_cpp"):
    _ngp_3d = _ext._ngp_3d_cpp
if hasattr(_ext, "_cic_2d_cpp"):
    _cic_2d = _ext._cic_2d_cpp
if hasattr(_ext, "_cic_3d_cpp"):
    _cic_3d = _ext._cic_3d_cpp
if hasattr(_ext, "_cic_2d_adaptive_cpp"):
    _cic_2d_adaptive = _ext._cic_2d_adaptive_cpp
if hasattr(_ext, "_cic_3d_adaptive_cpp"):
    _cic_3d_adaptive = _ext._cic_3d_adaptive_cpp
if hasattr(_ext, "_tsc_2d_cpp"):
    _tsc_2d = _ext._tsc_2d_cpp
if hasattr(_ext, "_tsc_3d_cpp"):
    _tsc_3d = _ext._tsc_3d_cpp
if hasattr(_ext, "_tsc_2d_adaptive_cpp"):
    _tsc_2d_adaptive = _ext._tsc_2d_adaptive_cpp
if hasattr(_ext, "_tsc_3d_adaptive_cpp"):
    _tsc_3d_adaptive = _ext._tsc_3d_adaptive_cpp
if hasattr(_ext, "_isotropic_2d_cpp"):
    _isotropic_2d = _ext._isotropic_2d_cpp
if hasattr(_ext, "_isotropic_3d_cpp"):
    _isotropic_3d = _ext._isotropic_3d_cpp
if hasattr(_ext, "_anisotropic_2d_cpp"):
    _anisotropic_2d = _ext._anisotropic_2d_cpp
if hasattr(_ext, "_anisotropic_3d_cpp"):
    _anisotropic_3d = _ext._anisotropic_3d_cpp

__all__ = getattr(
    _ext, "__all__", [name for name in _ext.__dict__ if not name.startswith("_")]
)
