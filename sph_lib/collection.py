"""Backend-agnostic deposition function wrappers."""

from __future__ import annotations

from typing import Any, Callable

from .core import _python_functions as _py_backend
from .core import _cpp_functions as _cpp_backend


_PYTHON_UNSUPPORTED = {
    "cic_2d_adaptive",
    "cic_3d_adaptive",
    "tsc_2d_adaptive",
    "tsc_3d_adaptive",
    "isotropic_2d",
    "isotropic_3d",
    "anisotropic_2d",
    "anisotropic_3d",
}


def _select_backend(use_python: bool):
    return _py_backend if use_python else _cpp_backend


def _call_backend(func_name: str, use_python: bool, *args: Any, use_openmp: bool, omp_threads: int):
    if use_python and func_name in _PYTHON_UNSUPPORTED:
        raise NotImplementedError(
            f"Python backend does not implement '{func_name}'. Set use_python=False to use the C++ backend."
        )
    backend = _select_backend(use_python)
    # All backend functions now have a leading underscore
    backend_func_name = f"_{func_name}"
    func: Callable[..., Any] = getattr(backend, backend_func_name)
    if use_python:
        return func(*args)
    return func(*args, use_openmp, omp_threads)


# NGP

def ngp_2d(positions, quantities, boxsizes, gridnums, periodic, *, use_python=False, use_openmp=True, omp_threads=0):
    """
    Deposit particle quantities onto a 2D grid using NGP (C++ backend).

    Parameters
    ----------
    pos : numpy.ndarray, shape (N, 2)
        Particle positions, where ``N`` is the number of particles.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (2,)
        Domain size per axis.
    gridnums : array_like, shape (2,)
        Number of grid cells per axis.
    periodic : array_like of bool, shape (2,)
        Periodic boundaries per axis.
    use_openmp : bool
        Enable OpenMP parallelism.
    omp_threads : int
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy)
        Weight sum per cell.
    """
    return _call_backend("ngp_2d", use_python, positions, quantities, boxsizes, gridnums, periodic,
                         use_openmp=use_openmp, omp_threads=omp_threads)


def ngp_3d(positions, quantities, boxsizes, gridnums, periodic, *, use_python=False, use_openmp=True, omp_threads=0):
    """
    Deposit particle quantities onto a 3D grid using NGP (C++ backend).

    Parameters
    ----------
    pos : numpy.ndarray, shape (N, 3)
        Particle positions, where ``N`` is the number of particles.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (3,)
        Domain size per axis.
    gridnums : array_like, shape (3,)
        Number of grid cells per axis.
    periodic : array_like of bool, shape (3,)
        Periodic boundaries per axis.
    use_openmp : bool
        Enable OpenMP parallelism.
    omp_threads : int
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, Gz, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy, Gz)
        Weight sum per cell.
    """
    return _call_backend("ngp_3d", use_python, positions, quantities, boxsizes, gridnums, periodic,
                         use_openmp=use_openmp, omp_threads=omp_threads)


# CIC

def cic_2d(positions, quantities, boxsizes, gridnums, periodic, *, use_python=False, use_openmp=True, omp_threads=0):
    """
    Deposit particle quantities onto a 2D grid using CIC (C++ backend).

    Parameters
    ----------
    pos : numpy.ndarray, shape (N, 2)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (2,)
        Domain size per axis.
    gridnums : array_like, shape (2,)
        Number of grid cells per axis.
    periodic : array_like of bool, shape (2,)
        Periodic boundaries per axis.
    use_openmp : bool
        Enable OpenMP parallelism.
    omp_threads : int
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy)
        Weight sum per cell.
    """
    return _call_backend("cic_2d", use_python, positions, quantities, boxsizes, gridnums, periodic,
                         use_openmp=use_openmp, omp_threads=omp_threads)


def cic_3d(positions, quantities, boxsizes, gridnums, periodic, *, use_python=False, use_openmp=True, omp_threads=0):
    """
    Deposit particle quantities onto a 3D grid using CIC (C++ backend).

    Parameters
    ----------
    pos : numpy.ndarray, shape (N, 3)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (3,)
        Domain size per axis.
    gridnums : array_like, shape (3,)
        Number of grid cells per axis.
    periodic : array_like of bool, shape (3,)
        Periodic boundaries per axis.
    use_openmp : bool
        Enable OpenMP parallelism.
    omp_threads : int
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, Gz, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy, Gz)
        Weight sum per cell.
    """
    return _call_backend("cic_3d", use_python, positions, quantities, boxsizes, gridnums, periodic,
                         use_openmp=use_openmp, omp_threads=omp_threads)


def cic_adaptive_2d(positions, quantities, boxsizes, gridnums, periodic, pcellsizesHalf, *, use_python=False, use_openmp=True, omp_threads=0):
    """
    Deposit particle quantities onto a 2D grid using adaptive CIC (C++ backend).

    Parameters
    ----------
    pos : numpy.ndarray, shape (N, 2)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (2,)
        Domain size per axis.
    gridnums : array_like, shape (2,)
        Number of grid cells per axis.
    periodic : array_like of bool, shape (2,)
        Periodic boundaries per axis.
    pcellsizesHalf : numpy.ndarray, shape (N, 2)
        Half cell sizes per particle (adaptive support).
    use_openmp : bool
        Enable OpenMP parallelism.
    omp_threads : int
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy)
        Weight sum per cell.
    """
    return _call_backend("cic_2d_adaptive", use_python, positions, quantities, boxsizes, gridnums, periodic, pcellsizesHalf,
                         use_openmp=use_openmp, omp_threads=omp_threads)


def cic_adaptive_3d(positions, quantities, boxsizes, gridnums, periodic, pcellsizesHalf, *, use_python=False, use_openmp=True, omp_threads=0):
    """
    Deposit particle quantities onto a 3D grid using adaptive CIC (C++ backend).

    Parameters
    ----------
    pos : numpy.ndarray, shape (N, 3)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (3,)
        Domain size per axis.
    gridnums : array_like, shape (3,)
        Number of grid cells per axis.
    periodic : array_like of bool, shape (3,)
        Periodic boundaries per axis.
    pcellsizesHalf : numpy.ndarray, shape (N, 3)
        Half cell sizes per particle (adaptive support).
    use_openmp : bool
        Enable OpenMP parallelism.
    omp_threads : int
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, Gz, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy, Gz)
        Weight sum per cell.
    """
    return _call_backend("cic_3d_adaptive", use_python, positions, quantities, boxsizes, gridnums, periodic, pcellsizesHalf,
                         use_openmp=use_openmp, omp_threads=omp_threads)


# TSC

def tsc_2d(positions, quantities, boxsizes, gridnums, periodic, *, use_python=False, use_openmp=True, omp_threads=0):
    """
    Deposit particle quantities onto a 2D grid using TSC (C++ backend).

    Parameters
    ----------
    pos : numpy.ndarray, shape (N, 2)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (2,)
        Domain size per axis.
    gridnums : array_like, shape (2,)
        Number of grid cells per axis.
    periodic : array_like of bool, shape (2,)
        Periodic boundaries per axis.
    use_openmp : bool
        Enable OpenMP parallelism.
    omp_threads : int
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy)
        Weight sum per cell.
    """
    return _call_backend("tsc_2d", use_python, positions, quantities, boxsizes, gridnums, periodic,
                         use_openmp=use_openmp, omp_threads=omp_threads)


def tsc_3d(positions, quantities, boxsizes, gridnums, periodic, *, use_python=False, use_openmp=True, omp_threads=0):
    """
    Deposit particle quantities onto a 3D grid using TSC (C++ backend).

    Parameters
    ----------
    pos : numpy.ndarray, shape (N, 3)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (3,)
        Domain size per axis.
    gridnums : array_like, shape (3,)
        Number of grid cells per axis.
    periodic : array_like of bool, shape (3,)
        Periodic boundaries per axis.
    use_openmp : bool
        Enable OpenMP parallelism.
    omp_threads : int
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, Gz, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy, Gz)
        Weight sum per cell.
    """
    return _call_backend("tsc_3d", use_python, positions, quantities, boxsizes, gridnums, periodic,
                         use_openmp=use_openmp, omp_threads=omp_threads)


def tsc_adaptive_2d(positions, quantities, boxsizes, gridnums, periodic, pcellsizesHalf, *, use_python=False, use_openmp=True, omp_threads=0):
    """
    Deposit particle quantities onto a 2D grid using adaptive TSC (C++ backend).

    Parameters
    ----------
    pos : numpy.ndarray, shape (N, 2)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (2,)
        Domain size per axis.
    gridnums : array_like, shape (2,)
        Number of grid cells per axis.
    periodic : array_like of bool, shape (2,)
        Periodic boundaries per axis.
    pcellsizesHalf : numpy.ndarray, shape (N, 2)
        Half cell sizes per particle (adaptive support).
    use_openmp : bool
        Enable OpenMP parallelism.
    omp_threads : int
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy)
        Weight sum per cell.
    """
    return _call_backend("tsc_2d_adaptive", use_python, positions, quantities, boxsizes, gridnums, periodic, pcellsizesHalf,
                         use_openmp=use_openmp, omp_threads=omp_threads)


def tsc_adaptive_3d(positions, quantities, boxsizes, gridnums, periodic, pcellsizesHalf, *, use_python=False, use_openmp=True, omp_threads=0):
    """
    Deposit particle quantities onto a 3D grid using adaptive TSC (C++ backend).

    Parameters
    ----------
    pos : numpy.ndarray, shape (N, 3)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (3,)
        Domain size per axis.
    gridnums : array_like, shape (3,)
        Number of grid cells per axis.
    periodic : array_like of bool, shape (3,)
        Periodic boundaries per axis.
    pcellsizesHalf : numpy.ndarray, shape (N, 3)
        Half cell sizes per particle (adaptive support).
    use_openmp : bool
        Enable OpenMP parallelism.
    omp_threads : int
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, Gz, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy, Gz)
        Weight sum per cell.
    """
    return _call_backend("tsc_3d_adaptive", use_python, positions, quantities, boxsizes, gridnums, periodic, pcellsizesHalf,
                         use_openmp=use_openmp, omp_threads=omp_threads)


# SPH kernels

def isotropic_2d(positions, quantities, boxsizes, gridnums, periodic, hsm, kernel_name, integration_method, min_kernel_evaluations, *, use_python=False, use_openmp=True, omp_threads=0):
    """
    Deposit particle quantities onto a 2D grid using an isotropic SPH kernel (C++ backend).

    Parameters
    ----------
    pos : numpy.ndarray, shape (N, 2)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (2,)
        Domain size per axis.
    gridnums : array_like, shape (2,)
        Number of grid cells per axis.
    periodic : array_like of bool, shape (2,)
        Periodic boundaries per axis.
    hsm : numpy.ndarray, shape (N,)
        Smoothing lengths per particle.
    kernel_name : str
        Kernel name (e.g., ``"gaussian"``, ``"cubic"``, ``"quintic"``, ``"wendland_c2"``).
    integration_method : str
        Integration method (``"midpoint"``, ``"trapezoidal"``, or ``"simpson"``).
    min_kernel_evaluations : int
        Minimum kernel samples per particle.
    use_openmp : bool
        Enable OpenMP parallelism.
    omp_threads : int
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy)
        Weight sum per cell.
    """
    return _call_backend("isotropic_2d", use_python, positions, quantities, boxsizes, gridnums, periodic, hsm,
                         kernel_name, integration_method, min_kernel_evaluations,
                         use_openmp=use_openmp, omp_threads=omp_threads)


def isotropic_3d(positions, quantities, boxsizes, gridnums, periodic, hsm, kernel_name, integration_method, min_kernel_evaluations, *, use_python=False, use_openmp=True, omp_threads=0):
    """
    Deposit particle quantities onto a 3D grid using an isotropic SPH kernel (C++ backend).

    Parameters
    ----------
    pos : numpy.ndarray, shape (N, 3)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (3,)
        Domain size per axis.
    gridnums : array_like, shape (3,)
        Number of grid cells per axis.
    periodic : array_like of bool, shape (3,)
        Periodic boundaries per axis.
    hsm : numpy.ndarray, shape (N,)
        Smoothing lengths per particle.
    kernel_name : str
        Kernel name (e.g., ``"gaussian"``, ``"cubic"``, ``"quintic"``, ``"wendland_c2"``).
    integration_method : str
        Integration method (``"midpoint"``, ``"trapezoidal"``, or ``"simpson"``).
    min_kernel_evaluations : int
        Minimum kernel samples per particle.
    use_openmp : bool
        Enable OpenMP parallelism.
    omp_threads : int
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, Gz, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy, Gz)
        Weight sum per cell.
    """
    return _call_backend("isotropic_3d", use_python, positions, quantities, boxsizes, gridnums, periodic, hsm,
                         kernel_name, integration_method, min_kernel_evaluations,
                         use_openmp=use_openmp, omp_threads=omp_threads)


def anisotropic_2d(positions, quantities, boxsizes, gridnums, periodic, hmat_eigvecs, hmat_eigvals, kernel_name, integration_method, min_kernel_evaluations, *, use_python=False, use_openmp=True, omp_threads=0):
    """
    Deposit particle quantities onto a 2D grid using an anisotropic SPH kernel (C++ backend).

    Parameters
    ----------
    pos : numpy.ndarray, shape (N, 2)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (2,)
        Domain size per axis.
    gridnums : array_like, shape (2,)
        Number of grid cells per axis.
    periodic : array_like of bool, shape (2,)
        Periodic boundaries per axis.
    hmat_eigvecs : numpy.ndarray, shape (N, 2, 2)
        Eigenvectors of the smoothing tensor per particle.
    hmat_eigvals : numpy.ndarray, shape (N, 2)
        Eigenvalues of the smoothing tensor per particle.
    kernel_name : str
        Kernel name (e.g., ``"gaussian"``, ``"cubic"``, ``"quintic"``, ``"wendland_c2"``).
    integration_method : str
        Integration method (``"midpoint"``, ``"trapezoidal"``, or ``"simpson"``).
    min_kernel_evaluations : int
        Minimum kernel samples per particle.
    use_openmp : bool
        Enable OpenMP parallelism.
    omp_threads : int
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy)
        Weight sum per cell.
    """
    return _call_backend("anisotropic_2d", use_python, positions, quantities, boxsizes, gridnums, periodic,
                         hmat_eigvecs, hmat_eigvals, kernel_name, integration_method, min_kernel_evaluations,
                         use_openmp=use_openmp, omp_threads=omp_threads)


def anisotropic_3d(positions, quantities, boxsizes, gridnums, periodic, hmat_eigvecs, hmat_eigvals, kernel_name, integration_method, min_kernel_evaluations, *, use_python=False, use_openmp=True, omp_threads=0):
    """
    Deposit particle quantities onto a 3D grid using an anisotropic SPH kernel (C++ backend).

    Parameters
    ----------
    pos : numpy.ndarray, shape (N, 3)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (3,)
        Domain size per axis.
    gridnums : array_like, shape (3,)
        Number of grid cells per axis.
    periodic : array_like of bool, shape (3,)
        Periodic boundaries per axis.
    hmat_eigvecs : numpy.ndarray, shape (N, 3, 3)
        Eigenvectors of the smoothing tensor per particle.
    hmat_eigvals : numpy.ndarray, shape (N, 3)
        Eigenvalues of the smoothing tensor per particle.
    kernel_name : str
        Kernel name (e.g., ``"gaussian"``, ``"cubic"``, ``"quintic"``, ``"wendland_c2"``).
    integration_method : str
        Integration method (``"midpoint"``, ``"trapezoidal"``, or ``"simpson"``).
    min_kernel_evaluations : int
        Minimum kernel samples per particle.
    use_openmp : bool
        Enable OpenMP parallelism.
    omp_threads : int
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, Gz, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy, Gz)
        Weight sum per cell.
    """
    return _call_backend("anisotropic_3d", use_python, positions, quantities, boxsizes, gridnums, periodic,
                         hmat_eigvecs, hmat_eigvals, kernel_name, integration_method, min_kernel_evaluations,
                         use_openmp=use_openmp, omp_threads=omp_threads)
