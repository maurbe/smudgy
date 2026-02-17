"""Backend-agnostic deposition function wrappers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy.typing as npt

from .core import _cpp_functions as _cpp_backend
from .core import _py_functions as _py_backend

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


def _select_backend(use_python: bool) -> Any:
    """Select the backend module (Python or C++).

    Args:
        use_python: If True, use the Python backend; otherwise, use the C++ backend.

    Returns:
        The selected backend module.

    """
    return _py_backend if use_python else _cpp_backend


def _call_backend(
    func_name: str,
    use_python: bool,
    *args: Any,
    use_openmp: bool,
    omp_threads: int,
) -> Any:
    """Call the selected backend function with the provided arguments.

    Args:
        func_name: Name of the backend function (without leading underscore).
        use_python: If True, use the Python backend; otherwise, use the C++ backend.
        *args: Positional arguments to pass to the backend function.
        use_openmp: Enable OpenMP parallelism (C++ backend only).
        omp_threads: Number of OpenMP threads (C++ backend only).

    Returns:
        The result of the backend function call.

    Raises:
        NotImplementedError: If the Python backend does not implement the requested function.

    """
    if use_python and func_name in _PYTHON_UNSUPPORTED:
        raise NotImplementedError(
            f"Python backend does not implement '{func_name}'. Set use_python=False to use the C++ backend."
        )
    backend = _select_backend(use_python)
    backend_func_name = f"_{func_name}"
    func: Callable[..., Any] = getattr(backend, backend_func_name)
    if use_python:
        return func(*args)
    return func(*args, use_openmp, omp_threads)


def ngp_2d(
    positions: npt.ArrayLike,
    quantities: npt.ArrayLike,
    boxsizes: Sequence[float],
    gridnums: Sequence[int],
    periodic: bool,
    *args: Any,
    use_python: bool = False,
    use_openmp: bool = True,
    omp_threads: int = 0,
):
    """Deposit particle quantities onto a 2D grid using NGP (C++ backend).

    Parameters
    ----------
    positions : numpy.ndarray, shape (N, 2)
        Particle positions, where ``N`` is the number of particles.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (2,)
        Domain size per axis.
    gridnums : array_like, shape (2,)
        Number of grid cells per axis.
    periodic : bool
        Global periodic boundaries.
    *args : Any
        Additional positional arguments (unused).
    use_python : bool, optional
        Use the Python backend if True, else C++ backend.
    use_openmp : bool, optional
        Enable OpenMP parallelism.
    omp_threads : int, optional
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy)
        Weight sum per cell.

    """
    return _call_backend(
        "ngp_2d",
        use_python,
        positions,
        quantities,
        boxsizes,
        gridnums,
        periodic,
        use_openmp=use_openmp,
        omp_threads=omp_threads,
    )


def ngp_3d(
    positions: npt.ArrayLike,
    quantities: npt.ArrayLike,
    boxsizes: Sequence[float],
    gridnums: Sequence[int],
    periodic: bool,
    *args: Any,
    use_python: bool = False,
    use_openmp: bool = True,
    omp_threads: int = 0,
):
    """Deposit particle quantities onto a 3D grid using NGP (C++ backend).

    Parameters
    ----------
    positions : numpy.ndarray, shape (N, 3)
        Particle positions, where ``N`` is the number of particles.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (3,)
        Domain size per axis.
    gridnums : array_like, shape (3,)
        Number of grid cells per axis.
    periodic : bool
        Global periodic boundaries.
    *args : Any
        Additional positional arguments (unused).
    use_python : bool, optional
        Use the Python backend if True, else C++ backend.
    use_openmp : bool, optional
        Enable OpenMP parallelism.
    omp_threads : int, optional
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, Gz, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy, Gz)
        Weight sum per cell.

    """
    return _call_backend(
        "ngp_3d",
        use_python,
        positions,
        quantities,
        boxsizes,
        gridnums,
        periodic,
        use_openmp=use_openmp,
        omp_threads=omp_threads,
    )


# CIC


def cic_2d(
    positions: npt.ArrayLike,
    quantities: npt.ArrayLike,
    boxsizes: Sequence[float],
    gridnums: Sequence[int],
    periodic: bool,
    *args: Any,
    use_python: bool = False,
    use_openmp: bool = True,
    omp_threads: int = 0,
):
    """Deposit particle quantities onto a 2D grid using CIC (C++ backend).

    Parameters
    ----------
    positions : numpy.ndarray, shape (N, 2)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (2,)
        Domain size per axis.
    gridnums : array_like, shape (2,)
        Number of grid cells per axis.
    periodic : bool
        Global periodic boundaries.
    *args : Any
        Additional positional arguments (unused).
    use_python : bool, optional
        Use the Python backend if True, else C++ backend.
    use_openmp : bool, optional
        Enable OpenMP parallelism.
    omp_threads : int, optional
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy)
        Weight sum per cell.

    """
    return _call_backend(
        "cic_2d",
        use_python,
        positions,
        quantities,
        boxsizes,
        gridnums,
        periodic,
        use_openmp=use_openmp,
        omp_threads=omp_threads,
    )


def cic_3d(
    positions: npt.ArrayLike,
    quantities: npt.ArrayLike,
    boxsizes: Sequence[float],
    gridnums: Sequence[int],
    periodic: bool,
    *args: Any,
    use_python: bool = False,
    use_openmp: bool = True,
    omp_threads: int = 0,
):
    """Deposit particle quantities onto a 3D grid using CIC (C++ backend).

    Parameters
    ----------
    positions : numpy.ndarray, shape (N, 3)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (3,)
        Domain size per axis.
    gridnums : array_like, shape (3,)
        Number of grid cells per axis.
    periodic : bool
        Global periodic boundaries.
    *args : Any
        Additional positional arguments (unused).
    use_python : bool, optional
        Use the Python backend if True, else C++ backend.
    use_openmp : bool, optional
        Enable OpenMP parallelism.
    omp_threads : int, optional
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, Gz, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy, Gz)
        Weight sum per cell.

    """
    return _call_backend(
        "cic_3d",
        use_python,
        positions,
        quantities,
        boxsizes,
        gridnums,
        periodic,
        use_openmp=use_openmp,
        omp_threads=omp_threads,
    )


def cic_adaptive_2d(
    positions: npt.ArrayLike,
    quantities: npt.ArrayLike,
    boxsizes: Sequence[float],
    gridnums: Sequence[int],
    periodic: bool,
    pcellsizesHalf: npt.ArrayLike,
    *args: Any,
    use_python: bool = False,
    use_openmp: bool = True,
    omp_threads: int = 0,
):
    """Deposit particle quantities onto a 2D grid using adaptive CIC (C++ backend).

    Parameters
    ----------
    positions : numpy.ndarray, shape (N, 2)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (2,)
        Domain size per axis.
    gridnums : array_like, shape (2,)
        Number of grid cells per axis.
    periodic : bool
        Global periodic boundaries.
    pcellsizesHalf : numpy.ndarray, shape (N, 2)
        Half cell sizes per particle (adaptive support).
    *args : Any
        Additional positional arguments (unused).
    use_python : bool, optional
        Use the Python backend if True, else C++ backend.
    use_openmp : bool, optional
        Enable OpenMP parallelism.
    omp_threads : int, optional
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy)
        Weight sum per cell.

    """
    return _call_backend(
        "cic_2d_adaptive",
        use_python,
        positions,
        quantities,
        boxsizes,
        gridnums,
        periodic,
        pcellsizesHalf,
        use_openmp=use_openmp,
        omp_threads=omp_threads,
    )


def cic_adaptive_3d(
    positions: npt.ArrayLike,
    quantities: npt.ArrayLike,
    boxsizes: Sequence[float],
    gridnums: Sequence[int],
    periodic: bool,
    pcellsizesHalf: npt.ArrayLike,
    *args: Any,
    use_python: bool = False,
    use_openmp: bool = True,
    omp_threads: int = 0,
):
    """Deposit particle quantities onto a 3D grid using adaptive CIC (C++ backend).

    Parameters
    ----------
    positions : numpy.ndarray, shape (N, 3)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (3,)
        Domain size per axis.
    gridnums : array_like, shape (3,)
        Number of grid cells per axis.
    periodic : bool
        Global periodic boundaries.
    pcellsizesHalf : numpy.ndarray, shape (N, 3)
        Half cell sizes per particle (adaptive support).
    *args : Any
        Additional positional arguments (unused).
    use_python : bool, optional
        Use the Python backend if True, else C++ backend.
    use_openmp : bool, optional
        Enable OpenMP parallelism.
    omp_threads : int, optional
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, Gz, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy, Gz)
        Weight sum per cell.

    """
    return _call_backend(
        "cic_3d_adaptive",
        use_python,
        positions,
        quantities,
        boxsizes,
        gridnums,
        periodic,
        pcellsizesHalf,
        use_openmp=use_openmp,
        omp_threads=omp_threads,
    )


# TSC


def tsc_2d(
    positions: npt.ArrayLike,
    quantities: npt.ArrayLike,
    boxsizes: Sequence[float],
    gridnums: Sequence[int],
    periodic: bool,
    *args: Any,
    use_python: bool = False,
    use_openmp: bool = True,
    omp_threads: int = 0,
):
    """Deposit particle quantities onto a 2D grid using TSC (C++ backend).

    Parameters
    ----------
    positions : numpy.ndarray, shape (N, 2)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (2,)
        Domain size per axis.
    gridnums : array_like, shape (2,)
        Number of grid cells per axis.
    periodic : bool
        Global periodic boundaries.
    *args : Any
        Additional positional arguments (unused).
    use_python : bool, optional
        Use the Python backend if True, else C++ backend.
    use_openmp : bool, optional
        Enable OpenMP parallelism.
    omp_threads : int, optional
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy)
        Weight sum per cell.

    """
    return _call_backend(
        "tsc_2d",
        use_python,
        positions,
        quantities,
        boxsizes,
        gridnums,
        periodic,
        use_openmp=use_openmp,
        omp_threads=omp_threads,
    )


def tsc_3d(
    positions: npt.ArrayLike,
    quantities: npt.ArrayLike,
    boxsizes: Sequence[float],
    gridnums: Sequence[int],
    periodic: bool,
    *args: Any,
    use_python: bool = False,
    use_openmp: bool = True,
    omp_threads: int = 0,
):
    """Deposit particle quantities onto a 3D grid using TSC (C++ backend).

    Parameters
    ----------
    positions : numpy.ndarray, shape (N, 3)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (3,)
        Domain size per axis.
    gridnums : array_like, shape (3,)
        Number of grid cells per axis.
    periodic : bool
        Global periodic boundaries.
    *args : Any
        Additional positional arguments (unused).
    use_python : bool, optional
        Use the Python backend if True, else C++ backend.
    use_openmp : bool, optional
        Enable OpenMP parallelism.
    omp_threads : int, optional
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, Gz, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy, Gz)
        Weight sum per cell.

    """
    return _call_backend(
        "tsc_3d",
        use_python,
        positions,
        quantities,
        boxsizes,
        gridnums,
        periodic,
        use_openmp=use_openmp,
        omp_threads=omp_threads,
    )


def tsc_adaptive_2d(
    positions: npt.ArrayLike,
    quantities: npt.ArrayLike,
    boxsizes: Sequence[float],
    gridnums: Sequence[int],
    periodic: bool,
    pcellsizesHalf: npt.ArrayLike,
    *args: Any,
    use_python: bool = False,
    use_openmp: bool = True,
    omp_threads: int = 0,
):
    """Deposit particle quantities onto a 2D grid using adaptive TSC (C++ backend).

    Parameters
    ----------
    positions : numpy.ndarray, shape (N, 2)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (2,)
        Domain size per axis.
    gridnums : array_like, shape (2,)
        Number of grid cells per axis.
    periodic : bool
        Global periodic boundaries.
    pcellsizesHalf : numpy.ndarray, shape (N, 2)
        Half cell sizes per particle (adaptive support).
    *args : Any
        Additional positional arguments (unused).
    use_python : bool, optional
        Use the Python backend if True, else C++ backend.
    use_openmp : bool, optional
        Enable OpenMP parallelism.
    omp_threads : int, optional
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy)
        Weight sum per cell.

    """
    return _call_backend(
        "tsc_2d_adaptive",
        use_python,
        positions,
        quantities,
        boxsizes,
        gridnums,
        periodic,
        pcellsizesHalf,
        use_openmp=use_openmp,
        omp_threads=omp_threads,
    )


def tsc_adaptive_3d(
    positions: npt.ArrayLike,
    quantities: npt.ArrayLike,
    boxsizes: Sequence[float],
    gridnums: Sequence[int],
    periodic: bool,
    pcellsizesHalf: npt.ArrayLike,
    *args: Any,
    use_python: bool = False,
    use_openmp: bool = True,
    omp_threads: int = 0,
):
    """Deposit particle quantities onto a 3D grid using adaptive TSC (C++ backend).

    Parameters
    ----------
    positions : numpy.ndarray, shape (N, 3)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (3,)
        Domain size per axis.
    gridnums : array_like, shape (3,)
        Number of grid cells per axis.
    periodic : bool
        Global periodic boundaries.
    pcellsizesHalf : numpy.ndarray, shape (N, 3)
        Half cell sizes per particle (adaptive support).
    *args : Any
        Additional positional arguments (unused).
    use_python : bool, optional
        Use the Python backend if True, else C++ backend.
    use_openmp : bool, optional
        Enable OpenMP parallelism.
    omp_threads : int, optional
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, Gz, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy, Gz)
        Weight sum per cell.

    """
    return _call_backend(
        "tsc_3d_adaptive",
        use_python,
        positions,
        quantities,
        boxsizes,
        gridnums,
        periodic,
        pcellsizesHalf,
        use_openmp=use_openmp,
        omp_threads=omp_threads,
    )


# SPH kernels


def isotropic_2d(
    positions: npt.ArrayLike,
    quantities: npt.ArrayLike,
    boxsizes: Sequence[float],
    gridnums: Sequence[int],
    periodic: bool,
    hsm: npt.ArrayLike,
    kernel_name: str,
    integration_method: str,
    min_kernel_evaluations: int,
    *args: Any,
    use_python: bool = False,
    use_openmp: bool = True,
    omp_threads: int = 0,
):
    """Deposit particle quantities onto a 2D grid using an isotropic SPH kernel (C++ backend).

    Parameters
    ----------
    positions : numpy.ndarray, shape (N, 2)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (2,)
        Domain size per axis.
    gridnums : array_like, shape (2,)
        Number of grid cells per axis.
    periodic : bool
        Global periodic boundaries.
    hsm : numpy.ndarray, shape (N,)
        Smoothing lengths per particle.
    kernel_name : str
        Kernel name (e.g., ``"gaussian"``, ``"cubic"``, ``"quintic"``, ``"wendland_c2"``).
    integration_method : str
        Integration method (``"midpoint"``, ``"trapezoidal"``, or ``"simpson"``).
    min_kernel_evaluations : int
        Minimum kernel samples per particle.
    *args : Any
        Additional positional arguments (unused).
    use_python : bool, optional
        Use the Python backend if True, else C++ backend.
    use_openmp : bool, optional
        Enable OpenMP parallelism.
    omp_threads : int, optional
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy)
        Weight sum per cell.

    """
    return _call_backend(
        "isotropic_2d",
        use_python,
        positions,
        quantities,
        boxsizes,
        gridnums,
        periodic,
        hsm,
        kernel_name,
        integration_method,
        min_kernel_evaluations,
        use_openmp=use_openmp,
        omp_threads=omp_threads,
    )


def isotropic_3d(
    positions: npt.ArrayLike,
    quantities: npt.ArrayLike,
    boxsizes: Sequence[float],
    gridnums: Sequence[int],
    periodic: bool,
    hsm: npt.ArrayLike,
    kernel_name: str,
    integration_method: str,
    min_kernel_evaluations: int,
    *args: Any,
    use_python: bool = False,
    use_openmp: bool = True,
    omp_threads: int = 0,
):
    """Deposit particle quantities onto a 3D grid using an isotropic SPH kernel (C++ backend).

    Parameters
    ----------
    positions : numpy.ndarray, shape (N, 3)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (3,)
        Domain size per axis.
    gridnums : array_like, shape (3,)
        Number of grid cells per axis.
    periodic : bool
        Global periodic boundaries.
    hsm : numpy.ndarray, shape (N,)
        Smoothing lengths per particle.
    kernel_name : str
        Kernel name (e.g., ``"gaussian"``, ``"cubic"``, ``"quintic"``, ``"wendland_c2"``).
    integration_method : str
        Integration method (``"midpoint"``, ``"trapezoidal"``, or ``"simpson"``).
    min_kernel_evaluations : int
        Minimum kernel samples per particle.
    *args : Any
        Additional positional arguments (unused).
    use_python : bool, optional
        Use the Python backend if True, else C++ backend.
    use_openmp : bool, optional
        Enable OpenMP parallelism.
    omp_threads : int, optional
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, Gz, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy, Gz)
        Weight sum per cell.

    """
    return _call_backend(
        "isotropic_3d",
        use_python,
        positions,
        quantities,
        boxsizes,
        gridnums,
        periodic,
        hsm,
        kernel_name,
        integration_method,
        min_kernel_evaluations,
        use_openmp=use_openmp,
        omp_threads=omp_threads,
    )


def anisotropic_2d(
    positions: npt.ArrayLike,
    quantities: npt.ArrayLike,
    boxsizes: Sequence[float],
    gridnums: Sequence[int],
    periodic: bool,
    hmat_eigvecs: npt.ArrayLike,
    hmat_eigvals: npt.ArrayLike,
    kernel_name: str,
    integration_method: str,
    min_kernel_evaluations: int,
    *args: Any,
    use_python: bool = False,
    use_openmp: bool = True,
    omp_threads: int = 0,
):
    """Deposit particle quantities onto a 2D grid using an anisotropic SPH kernel (C++ backend).

    Parameters
    ----------
    positions : numpy.ndarray, shape (N, 2)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (2,)
        Domain size per axis.
    gridnums : array_like, shape (2,)
        Number of grid cells per axis.
    periodic : bool
        Global periodic boundaries.
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
    *args : Any
        Additional positional arguments (unused).
    use_python : bool, optional
        Use the Python backend if True, else C++ backend.
    use_openmp : bool, optional
        Enable OpenMP parallelism.
    omp_threads : int, optional
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy)
        Weight sum per cell.

    """
    return _call_backend(
        "anisotropic_2d",
        use_python,
        positions,
        quantities,
        boxsizes,
        gridnums,
        periodic,
        hmat_eigvecs,
        hmat_eigvals,
        kernel_name,
        integration_method,
        min_kernel_evaluations,
        use_openmp=use_openmp,
        omp_threads=omp_threads,
    )


def anisotropic_3d(
    positions: npt.ArrayLike,
    quantities: npt.ArrayLike,
    boxsizes: Sequence[float],
    gridnums: Sequence[int],
    periodic: bool,
    hmat_eigvecs: npt.ArrayLike,
    hmat_eigvals: npt.ArrayLike,
    kernel_name: str,
    integration_method: str,
    min_kernel_evaluations: int,
    *args: Any,
    use_python: bool = False,
    use_openmp: bool = True,
    omp_threads: int = 0,
):
    """Deposit particle quantities onto a 3D grid using an anisotropic SPH kernel (C++ backend).

    Parameters
    ----------
    positions : numpy.ndarray, shape (N, 3)
        Particle positions.
    quantities : numpy.ndarray, shape (N, F)
        Per-particle fields to deposit.
    boxsizes : array_like, shape (3,)
        Domain size per axis.
    gridnums : array_like, shape (3,)
        Number of grid cells per axis.
    periodic : bool
        Global periodic boundaries.
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
    *args : Any
        Additional positional arguments (unused).
    use_python : bool, optional
        Use the Python backend if True, else C++ backend.
    use_openmp : bool, optional
        Enable OpenMP parallelism.
    omp_threads : int, optional
        Number of OpenMP threads (0 uses the default).

    Returns
    -------
    fields : numpy.ndarray, shape (Gx, Gy, Gz, F)
        Deposited field values.
    weights : numpy.ndarray, shape (Gx, Gy, Gz)
        Weight sum per cell.

    """
    return _call_backend(
        "anisotropic_3d",
        use_python,
        positions,
        quantities,
        boxsizes,
        gridnums,
        periodic,
        hmat_eigvecs,
        hmat_eigvals,
        kernel_name,
        integration_method,
        min_kernel_evaluations,
        use_openmp=use_openmp,
        omp_threads=omp_threads,
    )
