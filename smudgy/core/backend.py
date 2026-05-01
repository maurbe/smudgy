"""Backend-agnostic deposition function wrappers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy.typing as npt

from . import _cpp_functions as _cpp_backend
from . import _py_functions as _py_backend

_PYTHON_UNSUPPORTED = {
    "tophat_2d_adaptive",
    "tophat_3d_adaptive",
    "tsc_2d",
    "tsc_3d",
    "tsc_2d_adaptive",
    "tsc_3d_adaptive",
    "separable_2d",
    "separable_3d",
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
    use_openmp: bool,
    omp_threads: int,
    positions: npt.ArrayLike,
    quantities: npt.ArrayLike,
    particle_weights: npt.ArrayLike,
    boxsizes: Sequence[float],
    gridnums: Sequence[int],
    periodic: bool,
    kernel_name: str,
    smoothing_lengths: npt.ArrayLike | None,
    h_vecs: npt.ArrayLike | None,
    h_vals: npt.ArrayLike | None,
    integration_method: str,
    num_kernel_evaluations_per_axis: int,
    eta_crit: float,
) -> Any:

    if use_python and func_name in ["separable_2d", "separable_3d"]:
        kernel_name_raw = kernel_name
        kernel_name = kernel_name_raw.replace("_separable", "")

        if kernel_name in ["ngp", "tophat"]:
            func_name = f"{kernel_name}_{func_name[-2:]}"

    if use_python and func_name in _PYTHON_UNSUPPORTED:
        raise NotImplementedError(
            f"Python backend does not implement '{func_name}'. Set use_python=False to use the C++ backend."
        )

    if use_python:
        kwargs_ordered = [
            positions,
            quantities,
            particle_weights,
            boxsizes,
            gridnums,
            periodic,
        ]
    else:
        # prepare the correct arguments for the selected backend function
        global_params = [use_openmp, omp_threads, boxsizes, gridnums, periodic]
        shared_params = [positions, quantities, particle_weights]
        kernel_params = [integration_method, kernel_name]
        aliasing_params = [num_kernel_evaluations_per_axis, eta_crit]

        ngp_params = global_params + shared_params
        sep_params = global_params + shared_params + [smoothing_lengths] + kernel_params
        iso_params = (
            global_params
            + shared_params
            + [smoothing_lengths]
            + kernel_params
            + aliasing_params
        )
        aso_params = (
            global_params
            + shared_params
            + [h_vecs, h_vals]
            + kernel_params
            + aliasing_params
        )

        if func_name.startswith("ngp"):
            kwargs_ordered = ngp_params
        elif func_name.startswith("separable"):
            kwargs_ordered = sep_params
        elif func_name.startswith("isotropic"):
            kwargs_ordered = iso_params
        elif func_name.startswith("anisotropic"):
            kwargs_ordered = aso_params
        else:
            raise ValueError(
                f"Unknown function name '{func_name}' for cpp backend call."
            )

    print(f"Calling : {func_name}")
    backend = _select_backend(use_python)
    backend_func_name = f"_{func_name}"
    func: Callable[..., Any] = getattr(backend, backend_func_name)
    return func(*kwargs_ordered)
