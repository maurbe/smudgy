from importlib import import_module

_BACKEND_ROUTINES = {
    "deposit": ("deposition", "deposit"),
    "interpolate": ("interpolation", "interpolate"),
    "compute_hsml": ("smoothing", "compute_hsml"),
    "compute_hmat": ("smoothing", "compute_hmat"),
    "compute_density": ("interpolation", "density"),
}


def _check_backend(backend: str):
    if backend not in ("numpy", "taichi"):
        raise ValueError(f"Unknown backend: {backend!r}. Expected 'numpy' or 'taichi'.")


def dispatch(routine: str, *args, backend: str, **kwargs):
    """Call a routine from the selected backend without importing both backends."""
    _check_backend(backend)
    try:
        module_name, function_name = _BACKEND_ROUTINES[routine]
    except KeyError:
        raise ValueError(
            f"Unknown backend routine: {routine!r}. "
            f"Available: {sorted(_BACKEND_ROUTINES)}"
        ) from None

    module = import_module(f".{backend}.{module_name}", package=__name__)
    return getattr(module, function_name)(*args, **kwargs)
