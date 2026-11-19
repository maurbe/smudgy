"""Backend registry: which module of each backend implements which routine."""

from importlib import import_module
from typing import NamedTuple

BACKENDS = ("numpy", "quadrants")


class Routine(NamedTuple):
    module: str  # module inside backend/<name>/
    function: str


ROUTINES = {
    "deposit": Routine("deposition", "deposit"),
    "interpolate": Routine("interpolation", "interpolate"),
    "compute_hsml": Routine("smoothing", "compute_hsml"),
    "compute_hmat": Routine("smoothing", "compute_hmat"),
    "compute_density": Routine("interpolation", "density"),
    "project_2d": Routine("tensor_utils", "project_2d"),
}


def check_backend(backend):
    """Raise unless `backend` is one of :data:`BACKENDS`."""
    if not isinstance(backend, str):
        raise TypeError(f"'backend' must be a string, got {type(backend).__name__}.")
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend!r}. Expected one of {BACKENDS}.")


def get_routine(routine):
    """Return the :class:`Routine` entry for `routine`."""
    try:
        return ROUTINES[routine]
    except KeyError:
        raise ValueError(
            f"Unknown backend routine: {routine!r}. Available: {sorted(ROUTINES)}"
        ) from None


def _dispatch(routine: str, *args, backend: str, **kwargs):
    """Call a routine from the selected backend without importing both backends."""
    check_backend(backend)
    entry = get_routine(routine)
    module = import_module(f".{backend}.{entry.module}", package=__name__)
    return getattr(module, entry.function)(*args, **kwargs)
