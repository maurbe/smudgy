import warnings

import quadrants as qd

_initialized = False
_current_config = None

_ARCH_MAP = {
    "cpu": qd.cpu,
    "gpu": qd.gpu,
    "cuda": qd.cuda,
    "vulkan": qd.vulkan,
    "metal": qd.metal,
}


def init(arch=None, force=False, **kwargs):
    """Initialize the Quadrants runtime used by smudgy (safe to call repeatedly).

    Parameters
    ----------
    arch : str, optional
        One of ``"cpu"``, ``"gpu"``, ``"cuda"``, ``"vulkan"``, ``"metal"``. If
        None, the already running runtime is kept, or ``"gpu"`` is used on the
        first call.
    force : bool, default False
        Re-initialize the runtime even if it is already running, e.g. to switch
        the architecture. Switching mid-process recompiles all kernels.
    **kwargs
        Further keyword arguments passed to ``qd.init``.
    """
    global _initialized, _current_config

    if arch is None:
        if _initialized and not force:
            if kwargs:
                warnings.warn(
                    f"Quadrants is already initialized ({_current_config}); "
                    f"ignoring {sorted(kwargs)}. Pass arch=... or force=True "
                    "to re-initialize.",
                    stacklevel=2,
                )
            return
        arch = "gpu"

    if arch not in _ARCH_MAP:
        raise ValueError(
            f"Unknown Quadrants arch: {arch!r}. Expected one of {sorted(_ARCH_MAP)}"
        )

    config = {"arch": arch, **kwargs}
    if _initialized and not force and _current_config == config:
        return

    qd.init(arch=_ARCH_MAP[arch], **kwargs)
    _initialized = True
    _current_config = config


def ensure_initialized(arch="cpu"):
    """Initialize with `arch` only if no runtime is running yet.

    For helpers that just need *a* runtime: unlike ``init(arch=...)`` this
    never re-initializes (and so never discards) a runtime the user already
    started, e.g. on the GPU.
    """
    if not _initialized:
        init(arch=arch)


def current_config():
    """Return the configuration the Quadrants runtime was last initialized with."""
    return _current_config
