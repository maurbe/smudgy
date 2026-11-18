import quadrants as ti

_initialized = False
_current_config = None

_ARCH_MAP = {
    "cpu": ti.cpu,
    "gpu": ti.gpu,
    "cuda": ti.cuda,
    "vulkan": ti.vulkan,
    "metal": ti.metal,
}


def init(arch=None, force=False, **kwargs):
    """Initialize the Taichi runtime used by smudgy (safe to call repeatedly).

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
        Further keyword arguments passed to ``ti.init``.
    """
    global _initialized, _current_config

    if arch is None:
        if _initialized and not force:
            return
        arch = "gpu"

    if arch not in _ARCH_MAP:
        raise ValueError(
            f"Unknown Taichi arch: {arch!r}. Expected one of {sorted(_ARCH_MAP)}"
        )

    config = {"arch": arch, **kwargs}
    if _initialized and not force and _current_config == config:
        return

    ti.init(arch=_ARCH_MAP[arch], **kwargs)
    _initialized = True
    _current_config = config


def current_config():
    """Return the configuration the Taichi runtime was last initialized with."""
    return _current_config
