import taichi as ti

_initialized = False
_current_config = None

_ARCH_MAP = {
    "cpu": ti.cpu,
    "gpu": ti.gpu,
    "cuda": ti.cuda,
    "vulkan": ti.vulkan,
    "metal": ti.metal,
}


def init(arch="gpu", force=False, **kwargs):
    """Safe to call repeatedly: no-op if already initialized, unless
    force=True, which tears down and reconfigures the runtime -- e.g.
    to compare backends interactively in a notebook.
    """
    global _initialized, _current_config

    arch = "gpu" if arch is None else arch
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
    return _current_config
