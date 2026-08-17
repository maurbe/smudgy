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


def init(arch=None, force=False, **kwargs):
    """Safe to call repeatedly: no-op if already initialized, unless
    force=True, which tears down and reconfigures the runtime -- e.g.
    to compare backends interactively in a notebook.

    arch=None means "no preference": reuse whatever is already active if
    the runtime has been initialized before, or fall back to "gpu" on the
    very first call in the process. This matters because switching a live
    Taichi runtime's arch mid-process (e.g. gpu -> cpu -> gpu) has been
    observed to destabilize previously-compiled @ti.kernel functions --
    callers that don't care about arch should never force that switch just
    by omitting the argument.
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
    return _current_config
