import os
import warnings

import quadrants as qd

from .. import _node

_initialized = False
_current_config = None

_ARCH_MAP = {
    "cpu": qd.cpu,
    "gpu": qd.gpu,
    "cuda": qd.cuda,
    "vulkan": qd.vulkan,
    "metal": qd.metal,
}


def init(arch=None, force=False, device="auto", **kwargs):
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
    device : "auto", int or None, default "auto"
        GPU for this process (CUDA/Vulkan, not macOS). ``"auto"`` gives each
        MPI rank on a node its own GPU (node-local rank modulo the GPU count);
        None leaves the choice to the driver (device 0). Must be chosen before
        the first CUDA runtime in the process starts.
    **kwargs
        Further keyword arguments passed to ``qd.init``. Unless
        ``cpu_max_num_threads`` is given, ranks sharing a node split its cores.
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

    qd_kwargs = dict(kwargs)
    threads = _node.cpu_threads()
    if threads is not None:
        qd_kwargs.setdefault("cpu_max_num_threads", threads)
    os.environ.update(_node.gpu_env(arch, device))
    qd.init(arch=_ARCH_MAP[arch], **qd_kwargs)
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


# Quadrants indexes ndarrays with int32: element offsets wrap above 2**31 - 1,
# and the SPIR-V backends (Metal, Vulkan) compute 32-bit *byte* offsets, so
# 4-byte arrays wrap above 2**30 elements (4 GiB) there -- silently, with
# wrong results (measured on Metal for nn_inds of shape (19683000, 64)).
_MAX_ELEMENTS = 2**31 - 1
_MAX_BYTES_SPIRV = 2**32 - 1


def check_index_limits(**arrays):
    """Raise ValueError if an array (or shape tuple) is too large to index.

    Call before launching a kernel, with the largest arrays it reads or writes,
    e.g. ``check_index_limits(nn_inds=nn_inds, grid=(*gridnums, F))``.
    """
    from quadrants.lang import impl

    spirv = _initialized and impl.current_cfg().arch in (qd.metal, qd.vulkan)
    for name, a in arrays.items():
        if isinstance(a, tuple):
            size, nbytes = 1, 4
            for s in a:
                size *= int(s)
            nbytes *= size
        else:
            size, nbytes = a.size, a.nbytes
        if size > _MAX_ELEMENTS or (spirv and nbytes > _MAX_BYTES_SPIRV):
            limit = "4 GiB (Metal/Vulkan)" if spirv else f"{_MAX_ELEMENTS} elements"
            raise ValueError(
                f"'{name}' has {size} elements ({nbytes} bytes), beyond Quadrants' "
                f"int32 indexing limit of {limit}. Reduce N, the neighbour count "
                "or the grid size, or split the work across MPI ranks."
            )
