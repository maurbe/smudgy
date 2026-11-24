"""Hilbert keys and their sort for the domain decomposition, in Quadrants.

The numpy reference (:mod:`smudgy.decomposition`) builds the keys with one
vectorized pass per bit and axis, creating full-size temporaries each time,
and sorts them with a stable ``np.argsort``. Both run on the root rank only,
before the particles are scattered, so for large clouds they dominated the
set-up time of a multi-rank run. Here every particle's key is computed by one
thread, and the keys are sorted by Quadrants' radix sort.

Both functions return exactly what the numpy reference returns: the keys bit
for bit, and the permutation of a stable sort (an LSB radix sort is stable).
Only 32-bit integer arithmetic is used in the key kernel (the 63/64-bit key
is written as two 32-bit halves), so it runs on every arch, Metal included.
"""

import warnings

import numpy as np
import quadrants as qd
from quadrants.lang import impl

# Points per launch of the key kernel: keeps every ndarray far below the int32
# element (and the 4 GiB SPIR-V byte) indexing limit, whatever the total N.
_ENCODE_CHUNK = 1 << 24

# Radix-sort scan depth: handles counts up to 256**4 = 2**32 (> any int32 n).
_SORT_DEPTH = 4

# the radix sort needs GPU block-level primitives (qd.block.sync etc.)
_CPU_ARCHS = (qd.x64, qd.arm64)

_KEY_KERNELS = {}
_SORT_KERNEL = None
# per arch: False once the radix sort failed there (numpy is used from then on)
_sort_available = {}


def _key_kernel(dim: int, bits: int):
    """Kernel writing the Hilbert key of each row of ``coords`` as (hi, lo) u32 halves.

    Skilling (2004): AxesToTranspose, then the transpose interleaved MSB-first
    (bit ``b`` of axis ``a`` lands at key bit ``b * dim + dim - 1 - a``),
    exactly as :func:`smudgy.decomposition._axes_to_hilbert_transpose` and
    :func:`smudgy.decomposition._transpose_to_index`.
    """
    key = (dim, bits)
    if key in _KEY_KERNELS:
        return _KEY_KERNELS[key]

    @qd.kernel
    def kernel(
        coords: qd.types.ndarray(dtype=qd.u32, ndim=2),
        hi: qd.types.ndarray(dtype=qd.u32, ndim=1),
        lo: qd.types.ndarray(dtype=qd.u32, ndim=1),
    ):
        for i in range(coords.shape[0]):
            x = qd.Vector([coords[i, a] for a in qd.static(range(dim))], dt=qd.u32)
            # inverse undo
            for e in qd.static(range(bits - 1, 0, -1)):
                for a in qd.static(range(dim)):
                    if x[a] & qd.u32(1 << e):
                        x[0] ^= qd.u32((1 << e) - 1)
                    else:
                        t = (x[0] ^ x[a]) & qd.u32((1 << e) - 1)
                        x[0] ^= t
                        x[a] ^= t
            # Gray encode
            for a in qd.static(range(1, dim)):
                x[a] ^= x[a - 1]
            t = qd.u32(0)
            for e in qd.static(range(bits - 1, 0, -1)):
                if x[dim - 1] & qd.u32(1 << e):
                    t ^= qd.u32((1 << e) - 1)
            # interleave (x ^ t) MSB-first into the key's two 32-bit halves
            h = qd.u32(0)
            l = qd.u32(0)  # noqa: E741
            for b in qd.static(range(bits)):
                for a in qd.static(range(dim)):
                    bit = ((x[a] ^ t) >> qd.u32(b)) & qd.u32(1)
                    if qd.static(b * dim + dim - 1 - a < 32):
                        l |= bit << qd.u32(b * dim + dim - 1 - a)
                    else:
                        h |= bit << qd.u32(b * dim + dim - 1 - a - 32)
            hi[i] = h
            lo[i] = l

    _KEY_KERNELS[key] = kernel
    return kernel


def hilbert_keys(coords: np.ndarray, bits: int) -> np.ndarray:
    """Hilbert keys of quantized coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Per-axis integer coordinates in ``[0, 2**bits)``, shape (N, D) with
        D = 2 or 3, any unsigned integer dtype.
    bits : int
        Bits per axis (``D * bits <= 64`` and ``bits <= 32``).

    Returns
    -------
    np.ndarray
        Hilbert keys, shape (N,), dtype uint64 (bit-identical to
        :func:`smudgy.decomposition.hilbert_encode`).
    """
    n, dim = coords.shape
    if dim not in (2, 3) or bits > 32 or dim * bits > 64:
        raise ValueError(f"unsupported Hilbert key layout: dim={dim}, bits={bits}")
    kernel = _key_kernel(dim, bits)
    keys = np.empty(n, dtype=np.uint64)
    for start in range(0, n, _ENCODE_CHUNK):
        chunk = np.ascontiguousarray(coords[start : start + _ENCODE_CHUNK], dtype=np.uint32)
        hi = np.empty(chunk.shape[0], dtype=np.uint32)
        lo = np.empty(chunk.shape[0], dtype=np.uint32)
        kernel(chunk, hi, lo)
        out = keys[start : start + chunk.shape[0]]
        np.left_shift(hi, np.uint64(32), out=out, dtype=np.uint64)
        out |= lo
    return keys


def _get_sort_kernel():
    global _SORT_KERNEL
    if _SORT_KERNEL is None:
        from quadrants.algorithms import sort

        @qd.kernel
        def kernel(
            keys: qd.types.ndarray(),
            tmp_keys: qd.types.ndarray(),
            values: qd.types.ndarray(),
            tmp_values: qd.types.ndarray(),
            scratch: qd.types.ndarray(),
            n: qd.types.ndarray(),
        ):
            sort(keys, tmp_keys, values, tmp_values, scratch, n, qd.u64, True, 64, _SORT_DEPTH)

        _SORT_KERNEL = kernel
    return _SORT_KERNEL


@qd.kernel
def _iota(values: qd.types.ndarray(dtype=qd.i32, ndim=1)):
    for i in range(values.shape[0]):
        values[i] = i


def _fits_sort(n: int) -> bool:
    """Whether a length-`n` u64 radix sort stays within Quadrants' indexing limits."""
    if n >= 2**31:
        return False
    spirv = impl.current_cfg().arch in (qd.metal, qd.vulkan)
    return not (spirv and 8 * n >= 2**32)


def argsort_keys(keys: np.ndarray) -> np.ndarray:
    """Stable argsort of uint64 keys, by Quadrants' radix sort where available.

    Falls back to numpy (:func:`smudgy.decomposition.stable_argsort`, same
    result) on archs without the radix sort (CPU), for counts beyond its
    indexing limits, or if its output fails a sortedness check.

    Parameters
    ----------
    keys : np.ndarray
        Keys, shape (N,), dtype uint64.

    Returns
    -------
    np.ndarray
        Permutation sorting `keys`, shape (N,), dtype int64.
    """
    from ...decomposition import stable_argsort

    n = keys.shape[0]
    arch = impl.current_cfg().arch
    if arch in _CPU_ARCHS:  # no block-level primitives: numpy, silently
        return stable_argsort(keys)
    if n < 2 or _sort_available.get(arch) is False or not _fits_sort(n):
        return stable_argsort(keys)

    from quadrants.algorithms import sort_scratch_slots

    try:
        d_keys = qd.ndarray(qd.u64, (n,))
        d_keys.from_numpy(np.ascontiguousarray(keys, dtype=np.uint64))
        d_values = qd.ndarray(qd.i32, (n,))
        _iota(d_values)
        d_n = qd.ndarray(qd.i32, ())
        d_n.from_numpy(np.array(n, dtype=np.int32))
        _get_sort_kernel()(
            d_keys,
            qd.ndarray(qd.u64, (n,)),
            d_values,
            qd.ndarray(qd.i32, (n,)),
            qd.ndarray(qd.u32, (sort_scratch_slots(n, _SORT_DEPTH),)),
            d_n,
        )
        order = d_values.to_numpy().astype(np.int64)
    except Exception as err:  # e.g. a GPU arch lacking the block primitives
        _sort_available[arch] = False
        warnings.warn(
            f"Quadrants radix sort unavailable on {arch} ({type(err).__name__}); "
            "the decomposition sorts with numpy instead.",
            stacklevel=2,
        )
        return stable_argsort(keys)

    # O(N) guard: a valid permutation that sorts the keys (the radix sort is
    # marked experimental upstream; never trust it silently)
    sorted_keys = keys[order]
    if not (
        np.all(sorted_keys[1:] >= sorted_keys[:-1])
        and np.all(np.bincount(order, minlength=n) == 1)
    ):
        _sort_available[arch] = False
        warnings.warn(
            f"Quadrants radix sort returned an unsorted result on {arch}; "
            "the decomposition sorts with numpy instead.",
            stacklevel=2,
        )
        return stable_argsort(keys)
    _sort_available[arch] = True
    return order
