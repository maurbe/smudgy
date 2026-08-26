"""Execution module for smudgy, providing parallelized dispatching of computational functions across MPI ranks."""

import numpy as np
from mpi4py import MPI

from . import backend as backend_module

REDUCTION = {
    "compute_hsml": "gather",
    "compute_hmat": "gather",
    "compute_density": "gather",
    "project_2d": "gather",
    "interpolate": "gather",
    "deposit": "reduce_sum",
}


def _balanced_counts(n, size):
    """Per-rank item counts for a contiguous, balanced split of n items.

    Every rank's count differs by at most 1 (the first `n % size` ranks get
    one extra item). Shared by `_local_slice` (index-range chunking of
    already-replicated data) and `decomposition.hilbert_partition_and_scatter`
    (chunking a Hilbert-sorted particle array before `_scatterv_rows`), so
    both use exactly one definition of "balanced".
    """
    counts = np.full(size, n // size, dtype=np.int64)
    counts[: n % size] += 1
    return counts


def _local_slice(n, rank, size):
    """Contiguous, balanced [start, stop) range for this rank.

    Callers (in `pointcloud.py`) use this to slice down to their local rows
    *before* doing any expensive per-row gather (e.g. neighbor-index fancy
    indexing), rather than materializing a full-size array on every rank
    and chunking it afterward -- the latter makes that gather cost O(N) per
    rank instead of O(N/size), which dominates wall time at high rank counts.
    """
    counts = _balanced_counts(n, size)
    starts = np.concatenate(([0], np.cumsum(counts)))[:-1]
    return int(starts[rank]), int(starts[rank] + counts[rank])


def _gather(comm, local_result):
    """Stack per-rank results back in rank order (matches _local_slice)."""
    if isinstance(local_result, tuple):
        gathered = [comm.allgather(part) for part in local_result]
        return tuple(np.concatenate(parts, axis=0) for parts in gathered)
    return np.concatenate(comm.allgather(local_result), axis=0)


def _reduce_sum(comm, local_result):
    """Elementwise-sum each rank's full-size output (e.g. deposit grids)."""
    if isinstance(local_result, tuple):
        return tuple(comm.allreduce(part, op=MPI.SUM) for part in local_result)
    return comm.allreduce(local_result, op=MPI.SUM)


def _bcast(comm, obj, root=0):
    """Broadcast a (possibly None-on-non-root) Python object from root.

    Pickle-based, matching the allgather/allreduce style already used in
    this module. Only safe for small objects (scalars, tiny arrays): the
    pickled payload's byte count is passed to the underlying MPI_Bcast as
    a 32-bit int, so this silently breaks (MPI_ERR_ARG) for payloads
    anywhere near 2GB. Use `_bcast_array` for large per-particle arrays.
    """
    return comm.bcast(obj, root=root)


def _bcast_array(comm, arr, root=0):
    """Broadcast a numpy array via a raw buffer Bcast, not pickle.

    Avoids mpi4py's ~2GB pickle-payload ceiling (MPI_Bcast's count param
    is a 32-bit int; pickle-bcast counts raw bytes, buffer-bcast counts
    elements in the array's own dtype) -- large per-particle arrays
    (positions, nn_dists, nn_inds, ...) can exceed 2GB well before hitting
    any realistic element-count limit. `arr` must be non-None on `root`;
    shape/dtype are sent via a small preliminary pickle bcast so non-root
    ranks can allocate a matching receive buffer.
    """
    rank = comm.Get_rank()
    shape, dtype = comm.bcast(
        (arr.shape, arr.dtype) if rank == root else None, root=root
    )
    if rank != root:
        arr = np.empty(shape, dtype=dtype)
    comm.Bcast(arr, root=root)
    return arr


def _scatterv_rows(comm, arr, counts, root=0):
    """Scatter the leading-axis rows of an array to each rank via a raw-buffer Scatterv.

    Counterpart to `_bcast_array` for row-*unequal* distribution -- needed
    because a Hilbert-sorted, count-balanced chunk (see
    `decomposition.hilbert_partition_and_scatter`) isn't equal-sized across
    ranks when N isn't divisible by `size`. `arr` must be non-None, full
    leading-axis length, and C-contiguous on `root`, with rows already
    ordered so rows [0, counts[0]) go to rank 0, [counts[0], counts[0]+counts[1])
    to rank 1, etc.; ignored (may be None) elsewhere. `counts` (shape
    (size,)) must already be identical on every rank (e.g. via a preceding
    small `comm.bcast`, as `hilbert_partition_and_scatter` does). Avoids
    mpi4py's pickle ceiling the same way `_bcast_array` does.

    Returns
    -------
    np.ndarray, shape (counts[rank], *arr.shape[1:])
    """
    rank = comm.Get_rank()
    row_shape, dtype = comm.bcast(
        (arr.shape[1:], arr.dtype) if rank == root else None, root=root
    )
    counts = np.asarray(counts, dtype=np.int64)
    local_arr = np.empty((int(counts[rank]), *row_shape), dtype=dtype)

    row_size = int(np.prod(row_shape, dtype=np.int64)) if row_shape else 1
    send = None
    if rank == root:
        sendcounts = counts * row_size
        displs = np.concatenate(([0], np.cumsum(sendcounts)[:-1]))
        send = [np.ascontiguousarray(arr), (sendcounts, displs)]
    comm.Scatterv(send, local_arr, root=root)
    return local_arr


def _dispatch(func: str, *, backend: str, reduce: bool = True, **kwargs):
    """Run `func` on this rank's (already-local) kwargs, then recombine.

    Callers are responsible for passing already-rank-local arrays (sliced
    via `_local_slice` before any expensive per-row gather) -- this function
    no longer scatters full-size input itself. `reduce=False` skips the
    cross-rank recombination and returns this rank's local result as-is; use
    it for an intermediate dispatch whose output is immediately consumed
    locally by a further dispatch call (e.g. `project_2d` inside
    `PointCloud._prepare_deposition_smoothing`) rather than surfaced to the
    caller.
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    local_result = backend_module._dispatch(func, backend=backend, **kwargs)

    if size == 1 or not reduce:
        return local_result

    if func not in REDUCTION:
        raise ValueError(f"No MPI reduction strategy registered for '{func}'")

    reduction = REDUCTION[func]
    if reduction == "gather":
        return _gather(comm, local_result)
    if reduction == "reduce_sum":
        return _reduce_sum(comm, local_result)
    raise ValueError(f"Unknown reduction strategy '{reduction}' for '{func}'")
