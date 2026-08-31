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


def _reduce_sum_to_root(comm, local_result, root=0):
    """Elementwise-sum each rank's full-size output onto `root` only.

    The root-only counterpart to `_reduce_sum` (`comm.reduce` instead of
    `comm.allreduce`) -- cheaper when only `root` needs the combined result
    (e.g. `PointCloud.deposit(..., gather_to_root=True)`). Returns `None` on
    every rank other than `root` (mpi4py's own `Comm.reduce` convention).
    """
    if isinstance(local_result, tuple):
        return tuple(comm.reduce(part, op=MPI.SUM, root=root) for part in local_result)
    return comm.reduce(local_result, op=MPI.SUM, root=root)


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


def _alltoall_counts(comm, send_counts):
    """Alltoall exchange of per-destination row counts.

    Split out from `_alltoallv_rows` so multiple row arrays that share one
    send/recv pattern (e.g. ghost positions + weights + indices, all keyed
    by the same per-particle destination selection) only pay for one small
    Alltoall of counts, not one per array.

    send_counts : np.ndarray[int64], shape (size,)
        This rank's row count destined for each other rank (send_counts[r]
        is how many rows this rank is sending to rank r).

    Returns
    -------
    np.ndarray[int64], shape (size,)
        recv_counts[r] is how many rows this rank will receive from rank r.
    """
    send_counts = np.ascontiguousarray(np.asarray(send_counts, dtype=np.int64))
    recv_counts = np.empty_like(send_counts)
    comm.Alltoall(send_counts, recv_counts)
    return recv_counts


def _alltoallv_rows(comm, send_rows, send_counts, recv_counts):
    """Exchange variable-length row blocks between every pair of ranks via a raw-buffer Alltoallv.

    The genuinely all-to-all counterpart to `_scatterv_rows` (root -> all)
    and `_bcast_array` (root -> all): every rank simultaneously sends a
    (possibly different) row count to every other rank and receives a
    (possibly different) row count from every other rank -- there is no
    root here, so (unlike those two) shape/dtype can't be learned from a
    single authoritative sender. Each rank's own `send_rows` already
    carries its correct row shape/dtype even when it has zero rows to send
    to anyone (e.g. `np.empty((0, dim), dtype=np.float32)`), and that is
    used directly to size this rank's receive buffer -- callers are
    responsible for every rank agreeing on row shape/dtype for a given
    exchange (true by construction when exchanging one conceptual field,
    e.g. "positions", across ranks).

    send_rows : np.ndarray, shape (sum(send_counts), *row_shape)
        This rank's own rows, grouped by destination rank in rank order
        (rows [0, send_counts[0]) go to rank 0, etc.).
    send_counts, recv_counts : np.ndarray[int64], shape (size,)
        `recv_counts` must already be known (via `_alltoall_counts`).

    Returns
    -------
    np.ndarray, shape (sum(recv_counts), *row_shape), grouped by SOURCE
    rank in rank order.
    """
    send_counts = np.asarray(send_counts, dtype=np.int64)
    recv_counts = np.asarray(recv_counts, dtype=np.int64)
    row_shape = send_rows.shape[1:]
    dtype = send_rows.dtype

    row_size = int(np.prod(row_shape, dtype=np.int64)) if row_shape else 1
    sendcounts = send_counts * row_size
    senddispls = np.concatenate(([0], np.cumsum(sendcounts)[:-1]))
    recvcounts = recv_counts * row_size
    recvdispls = np.concatenate(([0], np.cumsum(recvcounts)[:-1]))

    recv_rows = np.empty((int(recv_counts.sum()), *row_shape), dtype=dtype)
    comm.Alltoallv(
        [np.ascontiguousarray(send_rows), (sendcounts, senddispls)],
        [recv_rows, (recvcounts, recvdispls)],
    )
    return recv_rows


def _gatherv_rows(comm, local_rows, root=0):
    """Gather the leading-axis rows of an array from every rank onto `root`
    only, via a raw-buffer Gatherv.

    The many-to-one counterpart to `_scatterv_rows` (root -> all) and
    `_alltoallv_rows` (all -> all) -- avoids mpi4py's ~2GB pickle-payload
    ceiling the same way those do, which matters here since this is meant to
    be the *efficient* alternative to `_gather`'s `allgather` for exactly the
    class of data that ceiling affects (large per-particle arrays). Every
    rank's own `local_rows` must share the same row shape/dtype (true by
    construction when gathering one conceptual array, e.g. a just-computed
    density). Row counts (one int per rank) are exchanged via a separate,
    tiny pickle-based `comm.gather` first -- consistent with how
    `hilbert_partition_and_scatter`/`route_query_positions` already size
    their own Scatterv the same way.

    Returns
    -------
    np.ndarray, shape (sum of every rank's row count, *row_shape), rows
    grouped by source rank in rank order, on `root`; `None` elsewhere.
    """
    rank = comm.Get_rank()
    local_rows = np.ascontiguousarray(local_rows)
    row_shape, dtype = local_rows.shape[1:], local_rows.dtype
    row_size = int(np.prod(row_shape, dtype=np.int64)) if row_shape else 1

    counts = comm.gather(local_rows.shape[0], root=root)

    recv_spec = None
    recv_buf = None
    if rank == root:
        counts = np.asarray(counts, dtype=np.int64)
        recvcounts = counts * row_size
        displs = np.concatenate(([0], np.cumsum(recvcounts)[:-1]))
        recv_buf = np.empty((int(counts.sum()), *row_shape), dtype=dtype)
        recv_spec = [recv_buf, (recvcounts, displs)]

    comm.Gatherv(local_rows, recv_spec, root=root)
    return recv_buf if rank == root else None


def _gather_to_root(comm, global_index, local_array, n_total, root=0):
    """Gather a local, `global_index`-ordered array back onto `root` only,
    reassembled into original order.

    The root-only counterpart to `_gather` (which ships the full result to
    *every* rank via `allgather` instead): what a caller using the
    local+ghost paths (`PointCloud.decompose`/`find_neighbors`, or
    `interpolate(query_positions=...)`) should use to turn a local-sized
    result (ordered by `decomposition.local_global_indices` or
    `query_routing.local_global_indices`) into a normal, original-order
    array without paying to replicate it onto every rank.

    `global_index` and `local_array` must share the same leading-axis length
    (this rank's own local row count). `n_total` -- already known
    identically on every rank at every real call site
    (`decomposition.counts.sum()`/`positions.shape[0]` for particles,
    `query_routing.counts.sum()` for query positions) -- sizes root's
    reassembled output; not re-derived here.

    Returns
    -------
    np.ndarray, shape (n_total, *local_array.shape[1:]), on `root`; `None`
    on every other rank.
    """
    rank = comm.Get_rank()
    recv_global = _gatherv_rows(comm, np.asarray(global_index, dtype=np.int64), root=root)
    recv_array = _gatherv_rows(comm, local_array, root=root)
    if rank != root:
        return None
    full = np.empty((n_total, *local_array.shape[1:]), dtype=local_array.dtype)
    full[recv_global] = recv_array
    return full


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
