"""MPI communication helpers and the backend call used by :class:`PointCloud`.

All array exchanges use mpi4py's buffer interface (no pickling), so they are
not limited by pickle's 2 GB payload size. Only small metadata goes through
the pickle-based collectives (``comm.bcast``/``comm.gather``).
"""

import numpy as np
from mpi4py import MPI

from . import backend as backend_module

# Elements per MPI call: buffer collectives take an int32 count, so larger
# arrays are reduced in chunks.
_MAX_COUNT = 2**30


def _balanced_counts(n, size):
    """Per-rank counts of a contiguous split of `n` items (differing by <= 1)."""
    counts = np.full(size, n // size, dtype=np.int64)
    counts[: n % size] += 1
    return counts


def _displacements(counts):
    """Exclusive prefix sum of `counts` (MPI displacement array)."""
    return np.concatenate(([0], np.cumsum(counts)[:-1]))


def _row_size(row_shape):
    """Elements per row of an array with trailing shape `row_shape`."""
    return int(np.prod(row_shape, dtype=np.int64)) if row_shape else 1


# =============================================================================
# Element-wise sums (deposition grids)
# =============================================================================
def _apply_parts(fn, local_result):
    """Apply `fn` to every array of a tuple result, or to a single array."""
    if isinstance(local_result, tuple):
        return tuple(fn(part) for part in local_result)
    return fn(local_result)


def _writable_contiguous(arr):
    """``(arr, buf)`` with `buf` a writable C-contiguous view of `arr`'s data.

    Deposition grids come back as ``np.moveaxis(grid, -1, 0)`` views of a
    C-contiguous ``(*gridnums, F)`` array; reducing that base in place avoids a
    copy. If no such view exists, `arr` is replaced by a contiguous copy.
    """
    if arr.flags.c_contiguous and arr.flags.writeable:
        return arr, arr
    if arr.ndim > 1:
        base = np.moveaxis(arr, 0, -1)
        if base.flags.c_contiguous and base.flags.writeable:
            return arr, base
    arr = np.array(arr, order="C")
    return arr, arr


def _sum_over_ranks(comm, arr, root=None):
    """Element-wise sum of `arr` over all ranks, in place.

    With ``root=None`` every rank receives the sum (Allreduce); otherwise only
    `root` does (Reduce) and every other rank gets None.
    """
    arr, buf = _writable_contiguous(np.asarray(arr))
    flat = buf.reshape(-1)  # view: buf is C-contiguous
    is_root = root is None or comm.Get_rank() == root
    for start in range(0, flat.size, _MAX_COUNT):
        chunk = flat[start : start + _MAX_COUNT]
        if root is None:
            comm.Allreduce(MPI.IN_PLACE, chunk, op=MPI.SUM)
        elif is_root:
            comm.Reduce(MPI.IN_PLACE, chunk, op=MPI.SUM, root=root)
        else:
            comm.Reduce(chunk, None, op=MPI.SUM, root=root)
    return arr if is_root else None


def _reduce_sum(comm, local_result):
    """Sum each array of `local_result` over all ranks, onto every rank (in place)."""
    return _apply_parts(lambda part: _sum_over_ranks(comm, part), local_result)


def _reduce_sum_to_root(comm, local_result, root=0):
    """Sum each array of `local_result` over all ranks onto `root` (None elsewhere)."""
    return _apply_parts(lambda part: _sum_over_ranks(comm, part, root=root), local_result)


# =============================================================================
# Row distribution
# =============================================================================
def _bcast(comm, obj, root=0):
    """Broadcast a small Python object from `root` (pickle-based)."""
    return comm.bcast(obj, root=root)


def _scatterv_rows(comm, arr, counts, root=0):
    """Scatter consecutive row blocks of `arr` from `root`: ``counts[r]`` rows to rank r.

    `arr` is only read on `root`; `counts` must be identical on every rank.
    Returns this rank's ``(counts[rank], *arr.shape[1:])`` block.
    """
    rank = comm.Get_rank()
    row_shape, dtype = comm.bcast(
        (arr.shape[1:], arr.dtype) if rank == root else None, root=root
    )
    counts = np.asarray(counts, dtype=np.int64)
    local_arr = np.empty((int(counts[rank]), *row_shape), dtype=dtype)

    send = None
    if rank == root:
        sendcounts = counts * _row_size(row_shape)
        send = [np.ascontiguousarray(arr), (sendcounts, _displacements(sendcounts))]
    comm.Scatterv(send, local_arr, root=root)
    return local_arr


def _alltoall_counts(comm, send_counts):
    """Exchange per-destination row counts: returns the rows to receive from each rank."""
    send_counts = np.ascontiguousarray(np.asarray(send_counts, dtype=np.int64))
    recv_counts = np.empty_like(send_counts)
    comm.Alltoall(send_counts, recv_counts)
    return recv_counts


def _alltoallv_rows(comm, send_rows, send_counts, recv_counts):
    """Exchange row blocks between all pairs of ranks.

    `send_rows` holds this rank's rows grouped by destination rank (rank
    order), ``send_counts[r]`` rows for rank r; `recv_counts` comes from
    :func:`_alltoall_counts`. Every rank must pass the same row shape and
    dtype, also when it sends no rows. Returns the received rows, grouped by
    source rank.
    """
    send_counts = np.asarray(send_counts, dtype=np.int64)
    recv_counts = np.asarray(recv_counts, dtype=np.int64)
    row_shape = send_rows.shape[1:]
    row_size = _row_size(row_shape)
    sendcounts = send_counts * row_size
    recvcounts = recv_counts * row_size

    recv_rows = np.empty((int(recv_counts.sum()), *row_shape), dtype=send_rows.dtype)
    comm.Alltoallv(
        [np.ascontiguousarray(send_rows), (sendcounts, _displacements(sendcounts))],
        [recv_rows, (recvcounts, _displacements(recvcounts))],
    )
    return recv_rows


def _gatherv_rows(comm, local_rows, root=0):
    """Concatenate every rank's rows (rank order) on `root`; None elsewhere."""
    rank = comm.Get_rank()
    local_rows = np.ascontiguousarray(local_rows)
    row_shape = local_rows.shape[1:]
    counts = comm.gather(local_rows.shape[0], root=root)

    recv_spec = recv_buf = None
    if rank == root:
        counts = np.asarray(counts, dtype=np.int64)
        recvcounts = counts * _row_size(row_shape)
        recv_buf = np.empty((int(counts.sum()), *row_shape), dtype=local_rows.dtype)
        recv_spec = [recv_buf, (recvcounts, _displacements(recvcounts))]

    comm.Gatherv(local_rows, recv_spec, root=root)
    return recv_buf


def _gather_to_root(comm, global_index, local_array, n_total, root=0):
    """Reassemble a distributed array on `root` in its original order.

    Row ``i`` of `local_array` goes to row ``global_index[i]`` of the
    ``(n_total, ...)`` result, which is returned on `root` (None elsewhere).
    """
    recv_global = _gatherv_rows(comm, np.asarray(global_index, dtype=np.int64), root=root)
    recv_array = _gatherv_rows(comm, local_array, root=root)
    if comm.Get_rank() != root:
        return None
    full = np.empty((n_total, *local_array.shape[1:]), dtype=local_array.dtype)
    full[recv_global] = recv_array
    return full


# =============================================================================
# Backend call
# =============================================================================
def _dispatch(func: str, *, backend: str, **kwargs):
    """Run backend routine `func` on this rank's data; no communication."""
    return backend_module._dispatch(func, backend=backend, **kwargs)
