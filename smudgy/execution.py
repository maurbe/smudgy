"""Execution module for smudgy, providing parallelized dispatching of computational functions across MPI ranks."""

import numpy as np
from mpi4py import MPI

from . import backend as backend_module

CHUNK_AXIS0 = {
    "compute_hsml": ("nn_dists",),
    "compute_hmat": ("query_positions", "neighbor_positions", "neighbor_weights"),
    "compute_density": ("neighbor_weights", "r_ij", "h"),
    "project_2d": ("h_tensor",),
    "interpolate": ("fields", "weights", "r_ij", "h"),
    "deposit": (
        "particle_positions",
        "particle_fields",
        "particle_weights",
        "particle_hsml",
        "particle_hmat_eigvecs",
        "particle_hmat_eigvals",
    ),
}

REDUCTION = {
    "compute_hsml": "gather",
    "compute_hmat": "gather",
    "compute_density": "gather",
    "project_2d": "gather",
    "interpolate": "gather",
    "deposit": "reduce_sum",
}


def _local_slice(n, rank, size):
    """Contiguous, balanced [start, stop) range for this rank."""
    counts = np.full(size, n // size, dtype=int)
    counts[: n % size] += 1
    starts = np.concatenate(([0], np.cumsum(counts)))[:-1]
    return int(starts[rank]), int(starts[rank] + counts[rank])


def _scatter(func, kwargs, rank, size):
    chunk_keys = CHUNK_AXIS0.get(func, ())
    if not chunk_keys:
        return kwargs

    n = kwargs[chunk_keys[0]].shape[0]
    start, stop = _local_slice(n, rank, size)

    local_kwargs = dict(kwargs)
    for key in chunk_keys:
        local_kwargs[key] = kwargs[key][start:stop]
    return local_kwargs


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


def _dispatch(func: str, *, backend: str, **kwargs):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    if size == 1:
        return backend_module._dispatch(func, backend=backend, **kwargs)

    if func not in REDUCTION:
        raise ValueError(f"No MPI reduction strategy registered for '{func}'")

    rank = comm.Get_rank()
    local_kwargs = _scatter(func, kwargs, rank, size)
    local_result = backend_module._dispatch(func, backend=backend, **local_kwargs)

    reduction = REDUCTION[func]
    if reduction == "gather":
        return _gather(comm, local_result)
    if reduction == "reduce_sum":
        return _reduce_sum(comm, local_result)
    raise ValueError(f"Unknown reduction strategy '{reduction}' for '{func}'")
