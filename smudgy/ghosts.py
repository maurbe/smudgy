"""Ghost-particle exchange and exact nearest-neighbour search across MPI ranks.

Each rank imports the particles of other ranks that lie within a search
radius around its own particles (ghost particles) and grows the radius until
every particle has found its true nearest neighbours. Periodic boundaries are
handled through scipy's periodic k-d tree; ghost positions are stored
unshifted.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from . import execution
from .backend.neighbors import build_kdtree, query_kdtree
from .decomposition import DecompositionInfo

FloatArray = npt.NDArray[np.floating]

_UNIT_BALL_VOLUME = {1: 2.0, 2: np.pi, 3: 4.0 / 3.0 * np.pi}


@dataclass
class GhostInfo:
    """Result of :func:`exchange_ghosts`.

    Parameters
    ----------
    ghost_positions : np.ndarray
        Positions of the particles imported from other ranks, shape
        (n_ghost, dim), wrapped into ``[0, boxsize)`` if periodic.
    ghost_weights : np.ndarray
        Weights of the ghost particles, shape (n_ghost,).
    ghost_source_rank : np.ndarray
        Owning rank of each ghost, shape (n_ghost,).
    ghost_source_local_index : np.ndarray
        Row index of each ghost in its source rank's local arrays, shape
        (n_ghost,).
    ghost_global_index : np.ndarray
        Original (pre-decomposition) index of each ghost, shape (n_ghost,).
    nn_inds : np.ndarray
        Nearest-neighbour indices, shape (n_target, k), into the combined array
        of local particles (rows ``0..n_local``) followed by ghosts.
    nn_dists : np.ndarray
        Nearest-neighbour distances, shape (n_target, k).
    radius : float
        Final search radius of this rank.
    export_local_index : np.ndarray
        Local row indices of the particles this rank exported as ghosts,
        grouped by destination rank, shape (n_exported,).
    export_counts : np.ndarray
        Number of particles exported to each rank, shape (size,).
    n_local : int
        Number of local particles at exchange time.
    """

    ghost_positions: np.ndarray = None
    ghost_weights: np.ndarray = None
    ghost_source_rank: np.ndarray = None
    ghost_source_local_index: np.ndarray = None
    ghost_global_index: np.ndarray = None
    nn_inds: np.ndarray = None
    nn_dists: np.ndarray = None
    radius: float = None
    export_local_index: np.ndarray = None
    export_counts: np.ndarray = None
    n_local: int = None


def wrap_periodic(positions: FloatArray, boxsize: FloatArray | None) -> FloatArray:
    """Wrap positions into ``[0, boxsize)`` per axis.

    Parameters
    ----------
    positions : np.ndarray
        Positions, shape (N, D).
    boxsize : np.ndarray or None
        Box size per axis; if None, the positions are returned unchanged.

    Returns
    -------
    np.ndarray
        Wrapped positions, shape (N, D).
    """
    if boxsize is None:
        return positions
    return np.mod(positions, np.asarray(boxsize))


def _shift_vectors(dim: int, periodic: bool) -> npt.NDArray[np.int64]:
    """{-1,0,1}^dim (shape (3**dim, dim)) if periodic, else [[0]*dim] only."""
    if not periodic:
        return np.zeros((1, dim), dtype=np.int64)
    return np.array(list(itertools.product((-1, 0, 1), repeat=dim)), dtype=np.int64)


def _boxes_overlap(a_min, a_max, b_min, b_max) -> bool:
    """Closed-interval axis-aligned bounding-box overlap test."""
    return bool(np.all(a_min <= b_max) and np.all(b_min <= a_max))


def _overlap_shifts(query_min, query_max, target_min, target_max, boxsize, shifts):
    """Which shift vectors make (target + shift*boxsize) overlap [query_min, query_max].

    `boxsize` is None for non-periodic domains, in which case `shifts` is
    already just the single zero vector (see `_shift_vectors`) and this
    reduces to one direct overlap check.
    """
    hits = []
    for s in shifts:
        offset = s * boxsize if boxsize is not None else 0.0
        if _boxes_overlap(query_min, query_max, target_min + offset, target_max + offset):
            hits.append(s)
    return hits


def _initial_radius(
    n_local, local_min, local_max, num_neighbors, dim,
    domain_min, domain_max, boxsize, periodic, safety_factor=2.0,
) -> float:
    """Return the first search radius of the ghost exchange.

    It is `safety_factor` times the radius of a ball holding `num_neighbors`
    particles at this rank's mean particle density. Ranks with fewer than 2 particles use a quarter of the smallest domain
    extent. The factor 2 lets most particles near the rank boundary converge
    in the first round; any positive start converges eventually.
    """
    if n_local < 2:
        extent = np.asarray(boxsize) if periodic else (domain_max - domain_min)
        return float(np.min(extent)) / 4.0
    volume = float(np.prod(np.maximum(local_max - local_min, 1e-12)))
    density = n_local / volume
    r_k = (num_neighbors / (density * _UNIT_BALL_VOLUME[dim])) ** (1.0 / dim)
    return safety_factor * r_k


def _select_ghosts_to_send(
    rank, size, wrapped_local, local_weights, local_global_indices,
    all_boxes, boxsize, periodic, dim,
):
    """Select this rank's particles inside every other rank's padded target box.

    Periodic images are taken into account. Returns the per-destination
    counts and the selected positions, weights, local and global indices,
    grouped by destination rank (the layout `execution._alltoallv_rows`
    expects). A rank never sends to itself: its own tree already holds its
    particles, periodic images included.
    """
    n_local = wrapped_local.shape[0]
    shifts = _shift_vectors(dim, periodic)
    send_counts = np.zeros(size, dtype=np.int64)
    pos_chunks, w_chunks, local_idx_chunks, global_idx_chunks = [], [], [], []

    if n_local > 0:
        local_min, local_max = wrapped_local.min(axis=0), wrapped_local.max(axis=0)
        for a in range(size):
            if a == rank:
                continue
            a_min, a_max, a_radius, a_has_particles = all_boxes[a]
            if not a_has_particles:
                continue
            a_min_padded, a_max_padded = a_min - a_radius, a_max + a_radius
            hits = _overlap_shifts(
                a_min_padded, a_max_padded, local_min, local_max, boxsize, shifts
            )
            if not hits:
                continue
            mask = np.zeros(n_local, dtype=bool)
            for s in hits:
                offset = s * boxsize if boxsize is not None else 0.0
                # a's padded box in this rank's (unshifted) coordinates
                region_min, region_max = a_min_padded - offset, a_max_padded - offset
                mask |= np.all(
                    (wrapped_local >= region_min) & (wrapped_local <= region_max), axis=1
                )
            if not mask.any():
                continue
            send_counts[a] = int(mask.sum())
            pos_chunks.append(wrapped_local[mask])
            w_chunks.append(local_weights[mask])
            local_idx_chunks.append(np.flatnonzero(mask))
            global_idx_chunks.append(local_global_indices[mask])

    def _concat(chunks, trailing_shape, dtype):
        return (
            np.concatenate(chunks, axis=0)
            if chunks
            else np.empty((0, *trailing_shape), dtype=dtype)
        )

    return (
        send_counts,
        _concat(pos_chunks, (dim,), wrapped_local.dtype),
        _concat(w_chunks, (), local_weights.dtype),
        _concat(local_idx_chunks, (), np.int64),
        _concat(global_idx_chunks, (), np.int64),
    )


def _query_k(tree, points, k, chunk_size=1 << 20):
    """K-NN of `points` as (n, k) float32 distances / int32 indices.

    Queried in chunks so scipy's float64/int64 output (2x the size) only ever
    exists for one chunk at a time. Indices are int64 only if the tree has
    >= 2**31 points.
    """
    n = points.shape[0]
    idx_dtype = np.int32 if tree.n < np.iinfo(np.int32).max else np.int64
    nn_dists = np.empty((n, k), dtype=np.float32)
    nn_inds = np.empty((n, k), dtype=idx_dtype)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        d, i = query_kdtree(tree, points[start:stop], k=k)
        nn_dists[start:stop] = d.reshape(-1, k)
        nn_inds[start:stop] = i.reshape(-1, k)
    return nn_dists, nn_inds


def _remap_ghost_indices(nn_inds, n_local, old_ghost_gidx, new_ghost_gidx):
    """Translate combined [local, ghost] indices from one ghost set to another.

    Local rows (< n_local) are unchanged; ghost rows are matched by global
    index. Every old ghost referenced must be present in the new set (true
    for exchange_ghosts' growing-radius rounds), otherwise this raises.
    """
    nn_inds = nn_inds.copy()
    is_ghost = nn_inds >= n_local
    gid = old_ghost_gidx[nn_inds[is_ghost] - n_local]
    order = np.argsort(new_ghost_gidx, kind="stable")
    sorted_gidx = new_ghost_gidx[order]
    pos = np.searchsorted(sorted_gidx, gid)
    pos = np.minimum(pos, max(len(sorted_gidx) - 1, 0))
    if gid.size and not np.array_equal(sorted_gidx[pos], gid):
        raise RuntimeError("ghost set shrank between exchange rounds")
    nn_inds[is_ghost] = order[pos] + n_local
    return nn_inds


def exchange_ghosts(
    comm: MPI.Comm,
    decomposition: DecompositionInfo,
    num_neighbors: int,
    dim: int,
    periodic: bool,
    boxsize: FloatArray | None,
    target_positions: FloatArray | None = None,
    max_iterations: int = 20,
    on_max_iterations: str = "raise",
) -> GhostInfo:
    """Import ghost particles and solve the exact k-nearest-neighbour problem.

    Collective call: every rank must call it, including ranks without local
    particles.

    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator.
    decomposition : DecompositionInfo
        Domain decomposition of the point cloud.
    num_neighbors : int
        Number of nearest neighbours k.
    dim : int
        Spatial dimension.
    periodic : bool
        Whether the domain is periodic.
    boxsize : np.ndarray or None
        Box size per axis (periodic domains only).
    target_positions : np.ndarray, optional
        Points to find neighbours for, shape (M, D). Defaults to this rank's
        own particles.
    max_iterations : int, default 20
        Maximum number of radius-growing iterations.
    on_max_iterations : str, default "raise"
        ``"raise"`` raises a ``RuntimeError`` if the search has not converged
        after ``max_iterations``; any other value only warns.

    Returns
    -------
    GhostInfo
        Ghost particles and the nearest-neighbour indices and distances.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_local = decomposition.local_positions.shape[0]
    total_n = comm.allreduce(n_local, op=MPI.SUM)
    if total_n < num_neighbors:
        raise ValueError(
            f"num_neighbors={num_neighbors} exceeds the total particle count "
            f"({total_n}); no search radius could ever find that many neighbors."
        )

    # PointCloud already wraps its particles; query targets may not be wrapped
    wrapped_local = wrap_periodic(
        decomposition.local_positions, boxsize if periodic else None
    )
    local_weights = decomposition.local_weights
    local_global_indices = decomposition.local_global_indices
    boxsize_arr = np.asarray(boxsize) if periodic else None

    wrapped_target = (
        wrapped_local
        if target_positions is None
        else wrap_periodic(target_positions, boxsize if periodic else None)
    )
    n_target = wrapped_target.shape[0]

    has_targets = n_target > 0
    if has_targets:
        target_min, target_max = wrapped_target.min(axis=0), wrapped_target.max(axis=0)
    else:
        target_min = target_max = np.zeros(dim, dtype=np.float64)

    has_particles = n_local > 0
    if has_particles:
        local_min, local_max = wrapped_local.min(axis=0), wrapped_local.max(axis=0)
    else:
        local_min = local_max = np.zeros(dim, dtype=np.float64)

    # from the particle density, not the target density: the radius must
    # reach num_neighbors particles, however many targets there are
    radius = _initial_radius(
        n_local, local_min, local_max, num_neighbors, dim,
        decomposition.domain_min, decomposition.domain_max, boxsize, periodic,
    )
    # Ghosts are only well defined for radii below half the periodic box
    # (minimum-image metric). The initial guess is capped there; only a
    # radius that has to grow beyond the cap is an error. A single rank
    # imports no ghosts, so the limit does not apply to it.
    radius_cap = (
        0.5 * float(np.min(boxsize_arr)) * (1.0 - 1e-6)
        if periodic and size > 1
        else np.inf
    )
    radius = min(radius, radius_cap)

    converged_mask = np.zeros(0, dtype=bool)
    nn_inds = nn_dists = prev_ghost_gidx = None
    recv_positions = recv_weights = recv_source_rank = None
    recv_local_idx = recv_global_idx = None

    for _iteration in range(max_iterations):
        # radii grow per rank, so the decision to raise is made collectively
        # (otherwise the other ranks would hang in the next collective)
        local_violation = radius > radius_cap
        any_violation = comm.allreduce(local_violation, op=MPI.LOR)
        if any_violation:
            max_radius = comm.allreduce(radius if local_violation else 0.0, op=MPI.MAX)
            raise ValueError(
                "ghost-exchange radius has grown to at least half the "
                f"(smallest) periodic box size on at least one rank (max "
                f"offending radius {max_radius}) -- scipy's periodic "
                "minimum-image distance metric is no longer well-defined "
                "beyond this point. This usually means num_neighbors is too "
                "large, or too many ranks are being used, for this domain "
                "size."
            )

        all_boxes = comm.allgather((target_min, target_max, radius, has_targets))

        send_counts, send_pos, send_w, send_lidx, send_gidx = _select_ghosts_to_send(
            rank, size, wrapped_local, local_weights, local_global_indices,
            all_boxes, boxsize_arr, periodic, dim,
        )
        recv_counts = execution._alltoall_counts(comm, send_counts)
        recv_positions = execution._alltoallv_rows(comm, send_pos, send_counts, recv_counts)
        recv_weights = execution._alltoallv_rows(comm, send_w, send_counts, recv_counts)
        recv_local_idx = execution._alltoallv_rows(comm, send_lidx, send_counts, recv_counts)
        recv_global_idx = execution._alltoallv_rows(comm, send_gidx, send_counts, recv_counts)
        recv_source_rank = np.repeat(np.arange(size, dtype=np.int64), recv_counts)

        if has_targets:
            combined_positions = np.concatenate([wrapped_local, recv_positions], axis=0)
            tree = build_kdtree(combined_positions, boxsize=boxsize_arr if periodic else None)
            if nn_inds is None:
                nn_dists, nn_inds = _query_k(tree, wrapped_target, num_neighbors)
            else:
                # Targets converged in an earlier round keep their (exact)
                # neighbors: the radius only grows, so this round's ghost set
                # is a superset of the one they were solved against -- only
                # their ghost indices need remapping into the new combined
                # array. Only the unconverged targets are re-queried.
                nn_inds[converged_mask] = _remap_ghost_indices(
                    nn_inds[converged_mask], n_local, prev_ghost_gidx, recv_global_idx
                )
                todo = np.flatnonzero(~converged_mask)
                if todo.size:
                    new_dists, new_inds = _query_k(
                        tree, wrapped_target[todo], num_neighbors
                    )
                    nn_inds = nn_inds.astype(new_inds.dtype, copy=False)
                    nn_dists[todo], nn_inds[todo] = new_dists, new_inds
            prev_ghost_gidx = recv_global_idx
            d_k = nn_dists[:, -1]
            finite = np.isfinite(d_k)
            if size == 1:
                # no other rank to import ghosts from: the local query is
                # already the exact K-NN (it can't be infinite either, since
                # total_n >= num_neighbors was checked above)
                converged_mask = finite
            else:
                # margin: d_k is float32-rounded, so a d_k a hair beyond the
                # radius could otherwise round down onto it
                converged_mask = finite & (d_k <= radius * (1.0 - 1e-6))
            local_converged = bool(np.all(converged_mask))
        else:
            nn_dists = np.empty((0, num_neighbors), dtype=np.float32)
            nn_inds = np.empty((0, num_neighbors), dtype=np.int32)
            converged_mask = np.zeros(0, dtype=bool)
            local_converged = True

        global_converged = comm.allreduce(local_converged, op=MPI.LAND)
        if global_converged:
            break

        if has_targets and not local_converged:
            unconverged = ~converged_mask
            grow_finite = finite & unconverged
            grow_deficit = (~finite) & unconverged
            new_radius = radius
            if np.any(grow_finite):
                new_radius = max(new_radius, float(np.max(d_k[grow_finite])) * (1.0 + 1e-5))
            if np.any(grow_deficit):
                new_radius = max(new_radius, radius * 2.0)
            # try the cap itself once before giving up
            radius = min(new_radius, radius_cap) if radius < radius_cap else new_radius
    else:
        n_unconverged = int(np.sum(~converged_mask))
        total_unconverged = comm.allreduce(n_unconverged, op=MPI.SUM)
        message = (
            f"Ghost exchange did not converge after {max_iterations} iterations "
            f"({total_unconverged} points globally still unconverged; "
            f"rank {rank} radius={radius})."
        )
        if on_max_iterations == "raise":
            raise RuntimeError(message)
        if rank == 0:
            warnings.warn(message)

    return GhostInfo(
        ghost_positions=recv_positions,
        ghost_weights=recv_weights,
        ghost_source_rank=recv_source_rank,
        ghost_source_local_index=recv_local_idx,
        ghost_global_index=recv_global_idx,
        nn_inds=nn_inds,
        nn_dists=nn_dists,
        radius=radius,
        n_local=n_local,
        # export manifest of the final round, used by push_to_ghosts
        export_local_index=send_lidx,
        export_counts=send_counts,
    )


def push_to_ghosts(
    comm: MPI.Comm, ghost_info: GhostInfo, local_array: np.ndarray
) -> np.ndarray:
    """Send per-particle values to the ranks that imported these particles as ghosts.

    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator.
    ghost_info : GhostInfo
        Result of :func:`exchange_ghosts`.
    local_array : np.ndarray
        Values of this rank's local particles, shape (n_local, ...).

    Returns
    -------
    np.ndarray
        Values of this rank's ghost particles, aligned with
        ``ghost_info.ghost_positions``.
    """
    send_values = local_array[ghost_info.export_local_index]
    recv_counts = execution._alltoall_counts(comm, ghost_info.export_counts)
    return execution._alltoallv_rows(
        comm, send_values, ghost_info.export_counts, recv_counts
    )
