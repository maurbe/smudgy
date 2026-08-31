"""Ghost-particle exchange and iterative true-KNN solver.

Step 2 of the MPI domain-decomposition roadmap: given a `DecompositionInfo`
(Step 1), finds each local particle's true global K-nearest-neighbor result
using only ghost particles fetched from other ranks -- never a full
broadcast. Handles periodic boundary conditions (`boxsize`) via periodic-image
bounding-box overlap checks for ghost SELECTION; the actual periodic distance
math needs no new code, since `backend.neighbors.build_kdtree`/`query_kdtree`
already support scipy's periodic `boxsize` mode (used today for the single
global tree in `PointCloud._check_tree`) -- a local tree built the same way
over (local + ghost) points computes correct wraparound distances
automatically, so ghost positions are sent and stored unmodified (no
+/-boxsize shifting of particle data is ever needed, only of the SEARCH
REGION used to decide who to fetch from -- see `_select_ghosts_to_send`).

This module is opt-in and self-contained: nothing else in the package reads
`GhostInfo` yet (see `PointCloud.find_neighbors`).

`PointCloud.__init__` already wraps `self.positions` into `[0, boxsize)` for
periodic clouds (fixing raw unwrapped input, and float32-rounding artifacts
right at the boundary that scipy's periodic `cKDTree` rejects outright). So
`decomposition.local_positions` -- a reordered/redistributed slice of
`self.positions` -- is canonical by construction in practice, which makes
this module's own `wrap_periodic` call a safety net, not the primary fix:
it's usually a no-op, kept so this module doesn't silently depend on
`__init__` being the only path its input could ever arrive from.
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
    """Result of `exchange_ghosts`.

    Parameters
    ----------
    ghost_positions : np.ndarray
        Canonical (wrapped into [0, boxsize) if periodic) positions of
        particles imported from other ranks, shape (n_ghost, dim). Unlike
        `DecompositionInfo.local_positions`, these are guaranteed wrapped --
        a caller combining this with local data should wrap
        `decomposition.local_positions` itself first (see `wrap_periodic`);
        this dataclass does not duplicate a wrapped copy of the local array.
    ghost_weights : np.ndarray
        Shape (n_ghost,).
    ghost_source_rank : np.ndarray
        Owning rank of each ghost, shape (n_ghost,) int64.
    ghost_source_local_index : np.ndarray
        Row index into the source rank's own `DecompositionInfo.local_*`
        arrays, shape (n_ghost,) int64.
    ghost_global_index : np.ndarray
        Original pre-decomposition row index of each ghost, shape (n_ghost,)
        int64 (== source rank's `local_global_indices[ghost_source_local_index]`).
    nn_inds : np.ndarray
        Shape (n_target, k) int64, where n_target is n_local unless
        `exchange_ghosts` was called with an explicit `target_positions`
        (Step 4b: solving K-NN for an arbitrary point set, e.g. routed
        interpolation query positions, instead of this rank's own particles),
        in which case n_target = target_positions.shape[0]. Indices are
        always into the CONCEPTUAL combined array of this rank's own
        particles -- [rows 0..n_local) -- followed by ghost particles --
        [rows n_local..n_local+n_ghost) -- regardless of what was queried: a
        value < n_local indexes this rank's own `decomposition.local_*`
        arrays directly; a value >= n_local indexes `ghost_*` at row
        (value - n_local).
    nn_dists : np.ndarray
        Shape (n_target, k) float (see `nn_inds`).
    radius : float
        This rank's final converged padding radius.
    n_local : int
        Local particle count at exchange time (the local/ghost split point
        used by `nn_inds` above).
    export_local_index : np.ndarray
        The mirror image of `ghost_source_local_index`, from this rank's own
        (sender) perspective: row index into this rank's own
        `DecompositionInfo.local_*` arrays for each particle it exported as
        a ghost to some other rank, grouped by destination rank ascending,
        shape (n_exported,) int64. Used by `push_to_ghosts` to ship a
        later-computed local value (density, smoothing length/tensor, a
        field) out along the exact same routing this exchange already
        established, without rediscovering it.
    export_counts : np.ndarray
        How many of this rank's local particles were exported to each
        destination rank, shape (size,) int64 -- the send-side counts that
        produced `export_local_index`'s grouping.

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
    """Wrap positions into [0, boxsize) per axis; identity if boxsize is None.

    `PointCloud.__init__` already does this (in float32, with an extra clip
    against boundary-rounding artifacts) for `self.positions`, so calling
    this on a slice/redistribution of that array -- as `exchange_ghosts`
    does on `decomposition.local_positions` -- is normally a no-op. Kept as
    a cheap, idempotent safety net rather than assuming `__init__` is the
    only path this data could ever arrive from.
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
    """R_0 = safety_factor * (radius whose D-ball contains num_neighbors
    points at this rank's own local density).

    Falls back to a domain-derived guess for ranks with fewer than 2 local
    particles (no meaningful local density estimate from 0-1 points).
    `safety_factor=2.0`: boundary-adjacent particles' local-only density
    estimate systematically undercounts (their true neighbors are split
    across the rank boundary -- the whole reason ghosting exists), so the
    bare density-derived radius would almost always need a growth round;
    doubling trades a little wasted first-round ghost import for a much
    better chance of first-round convergence. Tunable, not load-bearing for
    correctness -- the iterative solver converges from any positive start.
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
    """For THIS rank as sender: decide, for every destination rank a != rank,
    which of this rank's own local particles fall within a's padded box
    (accounting for periodic wraparound via `_overlap_shifts`), and package
    them grouped by destination rank in ascending rank order (matching
    `execution._alltoallv_rows`'s convention).

    A rank never selects itself as a destination: its own local periodic
    tree already contains all its own particles, and a local periodic query
    already finds correct local-to-local wraparound distances among them
    with no ghosting needed -- re-importing a particle as its own ghost
    would duplicate it, causing it to appear as its own neighbor.
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
                # translate a's requested (padded) region into THIS rank's
                # own native coordinate frame -- the particle data sent is
                # always the unmodified native position (see module docstring)
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
    """Iteratively fetch ghosts and solve true K-NN for `target_positions`.

    See module docstring for the periodic-boundary design. Every rank must
    call this collectively (it runs the same sequence of MPI collectives on
    every rank every iteration, including ranks with zero local particles --
    see `DecompositionInfo`'s `N < P` case -- so that no rank ever waits on
    a peer that has stopped participating).

    `target_positions`, default `None`, is the (possibly per-rank-varying
    size) set of points to solve K-NN for -- defaults to this rank's own
    `decomposition.local_positions` (Step 2's original use: find each local
    particle's true neighbors). Passing a different array (Step 4b: e.g.
    routed interpolation query positions) solves K-NN for those points
    instead, using the exact same mechanism: the *requested-region bounding
    box and convergence check* are based on `target_positions`, while the
    *initial radius estimate* is deliberately still based on this rank's own
    PARTICLE density (`n_local`/local bbox), never on `target_positions`'s
    own count/bbox -- the radius needed to reach `num_neighbors` particles is
    governed by how densely packed particles are nearby, which has nothing
    to do with how many target points happen to be nearby (a target batch
    far sparser than the local particles over a similar region would
    otherwise produce a wildly inflated first guess; see `_initial_radius`'s
    call site). Ghosts are still always drawn from other ranks' own particle
    data regardless (the sender side, `_select_ghosts_to_send`, only ever
    looks at *its own* particles against the *destination's* requested box --
    it does not care what that box represents). The provably-sufficient
    radius-growth argument (see below) depends only on what a given radius
    makes available to fetch, via that same sender-side overlap test --
    never on whether the requesting box came from the requester's own
    particles -- so it carries over
    unchanged. `target_positions` need not have associated weights (it is
    never sent to any rank as a ghost, only queried against).
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

    # Safety net, usually a no-op: PointCloud.__init__ already wraps
    # self.positions into [0, boxsize) for periodic clouds, and
    # decomposition.local_positions is just a reordered/redistributed slice
    # of that array -- see wrap_periodic's docstring.
    wrapped_local = wrap_periodic(
        decomposition.local_positions, boxsize if periodic else None
    )
    local_weights = decomposition.local_weights
    local_global_indices = decomposition.local_global_indices
    boxsize_arr = np.asarray(boxsize) if periodic else None

    # Unlike wrapped_local above, this is NOT usually a no-op when
    # target_positions is explicitly given: an arbitrary caller-supplied
    # point set (e.g. interpolation query positions) has no guarantee of
    # already lying in [0, boxsize) the way self.positions does.
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

    # Deliberately estimated from THIS RANK'S OWN PARTICLE density
    # (n_local/local_min/local_max), never from n_target/target_min/
    # target_max: the radius needed to reach num_neighbors *particles* is
    # governed by how densely packed particles are nearby, which has nothing
    # to do with how many target points happen to be nearby. Conflating the
    # two (an earlier version of this code did, using n_target's own count)
    # systematically mis-estimates the initial guess whenever target_positions
    # has a different density than the local particles -- e.g. far fewer
    # query points than particles covering a similar region produces a wildly
    # inflated first guess (found empirically: tripped the periodic half-box
    # guard on the very first iteration, before any real ghost-fetching had
    # even happened, for a query batch ~10x sparser than the local particles
    # over a similar bounding box). Using particle density instead matches
    # what `target_positions=None` (particles-as-their-own-target) already
    # does exactly, since n_target == n_local there.
    radius = _initial_radius(
        n_local, local_min, local_max, num_neighbors, dim,
        decomposition.domain_min, decomposition.domain_max, boxsize, periodic,
    )

    converged_mask = np.zeros(0, dtype=bool)
    nn_inds = nn_dists = None
    recv_positions = recv_weights = recv_source_rank = None
    recv_local_idx = recv_global_idx = None

    for _iteration in range(max_iterations):
        # The raise decision below MUST be identical on every rank: `radius`
        # is grown independently per rank, so checking it locally and
        # raising immediately (as an earlier version of this code did) can
        # trigger on some ranks but not others -- the raising rank(s) would
        # abort mid-collective-sequence while survivors hang forever waiting
        # on a peer that will never call the next `allgather`/`Alltoallv`
        # again. Every rank must agree on whether to raise, via a collective,
        # before any rank actually raises.
        local_violation = periodic and radius >= float(np.min(boxsize_arr)) / 2.0
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
            # combined_positions is always this rank's own particles + fetched
            # ghost particles (never target_positions itself -- a target point
            # is only ever queried against, not concatenated in as something
            # that could be returned as someone's neighbor).
            combined_positions = np.concatenate([wrapped_local, recv_positions], axis=0)
            tree = build_kdtree(combined_positions, boxsize=boxsize_arr if periodic else None)
            nn_dists, nn_inds = query_kdtree(tree, wrapped_target, k=num_neighbors)
            if num_neighbors == 1:
                nn_dists = nn_dists.reshape(-1, 1)
                nn_inds = nn_inds.reshape(-1, 1)
            d_k = nn_dists[:, -1]
            finite = np.isfinite(d_k)
            converged_mask = finite & (d_k <= radius)
            local_converged = bool(np.all(converged_mask))
        else:
            nn_dists = np.empty((0, num_neighbors), dtype=np.float64)
            nn_inds = np.empty((0, num_neighbors), dtype=np.int64)
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
            radius = new_radius
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
        # send_counts/send_lidx already hold the final (converged) iteration's
        # values here -- _select_ghosts_to_send (called every iteration,
        # ghosts.py above) recomputes them fresh each round, so by the time
        # the loop exits they're exactly this rank's export manifest for the
        # ghost set actually returned above. See push_to_ghosts.
        export_local_index=send_lidx,
        export_counts=send_counts,
    )


def push_to_ghosts(
    comm: MPI.Comm, ghost_info: GhostInfo, local_array: np.ndarray
) -> np.ndarray:
    """Ship `local_array`'s values out to whichever ranks imported those
    particles as ghosts of theirs, reusing `exchange_ghosts`'s already-
    established routing.

    `local_array` must be indexed the same way as `decomposition.local_*`
    (row i = this rank's i-th local particle) -- e.g. a just-computed
    `smoothing_lengths`, `density_isotropic`, or a locally-sliced field.
    No new selection or communication topology is discovered here:
    `ghost_info.export_local_index`/`export_counts` are exactly the arrays
    that built `ghost_info.ghost_positions` etc. in the first place, so
    replaying them for a new payload reproduces the identical per-destination
    grouping -- the result lands aligned row-for-row with `ghost_positions`
    (and friends), directly usable via
    `np.concatenate([local_array, push_to_ghosts(comm, ghost_info, local_array)])`
    wherever an `nn_inds` value `>= n_local` needs to index into it.
    """
    send_values = local_array[ghost_info.export_local_index]
    recv_counts = execution._alltoall_counts(comm, ghost_info.export_counts)
    return execution._alltoallv_rows(
        comm, send_values, ghost_info.export_counts, recv_counts
    )
