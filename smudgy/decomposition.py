"""Hilbert-curve spatial domain decomposition.

Step 1 of the MPI domain-decomposition roadmap: partitions particles across
MPI ranks into count-balanced, spatially-contiguous chunks (sorted by
Hilbert-curve code), then redistributes positions/weights via
`execution._scatterv_rows`. Produces a `DecompositionInfo` side artifact
(local chunk + provenance) for later steps (ghost exchange, local-only
compute, gather-to-root) to build on.

Also home to `route_query_positions` (Step 4b): routes an arbitrary
(non-particle) query-position array to ranks using the *same* Hilbert-code
partition, via `DecompositionInfo.boundary_codes` -- reusing the partition
`decompose()` already computed rather than deriving a new one for query
points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from . import execution

FloatArray = npt.NDArray[np.floating]

# Bits per axis chosen so dim * bits_per_dim <= 64 (fits in a uint64 code),
# while using as much of that budget as possible for resolution.
_DEFAULT_BITS_PER_DIM = {1: 64, 2: 32, 3: 21}


@dataclass
class DecompositionInfo:
    """Result of a Hilbert-curve domain decomposition (see `hilbert_partition_and_scatter`).

    Parameters
    ----------
    local_positions : np.ndarray
        This rank's chunk of positions, shape (n_local, D), Hilbert-sorted.
    local_weights : np.ndarray
        This rank's chunk of weights, shape (n_local,).
    local_global_indices : np.ndarray
        Original row index (into the pre-decomposition array) of each local
        particle, shape (n_local,). Needed to reassemble results in the
        caller's original order later.
    counts : np.ndarray
        Per-rank particle counts, shape (size,), identical on every rank.
    domain_min : np.ndarray
        Lower extent, shape (D,), used for Hilbert quantization.
    domain_max : np.ndarray
        Upper extent, shape (D,), used for Hilbert quantization.
    boundary_codes : np.ndarray
        Shape (size+1,) uint64, identical on every rank. The actual
        count-based partition cut points used above: rank `i` owns Hilbert
        codes in `[boundary_codes[i], boundary_codes[i+1])`. `boundary_codes[0]`
        is always 0 and `boundary_codes[size]` is always the maximum uint64
        value, so together the intervals cover the entire code space (not
        just where particles happen to be) -- needed so an arbitrary query
        position (not a particle) can still be routed to exactly one rank,
        consistently with the partition actually used above. A rank with 0
        local particles gets a zero-width interval (its own boundary equals
        the next non-empty rank's), so it is correctly never routed anything.
        See `route_query_positions`.
    root_order : np.ndarray
        Shape (N,) int64, the Hilbert-sort permutation used to build every
        rank's chunk above -- populated on `root` only (`None` on every
        other rank; the one intentionally rank-asymmetric field on this
        dataclass). This is the one piece of O(N) bookkeeping deliberately
        kept on `root` after construction: a field added later (`add_fields`)
        needs to be split across ranks *consistently* with how positions/
        weights were originally split, and by the time a field arrives,
        `root` no longer has the full positions array to re-derive that
        split from -- `values[root_order]` reorders a newly-arriving full-N
        field into the same order `local_global_indices` already encodes in
        scattered form, before it's chunked out via `execution._scatterv_rows`.

    """

    local_positions: np.ndarray = None
    local_weights: np.ndarray = None
    local_global_indices: np.ndarray = None
    counts: np.ndarray = None
    domain_min: np.ndarray = None
    domain_max: np.ndarray = None
    boundary_codes: np.ndarray = None
    root_order: np.ndarray = None


@dataclass
class QueryRouting:
    """Result of `route_query_positions`.

    Routes an arbitrary (M, D) query-position array to ranks using the exact
    same Hilbert-code partition `hilbert_partition_and_scatter` already
    computed for particles (`DecompositionInfo.boundary_codes`) -- so a query
    point lands on whichever rank's spatial region, as defined by that
    partition, contains it. Unrelated to and independent from any earlier
    `route_query_positions` call: there is no persistent "query decomposition"
    the way there is for particles, this is recomputed fresh per call.

    Parameters
    ----------
    local_positions : np.ndarray
        This rank's chunk of query positions, shape (n_local_queries, D).
    local_global_indices : np.ndarray
        Original row index (into the caller's (M, D) query array) of each
        local query point, shape (n_local_queries,) -- mirrors
        `DecompositionInfo.local_global_indices`'s role, needed to reassemble
        a per-rank result back into the caller's original order.
    counts : np.ndarray
        Per-rank query-point counts, shape (size,), identical on every rank.

    """

    local_positions: np.ndarray = None
    local_global_indices: np.ndarray = None
    counts: np.ndarray = None


def _quantize(
    positions: FloatArray,
    domain_min: FloatArray,
    domain_max: FloatArray,
    bits_per_dim: int,
    periodic: bool,
) -> npt.NDArray[np.uint64]:
    """Map `positions` to per-axis integer coordinates in [0, 2**bits_per_dim - 1].

    Periodic axes are wrapped modulo the domain extent (so an out-of-range
    coordinate lands at its true periodic image); non-periodic axes are
    clipped (a defensive guard against float rounding only -- no real input
    point should ever be outside its own bounding box). A degenerate axis
    (domain_max == domain_min) is quantized to 0 uniformly rather than
    dividing by zero.
    """
    domain_min = np.asarray(domain_min, dtype=np.float64)
    domain_max = np.asarray(domain_max, dtype=np.float64)
    extent = domain_max - domain_min
    positions = np.asarray(positions, dtype=np.float64)

    degenerate = extent <= 0
    safe_extent = np.where(degenerate, 1.0, extent)

    if periodic:
        rel = np.mod(positions - domain_min, safe_extent)
    else:
        rel = np.clip(positions - domain_min, 0.0, safe_extent)

    frac = np.where(degenerate, 0.0, rel / safe_extent)
    num_bins = 1 << bits_per_dim
    # Scale by num_bins (not max_index) so bins are evenly filled -- frac==1.0
    # (a clipped position exactly at domain_max) is the only case that can
    # reach num_bins itself, which the final clip pulls back down below it.
    #
    # The clip's upper bound is deliberately computed via np.nextafter, not
    # float(num_bins - 1): for bits_per_dim=64 (dim=1's default -- the only
    # case wide enough to matter, since 2/3D's 32/21 bits stay well within
    # float64's exact-integer range), float64 cannot distinguish 2**64 - 1
    # from 2**64 at all (both round to the same value) -- so float(max_index)
    # was silently equal to float(num_bins) in that case, and the clip did
    # nothing: a value of exactly num_bins (2**64) could reach the final
    # `.astype(np.uint64)` cast, which is out of uint64's representable range
    # ([0, 2**64 - 1]) and undefined behavior (numpy warns "invalid value
    # encountered in cast" and the result is not well-defined). nextafter
    # gives the largest float64 strictly below num_bins regardless of bit
    # width -- off by a handful of bins out of 2**64 at the very top for the
    # 64-bit case, utterly negligible for Hilbert-curve bucketing at any
    # real particle count, and exact for the 32/21-bit cases.
    safe_max = np.nextafter(float(num_bins), 0.0)
    coords = np.clip(np.floor(frac * float(num_bins)), 0.0, safe_max).astype(np.uint64)
    return coords


def _axes_to_hilbert_transpose(
    coords: list[npt.NDArray[np.uint64]], bits_per_dim: int
) -> list[npt.NDArray[np.uint64]]:
    """Skilling (2004) AxesToTranspose, vectorized over points.

    `coords`: list of D uint64 arrays, shape (N,), each holding an integer
    in [0, 2**bits_per_dim). Returns the Hilbert "transpose" representation
    (same shapes), which `_transpose_to_index` linearizes into a single code.

    The per-dimension loop below (`for i in range(dim)`) has a genuine data
    dependency on `x[0]` across iterations (matching Skilling's reference
    algorithm) and so stays a plain Python loop -- but `dim` is only 1-3, and
    every operation inside is a vectorized numpy op over all N points, so
    this is cheap regardless of N.
    """
    x = [c.copy() for c in coords]
    dim = len(x)
    m = np.uint64(1) << np.uint64(bits_per_dim - 1)

    q = m
    while q > 1:
        p = q - np.uint64(1)
        for i in range(dim):
            has_bit = (x[i] & q) != 0
            x[0] = np.where(has_bit, x[0] ^ p, x[0])
            t = (x[0] ^ x[i]) & p
            x[0] = np.where(~has_bit, x[0] ^ t, x[0])
            x[i] = np.where(~has_bit, x[i] ^ t, x[i])
        q >>= np.uint64(1)

    for i in range(1, dim):
        x[i] = x[i] ^ x[i - 1]

    t = np.zeros_like(x[0])
    q = m
    while q > 1:
        has_bit = (x[dim - 1] & q) != 0
        t = np.where(has_bit, t ^ (q - np.uint64(1)), t)
        q >>= np.uint64(1)

    for i in range(dim):
        x[i] = x[i] ^ t

    return x


def _transpose_to_index(
    transpose: list[npt.NDArray[np.uint64]], bits_per_dim: int
) -> npt.NDArray[np.uint64]:
    """Linearize the Hilbert transpose form into a single uint64 code per point.

    Bit `b` (MSB-first) of each axis's transpose word is appended to the
    output code, axis-by-axis, for b from `bits_per_dim - 1` down to 0.
    """
    dim = len(transpose)
    code = np.zeros_like(transpose[0])
    for b in range(bits_per_dim - 1, -1, -1):
        for i in range(dim):
            bit = (transpose[i] >> np.uint64(b)) & np.uint64(1)
            code = (code << np.uint64(1)) | bit
    return code


def hilbert_encode(
    positions: FloatArray,
    domain_min: FloatArray,
    domain_max: FloatArray,
    periodic: bool = False,
    bits_per_dim: int | None = None,
) -> npt.NDArray[np.uint64]:
    """Vectorized Hilbert-curve encoding of positions into uint64 keys.

    Quantizes `positions` (shape (N, D)) into fixed-point integer
    coordinates within [domain_min, domain_max] per axis, then computes each
    point's position along a Peano-Hilbert curve threaded through that
    domain (Skilling 2004). Points close in D-dimensional space get close
    Hilbert codes -- unlike a simpler Morton/Z-order interleave, the Hilbert
    curve never makes long-range jumps, which matters for keeping later
    ghost-exchange communication volume small.

    `periodic` selects wrapping (mod domain extent) vs. clipping for
    out-of-range coordinates -- see `_quantize`. `bits_per_dim` defaults to
    the widest resolution that fits dim*bits_per_dim into 64 bits (64/32/21
    for dim=1/2/3); exposed mainly for testing against a reference
    implementation at a small, hand-checkable bit depth.

    Returns
    -------
    npt.NDArray[np.uint64], shape (N,)
    """
    dim = positions.shape[1]
    if bits_per_dim is None:
        bits_per_dim = _DEFAULT_BITS_PER_DIM[dim]

    coords = [
        _quantize(positions[:, i], domain_min[i], domain_max[i], bits_per_dim, periodic)
        for i in range(dim)
    ]

    if dim == 1:
        return coords[0]

    transpose = _axes_to_hilbert_transpose(coords, bits_per_dim)
    return _transpose_to_index(transpose, bits_per_dim)


def hilbert_partition_and_scatter(
    comm: MPI.Comm,
    positions: FloatArray | None,
    weights: FloatArray | None,
    domain_min: FloatArray,
    domain_max: FloatArray,
    periodic: bool = False,
    root: int = 0,
) -> DecompositionInfo:
    """Rank-0 Hilbert sort + particle-count-balanced split, then Scatterv to all ranks.

    `positions`/`weights` must be the full, global (N,D)/(N,) arrays on
    `root`; ignored (may be `None`) on every other rank -- mirrors
    `execution._bcast_array`'s calling convention, so this is a drop-in
    replacement for that broadcast once a later step stops needing full
    replication. `domain_min`/`domain_max` (shape (D,)) must already be
    resolved and identical on every rank before calling this.

    Returns
    -------
    DecompositionInfo
        `local_positions`/`local_weights` (this rank's Hilbert-sorted
        chunk), `local_global_indices` (original row index of each local
        particle), `counts` (per-rank counts, identical on every rank).
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size == 1:
        # No other rank to partition against -- skip the Hilbert encode/sort/
        # scatter entirely (expensive: O(N) encode + O(N log N) sort, on the
        # same order as the neighbor search itself) and hand back this
        # process's own arrays unchanged, in their original order, as a
        # single-rank "partition". Gated on `size == 1` rather than
        # `rank == root`: with one process `comm.Get_rank()` is always 0, so
        # gating on `rank == root` would (for a caller-supplied `root != 0`)
        # take the `else` branch below and `comm.bcast` from a rank that
        # doesn't exist in a 1-process communicator -- this also incidentally
        # fixes that latent crash, since there is exactly one process and its
        # `positions`/`weights` are necessarily the real (only) data
        # regardless of what `root` was asked for.
        n = positions.shape[0]
        counts = np.array([n], dtype=np.int64)
        local_global_indices = np.arange(n, dtype=np.int64)
        return DecompositionInfo(
            local_positions=np.ascontiguousarray(positions),
            local_weights=np.ascontiguousarray(weights),
            local_global_indices=local_global_indices,
            counts=counts,
            domain_min=np.asarray(domain_min),
            domain_max=np.asarray(domain_max),
            # Reuses the general formula (rather than hand-writing
            # [0, max_uint64]) so this stays byte-for-byte consistent with
            # what the size>1 path would produce for a single-rank `counts`
            # by construction: with an empty `codes_sorted`, its `if n > 0`
            # fill branch never runs, always leaving [0, max_uint64].
            boundary_codes=_partition_boundary_codes(
                np.empty(0, dtype=np.uint64), counts
            ),
            root_order=local_global_indices,
        )

    if rank == root:
        n = positions.shape[0]
        codes = hilbert_encode(positions, domain_min, domain_max, periodic=periodic)
        order = np.argsort(codes, kind="stable")
        counts = execution._balanced_counts(n, size)
        positions_sorted = np.ascontiguousarray(positions[order])
        weights_sorted = np.ascontiguousarray(weights[order])
        global_indices_sorted = np.ascontiguousarray(order.astype(np.int64))
        boundary_codes = _partition_boundary_codes(codes[order], counts)
    else:
        counts = positions_sorted = weights_sorted = global_indices_sorted = None
        boundary_codes = None

    counts, boundary_codes = comm.bcast(
        (counts, boundary_codes) if rank == root else None, root=root
    )

    local_positions = execution._scatterv_rows(comm, positions_sorted, counts, root=root)
    local_weights = execution._scatterv_rows(comm, weights_sorted, counts, root=root)
    local_global_indices = execution._scatterv_rows(
        comm, global_indices_sorted, counts, root=root
    )

    return DecompositionInfo(
        local_positions=local_positions,
        local_weights=local_weights,
        local_global_indices=local_global_indices,
        counts=counts,
        domain_min=np.asarray(domain_min),
        domain_max=np.asarray(domain_max),
        boundary_codes=boundary_codes,
        # global_indices_sorted IS the sort permutation (order), already
        # computed above and None on non-root ranks -- kept here rather
        # than discarded, see DecompositionInfo.root_order's docstring.
        root_order=global_indices_sorted,
    )


def _partition_boundary_codes(
    codes_sorted: npt.NDArray[np.uint64], counts: npt.NDArray[np.int64]
) -> npt.NDArray[np.uint64]:
    """Derive the (size+1,) Hilbert-code cut points actually used by a
    count-based partition of `codes_sorted` (already sorted ascending) into
    per-rank chunks of size `counts`.

    `boundary_codes[i]` is the code of the first element of rank `i`'s chunk;
    `boundary_codes[0]` is always 0 and `boundary_codes[size]` is always the
    maximum uint64 value, so consecutive pairs
    `[boundary_codes[i], boundary_codes[i+1])` partition the *entire* code
    space (not just codes that actually occur), including empty (zero-width)
    intervals for any rank with 0 particles. See `DecompositionInfo`.
    """
    size = counts.shape[0]
    n = codes_sorted.shape[0]
    max_code = np.iinfo(np.uint64).max
    boundary_codes = np.full(size + 1, max_code, dtype=np.uint64)
    boundary_codes[0] = 0
    if n > 0:
        starts = np.cumsum(counts)[:-1]  # rank i's chunk starts at starts[i-1], i=1..size-1
        valid = starts < n
        clipped_starts = np.clip(starts, 0, n - 1)
        boundary_codes[1:size] = np.where(valid, codes_sorted[clipped_starts], max_code)
    return boundary_codes


def route_query_positions(
    comm: MPI.Comm,
    decomposition: DecompositionInfo,
    query_positions: FloatArray | None,
    periodic: bool = False,
    root: int = 0,
) -> QueryRouting:
    """Route an arbitrary (M, D) query-position array to ranks by the same
    Hilbert-code partition already used for particles, then Scatterv.

    `query_positions` must be the full, global (M, D) array on `root`;
    ignored (may be `None`) on every other rank -- same calling convention as
    `hilbert_partition_and_scatter`. `decomposition` must already have
    `boundary_codes` set (i.e. come from `hilbert_partition_and_scatter`).
    Uses `decomposition.domain_min`/`domain_max` (the *same* domain particles
    were quantized against) so a query point's code is computed on the
    identical curve, making "which rank owns this code" well-defined
    everywhere in the domain, not just where particles happen to be. An
    out-of-domain, non-periodic query position is handled by
    `hilbert_encode`'s existing defensive clipping (see `_quantize`) -- it
    routes to whichever rank owns the nearest boundary region, which affects
    only which rank does the work, never the correctness of the final K-NN
    answer found later by `ghosts.exchange_ghosts`.

    Returns
    -------
    QueryRouting
        `local_positions` (this rank's routed chunk), `local_global_indices`
        (original row index of each local query point, into the caller's
        (M, D) array), `counts` (per-rank counts, identical on every rank).
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size == 1:
        # Provably a no-op at size==1: `decomposition.boundary_codes` is
        # always [0, max_uint64] there (see `hilbert_partition_and_scatter`'s
        # own size==1 path), so `searchsorted` against it would always
        # resolve every query point to rank 0 anyway -- skip computing that
        # and just route everything here directly, in original order.
        m = query_positions.shape[0]
        local_global_indices = np.arange(m, dtype=np.int64)
        return QueryRouting(
            local_positions=np.ascontiguousarray(query_positions),
            local_global_indices=local_global_indices,
            counts=np.array([m], dtype=np.int64),
        )

    if rank == root:
        codes = hilbert_encode(
            query_positions,
            decomposition.domain_min,
            decomposition.domain_max,
            periodic=periodic,
        )
        owning_rank = np.clip(
            np.searchsorted(decomposition.boundary_codes, codes, side="right") - 1,
            0,
            size - 1,
        )
        order = np.argsort(owning_rank, kind="stable")
        counts = np.bincount(owning_rank, minlength=size).astype(np.int64)
        positions_sorted = np.ascontiguousarray(query_positions[order])
        global_indices_sorted = np.ascontiguousarray(order.astype(np.int64))
    else:
        counts = positions_sorted = global_indices_sorted = None

    counts = comm.bcast(counts if rank == root else None, root=root)

    local_positions = execution._scatterv_rows(comm, positions_sorted, counts, root=root)
    local_global_indices = execution._scatterv_rows(
        comm, global_indices_sorted, counts, root=root
    )

    return QueryRouting(
        local_positions=local_positions,
        local_global_indices=local_global_indices,
        counts=counts,
    )
