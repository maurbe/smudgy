"""Hilbert-curve spatial domain decomposition.

Step 1 of the MPI domain-decomposition roadmap: partitions particles across
MPI ranks into count-balanced, spatially-contiguous chunks (sorted by
Hilbert-curve code), then redistributes positions/weights via
`execution._scatterv_rows`. Produces a `DecompositionInfo` side artifact
(local chunk + provenance) for later steps (ghost exchange, local-only
compute, gather-to-root) to build on.

This module is currently opt-in and self-contained: nothing else in the
package reads `DecompositionInfo` yet (see `PointCloud.decompose`).
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

    """

    local_positions: np.ndarray = None
    local_weights: np.ndarray = None
    local_global_indices: np.ndarray = None
    counts: np.ndarray = None
    domain_min: np.ndarray = None
    domain_max: np.ndarray = None


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
    max_index = num_bins - 1
    # Scale by num_bins (not max_index) so bins are evenly filled -- frac==1.0
    # (a clipped position exactly at domain_max) is the only case that can
    # reach num_bins itself, which the final clip pulls back down to max_index.
    coords = np.clip(np.floor(frac * float(num_bins)), 0, float(max_index)).astype(
        np.uint64
    )
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

    if rank == root:
        n = positions.shape[0]
        codes = hilbert_encode(positions, domain_min, domain_max, periodic=periodic)
        order = np.argsort(codes, kind="stable")
        counts = execution._balanced_counts(n, size)
        positions_sorted = np.ascontiguousarray(positions[order])
        weights_sorted = np.ascontiguousarray(weights[order])
        global_indices_sorted = np.ascontiguousarray(order.astype(np.int64))
    else:
        counts = positions_sorted = weights_sorted = global_indices_sorted = None

    counts = comm.bcast(counts if rank == root else None, root=root)

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
    )
