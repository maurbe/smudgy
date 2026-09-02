"""Unit tests for `smudgy.decomposition.hilbert_encode`.

Focus: numerical correctness of the Hilbert-curve encoding itself (a pure,
MPI-free function), checked against an independent reference implementation
and against the defining mathematical properties of a space-filling curve
(bijective; consecutive codes in sorted order are grid-adjacent cells) --
rather than against the encoder's own internal constants, which would be a
tautological test.
"""

from __future__ import annotations

import numpy as np
import pytest

from smudgy.decomposition import _partition_boundary_codes, hilbert_encode


def _xy2d_reference(n: int, x: int, y: int) -> int:
    """Standard, independently-known-correct 2D Hilbert curve algorithm.

    n must be a power of 2. Used only as a reference to check `hilbert_encode`
    against -- deliberately not shared code with the implementation under test.
    """
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s //= 2
    return d


class TestHilbertEncode2DReference:
    def test_matches_reference_ordering_exactly(self):
        """Full 2D grid at a small bit depth: codes must match the reference
        implementation's *ordering* exactly (not just be some valid permutation)."""
        bits = 4
        n = 1 << bits
        xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        xs, ys = xs.ravel(), ys.ravel()
        positions = np.stack([xs, ys], axis=1).astype(np.float64)

        codes = hilbert_encode(
            positions,
            domain_min=np.array([0.0, 0.0]),
            domain_max=np.array([float(n), float(n)]),
            periodic=False,
            bits_per_dim=bits,
        )
        ref = np.array([_xy2d_reference(n, int(x), int(y)) for x, y in zip(xs, ys)])

        np.testing.assert_array_equal(codes.astype(np.int64), ref)


class TestHilbertEncodeProperties:
    def test_2d_bijective(self):
        bits = 4
        n = 1 << bits
        xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        positions = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
        codes = hilbert_encode(
            positions, np.array([0.0, 0.0]), np.array([float(n)] * 2), bits_per_dim=bits
        )
        assert len(set(codes.tolist())) == n * n

    def test_3d_bijective_and_locality_preserving(self):
        """Exhaustive check of the defining space-filling-curve property: every
        pair of consecutive points in sorted Hilbert order must be a
        grid-adjacent cell (Chebyshev distance exactly 1). This is a strong,
        independent correctness check that doesn't require knowing the
        'correct' 3D ordering in advance."""
        bits = 4
        n = 1 << bits
        gx, gy, gz = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
        positions = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(
            np.float64
        )
        codes = hilbert_encode(
            positions, np.array([0.0] * 3), np.array([float(n)] * 3), bits_per_dim=bits
        )

        assert len(set(codes.tolist())) == n**3

        order = np.argsort(codes, kind="stable")
        sorted_positions = positions[order]
        chebyshev = np.abs(np.diff(sorted_positions, axis=0)).max(axis=1)
        assert np.all(chebyshev == 1)

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_origin_corner_maps_to_zero(self, dim):
        point = np.zeros((1, dim))
        code = hilbert_encode(point, np.zeros(dim), np.ones(dim))
        assert code[0] == 0

    def test_shape_and_dtype(self):
        positions = np.random.default_rng(0).uniform(size=(50, 3))
        codes = hilbert_encode(positions, np.zeros(3), np.ones(3))
        assert codes.shape == (50,)
        assert codes.dtype == np.uint64

    def test_degenerate_axis_does_not_raise_or_produce_nan(self):
        """domain_min == domain_max on one axis must not divide by zero."""
        positions = np.array([[0.5, 1.0], [0.5, 2.0], [0.5, 3.0]])
        codes = hilbert_encode(
            positions, domain_min=np.array([0.5, 0.0]), domain_max=np.array([0.5, 4.0])
        )
        assert np.all(np.isfinite(codes.astype(np.float64)))
        # the degenerate axis contributes nothing -- codes must still be distinct
        # since the non-degenerate axis varies
        assert len(set(codes.tolist())) == 3


class TestHilbertEncodePeriodicity:
    def test_periodic_wraps_not_clips(self):
        """A point just outside [domain_min, domain_max) must, under
        periodic=True, land near its true periodic image -- not at the
        clipped boundary the way periodic=False would place it."""
        # -0.001 wraps to ~0.999, landing on top of the second point
        positions = np.array([[-0.001, 0.5], [0.999, 0.5]])
        domain_min, domain_max = np.array([0.0, 0.0]), np.array([1.0, 1.0])

        periodic_codes = hilbert_encode(
            positions, domain_min, domain_max, periodic=True, bits_per_dim=8
        )
        clipped_codes = hilbert_encode(
            positions, domain_min, domain_max, periodic=False, bits_per_dim=8
        )

        # periodic: both points collapse to (~)the same quantized bin
        assert periodic_codes[0] == periodic_codes[1]
        # non-periodic (clipped): -0.001 is pinned to the domain_min edge,
        # far from 0.999 -- codes must differ
        assert clipped_codes[0] != clipped_codes[1]


class TestPartitionBoundaryCodes:
    """Unit tests for `_partition_boundary_codes` (Step 4b): the (size+1,)
    cut-point array `route_query_positions` uses to route arbitrary query
    positions consistently with the *actual* particle partition, without
    needing MPI (a pure function of already-sorted codes + counts)."""

    def test_matches_actual_partition_cut_points(self):
        codes_sorted = np.array([0, 5, 5, 9, 20, 21, 40, 41, 41, 99], dtype=np.uint64)
        counts = np.array([3, 3, 4], dtype=np.int64)  # ranks own [0:3), [3:6), [6:10)
        boundaries = _partition_boundary_codes(codes_sorted, counts)

        assert boundaries.dtype == np.uint64
        assert boundaries.shape == (4,)
        assert boundaries[0] == 0
        assert boundaries[1] == codes_sorted[3]  # first code of rank 1's chunk
        assert boundaries[2] == codes_sorted[6]  # first code of rank 2's chunk
        assert boundaries[3] == np.iinfo(np.uint64).max

    def test_monotonically_non_decreasing(self):
        rng = np.random.default_rng(0)
        codes_sorted = np.sort(rng.integers(0, 10_000, size=200).astype(np.uint64))
        counts = np.array([40, 60, 100], dtype=np.int64)
        boundaries = _partition_boundary_codes(codes_sorted, counts)
        assert np.all(np.diff(boundaries.astype(np.float64)) >= 0)

    def test_empty_rank_gets_zero_width_interval(self):
        """N < P: a rank with 0 particles must not silently claim any code
        range -- its interval collapses to width 0 (shared boundary with the
        next non-empty rank), so a query point can never route there."""
        codes_sorted = np.array([1, 2], dtype=np.uint64)
        counts = np.array([1, 0, 1], dtype=np.int64)  # rank 1 is empty
        boundaries = _partition_boundary_codes(codes_sorted, counts)

        assert boundaries[1] == boundaries[2]  # rank 1's interval is empty

    def test_trailing_empty_ranks_all_collapse_to_max(self):
        codes_sorted = np.array([1, 2, 3], dtype=np.uint64)
        counts = np.array([3, 0, 0], dtype=np.int64)
        boundaries = _partition_boundary_codes(codes_sorted, counts)

        max_code = np.iinfo(np.uint64).max
        assert boundaries[1] == max_code
        assert boundaries[2] == max_code
        assert boundaries[3] == max_code

    def test_zero_particles_total(self):
        """N == 0 (degenerate, but must not crash): every rank's interval is
        empty, and the array is still well-formed."""
        codes_sorted = np.array([], dtype=np.uint64)
        counts = np.array([0, 0], dtype=np.int64)
        boundaries = _partition_boundary_codes(codes_sorted, counts)

        assert boundaries.shape == (3,)
        assert boundaries[0] == 0
        max_code = np.iinfo(np.uint64).max
        assert boundaries[1] == max_code
        assert boundaries[2] == max_code


class _FakeSingleRankComm:
    """Minimal stand-in for `mpi4py.MPI.Comm` covering only what a size==1
    call needs. `bcast`/`Scatterv`/etc. deliberately raise if reached, since
    the whole point of `hilbert_partition_and_scatter`/`route_query_positions`'s
    size==1 fast path is to never touch any of them."""

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def bcast(self, *args, **kwargs):
        raise AssertionError("bcast must not be called on the size==1 fast path")

    def Scatterv(self, *args, **kwargs):
        raise AssertionError("Scatterv must not be called on the size==1 fast path")


class TestSizeOneFastPath:
    """`hilbert_partition_and_scatter`/`route_query_positions` skip Hilbert
    encoding/sorting/scattering entirely at size==1 (no other rank to
    partition against) -- see their docstrings/inline comments. These tests
    confirm the fast path is actually taken (not just that results happen to
    look right), and that it produces the identity permutation in original
    input order."""

    def test_hilbert_partition_and_scatter_skips_encode_and_sort(self, monkeypatch):
        from smudgy import decomposition

        def _boom(*args, **kwargs):
            raise AssertionError("hilbert_encode must not be called at size==1")

        monkeypatch.setattr(decomposition, "hilbert_encode", _boom)

        rng = np.random.default_rng(0)
        n, dim = 12, 3
        positions = rng.uniform(0, 1, size=(n, dim)).astype(np.float32)
        weights = rng.uniform(0.5, 1.5, size=n).astype(np.float32)

        info = decomposition.hilbert_partition_and_scatter(
            _FakeSingleRankComm(),
            positions,
            weights,
            domain_min=positions.min(axis=0),
            domain_max=positions.max(axis=0),
        )

        np.testing.assert_array_equal(info.local_positions, positions)
        np.testing.assert_array_equal(info.local_weights, weights)
        np.testing.assert_array_equal(info.local_global_indices, np.arange(n))
        np.testing.assert_array_equal(info.root_order, np.arange(n))
        np.testing.assert_array_equal(info.counts, [n])
        assert info.boundary_codes[0] == 0
        assert info.boundary_codes[-1] == np.iinfo(np.uint64).max

    def test_hilbert_partition_and_scatter_ignores_root_at_size_one(self, monkeypatch):
        """A caller-supplied non-default `root` must not matter at size==1:
        `comm.Get_rank()` is always 0 there, so gating on `rank == root`
        (instead of `size == 1`) would take the wrong branch and try to
        `comm.bcast(None, ...)` from a rank that doesn't exist -- this is the
        latent crash the size==1 gating fixes as a side effect."""
        from smudgy import decomposition

        monkeypatch.setattr(
            decomposition,
            "hilbert_encode",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("unexpected call")),
        )

        rng = np.random.default_rng(0)
        n, dim = 5, 2
        positions = rng.uniform(0, 1, size=(n, dim)).astype(np.float32)
        weights = np.ones(n, dtype=np.float32)

        info = decomposition.hilbert_partition_and_scatter(
            _FakeSingleRankComm(),
            positions,
            weights,
            domain_min=positions.min(axis=0),
            domain_max=positions.max(axis=0),
            root=1,  # non-default, non-existent rank in a 1-process comm
        )
        np.testing.assert_array_equal(info.local_positions, positions)

    def test_route_query_positions_skips_encode_and_searchsorted(self, monkeypatch):
        from smudgy import decomposition

        rng = np.random.default_rng(1)
        n, dim = 12, 3
        positions = rng.uniform(0, 1, size=(n, dim)).astype(np.float32)
        weights = np.ones(n, dtype=np.float32)
        info = decomposition.hilbert_partition_and_scatter(
            _FakeSingleRankComm(),
            positions,
            weights,
            domain_min=positions.min(axis=0),
            domain_max=positions.max(axis=0),
        )

        def _boom(*args, **kwargs):
            raise AssertionError("hilbert_encode must not be called at size==1")

        monkeypatch.setattr(decomposition, "hilbert_encode", _boom)

        query = rng.uniform(0, 1, size=(5, dim)).astype(np.float32)
        routing = decomposition.route_query_positions(
            _FakeSingleRankComm(), info, query,
        )

        np.testing.assert_array_equal(routing.local_positions, query)
        np.testing.assert_array_equal(routing.local_global_indices, np.arange(5))
        np.testing.assert_array_equal(routing.counts, [5])
