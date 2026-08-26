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

from smudgy.decomposition import hilbert_encode


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
