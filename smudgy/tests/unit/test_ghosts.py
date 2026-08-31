"""Unit tests for `smudgy.ghosts` helper functions (pure, MPI-free).

Focus: the periodic-image bounding-box overlap logic (the part of Step 2
that's genuinely new relative to a non-periodic ghost exchange) and the
initial-radius/wrapping helpers, in isolation from any real MPI exchange.
"""

from __future__ import annotations

import numpy as np
import pytest

from smudgy.ghosts import (
    _boxes_overlap,
    _initial_radius,
    _overlap_shifts,
    _shift_vectors,
    wrap_periodic,
)


class TestShiftVectors:
    @pytest.mark.parametrize("dim,expected_count", [(1, 3), (2, 9), (3, 27)])
    def test_periodic_shift_counts(self, dim, expected_count):
        shifts = _shift_vectors(dim, periodic=True)
        assert shifts.shape == (expected_count, dim)
        # the zero vector must be included
        assert np.any(np.all(shifts == 0, axis=1))

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_non_periodic_returns_only_zero_vector(self, dim):
        shifts = _shift_vectors(dim, periodic=False)
        assert shifts.shape == (1, dim)
        assert np.all(shifts == 0)


class TestBoxesOverlap:
    def test_disjoint(self):
        assert not _boxes_overlap(
            np.array([0.0]), np.array([1.0]), np.array([2.0]), np.array([3.0])
        )

    def test_nested(self):
        assert _boxes_overlap(
            np.array([0.0]), np.array([10.0]), np.array([2.0]), np.array([3.0])
        )

    def test_identical(self):
        assert _boxes_overlap(
            np.array([0.0]), np.array([1.0]), np.array([0.0]), np.array([1.0])
        )

    def test_touching_at_a_point_counts_as_overlap(self):
        # closed-interval convention: touching boundaries overlap
        assert _boxes_overlap(
            np.array([0.0]), np.array([1.0]), np.array([1.0]), np.array([2.0])
        )

    def test_multidim(self):
        assert _boxes_overlap(
            np.array([0.0, 0.0]), np.array([1.0, 1.0]),
            np.array([0.5, 5.0]), np.array([1.5, 6.0]),
        ) is False  # overlaps on x, disjoint on y -> no overlap overall


class TestOverlapShifts:
    def test_non_periodic_reduces_to_single_direct_check(self):
        shifts = _shift_vectors(1, periodic=False)
        hits = _overlap_shifts(
            np.array([0.0]), np.array([1.0]), np.array([5.0]), np.array([6.0]),
            None, shifts,
        )
        assert hits == []

        hits = _overlap_shifts(
            np.array([0.0]), np.array([1.0]), np.array([0.5]), np.array([1.5]),
            None, shifts,
        )
        assert len(hits) == 1
        assert np.all(hits[0] == 0)

    def test_periodic_finds_wraparound_overlap_missed_by_direct_check(self):
        """1D box near x=0 and a target box near x=boxsize=1.0 -- only
        overlap once the target is shifted by -boxsize."""
        boxsize = np.array([1.0])
        shifts = _shift_vectors(1, periodic=True)

        query_min, query_max = np.array([-0.05]), np.array([0.05])
        target_min, target_max = np.array([0.98]), np.array([1.0])

        # sanity: no overlap without shifting
        assert not _boxes_overlap(query_min, query_max, target_min, target_max)

        hits = _overlap_shifts(query_min, query_max, target_min, target_max, boxsize, shifts)
        assert len(hits) >= 1
        assert any(s[0] == -1 for s in hits)

    def test_periodic_disjoint_even_after_wraparound(self):
        boxsize = np.array([10.0])
        shifts = _shift_vectors(1, periodic=True)
        hits = _overlap_shifts(
            np.array([4.0]), np.array([5.0]), np.array([1.0]), np.array([2.0]),
            boxsize, shifts,
        )
        assert hits == []


class TestInitialRadius:
    def test_matches_hand_computed_value(self):
        # 1D unit interval, 10 particles -> volume=1, density=10,
        # unit-ball-volume(1D)=2 -> r_k = (num_neighbors / (10*2))**1
        r = _initial_radius(
            n_local=10, local_min=np.array([0.0]), local_max=np.array([1.0]),
            num_neighbors=4, dim=1,
            domain_min=np.array([0.0]), domain_max=np.array([1.0]),
            boxsize=None, periodic=False, safety_factor=1.0,
        )
        expected = 4 / (10 * 2.0)
        assert r == pytest.approx(expected)

    def test_safety_factor_scales_linearly(self):
        kwargs = dict(
            n_local=10, local_min=np.array([0.0]), local_max=np.array([1.0]),
            num_neighbors=4, dim=1,
            domain_min=np.array([0.0]), domain_max=np.array([1.0]),
            boxsize=None, periodic=False,
        )
        r1 = _initial_radius(safety_factor=1.0, **kwargs)
        r2 = _initial_radius(safety_factor=2.0, **kwargs)
        assert r2 == pytest.approx(2 * r1)

    @pytest.mark.parametrize("n_local", [0, 1])
    def test_fallback_for_too_few_local_particles(self, n_local):
        r = _initial_radius(
            n_local=n_local, local_min=np.zeros(3), local_max=np.zeros(3),
            num_neighbors=8, dim=3,
            domain_min=np.zeros(3), domain_max=np.array([4.0, 4.0, 4.0]),
            boxsize=None, periodic=False,
        )
        assert np.isfinite(r)
        assert r > 0

    def test_periodic_fallback_uses_boxsize(self):
        r = _initial_radius(
            n_local=0, local_min=np.zeros(2), local_max=np.zeros(2),
            num_neighbors=8, dim=2,
            domain_min=np.zeros(2), domain_max=np.array([1.0, 1.0]),
            boxsize=np.array([2.0, 2.0]), periodic=True,
        )
        assert r == pytest.approx(2.0 / 4.0)


class TestWrapPeriodic:
    def test_identity_when_not_periodic(self):
        positions = np.array([[-0.5, 3.0], [1.5, -2.0]])
        result = wrap_periodic(positions, None)
        np.testing.assert_array_equal(result, positions)

    def test_wraps_into_box(self):
        positions = np.array([[-0.1, 1.0], [1.1, 0.5]])
        result = wrap_periodic(positions, np.array([1.0, 1.0]))
        np.testing.assert_allclose(result, [[0.9, 0.0], [0.1, 0.5]], atol=1e-6)

    def test_already_in_box_unchanged(self):
        positions = np.array([[0.3, 0.7]])
        result = wrap_periodic(positions, np.array([1.0, 1.0]))
        np.testing.assert_allclose(result, positions)
