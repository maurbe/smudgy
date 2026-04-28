"""Unit tests for KD-tree utilities in smudgy.utils."""

import numpy as np
import pytest

from smudgy.utils import build_kdtree, query_kdtree


def test_build_kdtree_basic():
    """Test that building a KD-tree with default parameters does not raise errors."""
    points = np.random.rand(10, 2)
    tree = build_kdtree(points)
    assert hasattr(tree, "query")
    assert tree.n == 10


@pytest.mark.parametrize("boxsize", [None, 1.0, [1.0, 1.0]])
def test_build_kdtree_periodic(boxsize):
    """Test that building a KD-tree with periodic boundaries does not raise errors."""
    points = np.random.rand(5, 2)
    tree = build_kdtree(points, boxsize=boxsize)
    assert hasattr(tree, "query")


def test_query_kdtree_basic():
    """Test that querying the KD-tree returns expected shapes and types."""
    points = np.random.rand(10, 2)
    tree = build_kdtree(points)
    query_points = np.random.rand(3, 2)
    dists, inds = query_kdtree(tree, query_points, k=2)
    assert dists.shape == (3, 2)
    assert inds.shape == (3, 2)
    assert np.issubdtype(dists.dtype, np.floating)
    assert np.issubdtype(inds.dtype, np.integer)


def test_query_kdtree_k_equals_n():
    """Test that querying with k equal to the number of points returns all points as neighbors."""
    points = np.random.rand(6, 2)
    tree = build_kdtree(points)
    dists, inds = query_kdtree(tree, points, k=6)
    assert dists.shape == (6, 6)
    assert inds.shape == (6, 6)


def test_query_kdtree_invalid_k():
    """Test that querying with k > n returns inf distances and appropriate indices."""
    points = np.random.rand(5, 2)
    tree = build_kdtree(points)
    query_points = np.random.rand(2, 2)
    dists, inds = query_kdtree(tree, query_points, k=10)
    # For k > n, cKDTree.query returns inf for missing neighbors
    assert np.any(np.isinf(dists)), "Distances should contain inf when k > n."
