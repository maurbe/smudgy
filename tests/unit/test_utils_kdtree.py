import numpy as np
import pytest
from smudgy.utils import build_kdtree, query_kdtree

def test_build_kdtree_basic():
    points = np.random.rand(10, 2)
    tree = build_kdtree(points)
    assert hasattr(tree, 'query')
    assert tree.n == 10

@pytest.mark.parametrize('boxsize', [None, 1.0, [1.0, 1.0]])
def test_build_kdtree_periodic(boxsize):
    points = np.random.rand(5, 2)
    tree = build_kdtree(points, boxsize=boxsize)
    assert hasattr(tree, 'query')


def test_query_kdtree_basic():
    points = np.random.rand(10, 2)
    tree = build_kdtree(points)
    query_points = np.random.rand(3, 2)
    dists, inds = query_kdtree(tree, query_points, k=2)
    assert dists.shape == (3, 2)
    assert inds.shape == (3, 2)
    assert np.issubdtype(dists.dtype, np.floating)
    assert np.issubdtype(inds.dtype, np.integer)


def test_query_kdtree_k_equals_n():
    points = np.random.rand(6, 2)
    tree = build_kdtree(points)
    dists, inds = query_kdtree(tree, points, k=6)
    assert dists.shape == (6, 6)
    assert inds.shape == (6, 6)


def test_query_kdtree_invalid_k():
    points = np.random.rand(5, 2)
    tree = build_kdtree(points)
    query_points = np.random.rand(2, 2)
    dists, inds = query_kdtree(tree, query_points, k=10)
    # For k > n, cKDTree.query returns inf for missing neighbors
    assert np.any(np.isinf(dists)), "Distances should contain inf when k > n."
