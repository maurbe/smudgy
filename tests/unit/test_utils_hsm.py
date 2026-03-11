import numpy as np
import pytest
from smudgy.utils import build_kdtree, compute_hsm, compute_hsm_tensor, project_hsm_tensor_to_2d

def test_compute_hsm_basic():
    points = np.random.rand(10, 2)
    tree = build_kdtree(points)
    hsm, nn_inds, nn_dists = compute_hsm(tree, num_neighbors=3)
    assert hsm.shape == (10,)
    assert nn_inds.shape == (10, 3)
    assert nn_dists.shape == (10, 3)
    assert np.all(hsm > 0)


def test_compute_hsm_query_positions():
    points = np.random.rand(10, 2)
    tree = build_kdtree(points)
    queries = np.random.rand(5, 2)
    hsm, nn_inds, nn_dists = compute_hsm(tree, num_neighbors=3, query_positions=queries)
    assert hsm.shape == (5,)
    assert nn_inds.shape == (5, 3)
    assert nn_dists.shape == (5, 3)


def test_compute_hsm_tensor_basic():
    points = np.random.rand(10, 3)
    weights = np.random.rand(10)
    tree = build_kdtree(points)
    H, eigvals, eigvecs, nn_inds, nn_dists, rel_coords = compute_hsm_tensor(tree, weights, num_neighbors=3)
    assert H.shape == (10, 3, 3)
    assert eigvals.shape == (10, 3)
    assert eigvecs.shape == (10, 3, 3)
    assert nn_inds.shape == (10, 3)
    assert nn_dists.shape == (10, 3)
    assert rel_coords.shape == (10, 3, 3)


def test_compute_hsm_tensor_2d():
    points = np.random.rand(10, 2)
    weights = np.random.rand(10)
    tree = build_kdtree(points)
    H, eigvals, eigvecs, nn_inds, nn_dists, rel_coords = compute_hsm_tensor(tree, weights, num_neighbors=3)
    assert H.shape == (10, 2, 2)
    assert eigvals.shape == (10, 2)
    assert eigvecs.shape == (10, 2, 2)
    assert nn_inds.shape == (10, 3)
    assert nn_dists.shape == (10, 3)
    assert rel_coords.shape == (10, 3, 2)


def test_project_hsm_tensor_to_2d_plane():
    h_tensor = np.random.rand(5, 3, 3)
    h_tensor_2d, eigvals, eigvecs = project_hsm_tensor_to_2d(h_tensor, plane="xy")
    assert h_tensor_2d.shape == (5, 2, 2)
    assert eigvals.shape == (5, 2)
    assert eigvecs.shape == (5, 2, 2)


def test_project_hsm_tensor_to_2d_basis():
    h_tensor = np.random.rand(3, 3, 3)
    basis = ([1, 0, 0], [0, 1, 0])
    h_tensor_2d, eigvals, eigvecs = project_hsm_tensor_to_2d(h_tensor, basis=basis)
    assert h_tensor_2d.shape == (3, 2, 2)
    assert eigvals.shape == (3, 2)
    assert eigvecs.shape == (3, 2, 2)


def test_project_hsm_tensor_to_2d_error():
    h_tensor = np.random.rand(2, 3, 3)
    with pytest.raises(ValueError):
        project_hsm_tensor_to_2d(h_tensor)
    with pytest.raises(ValueError):
        project_hsm_tensor_to_2d(h_tensor, plane="abc")
    with pytest.raises(ValueError):
        project_hsm_tensor_to_2d(h_tensor, plane="xy", basis=([1,0,0],[0,1,0]))
