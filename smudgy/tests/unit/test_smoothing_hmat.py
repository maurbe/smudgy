"""Tests for the covariant smoothing tensors of `compute_hmat` (Marinho 2021).

Every property test runs against both backends; the last section compares the
two backends directly on random point clouds.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from smudgy.backend.numpy import smoothing as np_smoothing
from smudgy.backend.taichi import smoothing as ti_smoothing

BACKENDS = {"numpy": np_smoothing, "taichi": ti_smoothing}


@pytest.fixture(params=list(BACKENDS))
def backend(request):
    """Backend module under test."""
    return BACKENDS[request.param]


def _hmat(backend, rel, weights=None, query=None, **kwargs):
    """Run `compute_hmat` for neighbor offsets `rel` of shape (N, K, D) or (K, D)."""
    rel = np.asarray(rel, dtype=np.float32)
    if rel.ndim == 2:
        rel = rel[np.newaxis]
    n, k, dim = rel.shape
    query = np.zeros((n, dim), dtype=np.float32) if query is None else query
    weights = np.ones((n, k), dtype=np.float32) if weights is None else weights
    return backend.compute_hmat(
        query_positions=query,
        neighbor_positions=query[:, np.newaxis, :] + rel,
        neighbor_weights=weights,
        **kwargs,
    )


def _random_offsets(dim, n=64, k=16, seed=0, scales=None):
    rng = np.random.default_rng(seed)
    rel = rng.normal(size=(n, k, dim))
    if scales is not None:
        rel *= np.asarray(scales)[:dim]
    return rel.astype(np.float32)


def _mahalanobis(H, rel):
    """|H^-1 r| for each neighbor, shape (N, K)."""
    xi = np.einsum("nij,nkj->nki", np.linalg.inv(H.astype(np.float64)), rel)
    return np.linalg.norm(xi, axis=-1)


# =============================================================================
# Analytic configurations
# =============================================================================


@pytest.mark.parametrize("dim", [2, 3])
def test_isotropic_limit(backend, dim):
    """+-r along every axis: H = r I, the isotropic smoothing length."""
    r = 0.7
    rel = np.concatenate([r * np.eye(dim), -r * np.eye(dim)])
    H, axes, _, _ = _hmat(backend, rel)
    np.testing.assert_allclose(H[0], r * np.eye(dim), atol=1e-6)
    np.testing.assert_allclose(axes[0], r, rtol=1e-6)


def test_1d_axis_is_farthest_neighbor(backend):
    """1D: the single axis is the distance to the farthest neighbor."""
    rel = np.array([[0.0], [0.1], [-0.4], [0.25]])
    H, axes, eigvecs, _ = _hmat(backend, rel)
    np.testing.assert_allclose(H[0], [[0.4]], rtol=1e-6)
    np.testing.assert_allclose(axes[0], [0.4], rtol=1e-6)
    np.testing.assert_array_equal(eigvecs[0], [[1.0]])


@pytest.mark.parametrize("dim", [2, 3])
def test_analytic_ellipsoid(backend, dim):
    """+-a e1, +-b e2 (+-c e3): axes (a, b, c) along the coordinate axes."""
    lengths = np.array([3.0, 2.0, 1.0])[:dim]
    rel = np.concatenate([np.diag(lengths), -np.diag(lengths)])
    H, axes, eigvecs, _ = _hmat(backend, rel)
    np.testing.assert_allclose(axes[0], lengths, rtol=1e-5)
    np.testing.assert_allclose(eigvecs[0], np.eye(dim), atol=1e-5)
    np.testing.assert_allclose(H[0], np.diag(lengths), atol=1e-5)


@pytest.mark.parametrize("dim", [2, 3])
def test_rotation_equivariance(backend, dim):
    """Rotating the neighbors rotates H: H(R r) = R H(r) R^T."""
    rng = np.random.default_rng(1)
    rel = _random_offsets(dim, n=16, scales=[1.0, 0.4, 0.2])
    R, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    R *= np.sign(np.linalg.det(R))
    H, _, _, _ = _hmat(backend, rel)
    H_rot, _, _, _ = _hmat(backend, rel @ R.T)
    np.testing.assert_allclose(H_rot, R @ H @ R.T, atol=1e-4 * np.abs(H).max())


@pytest.mark.parametrize(
    ("dim", "points", "expected"),
    [
        (2, [[1, 0], [-1, 0], [0.5, 0], [-0.3, 0]], [1.0, 0.1]),  # collinear
        (3, [[1, 0, 0], [-1, 0, 0], [0.5, 0, 0]], [1.0, 0.1, 0.1]),  # collinear
        (
            3,
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
            [1.0, 1.0, 0.1],
        ),  # coplanar
    ],
)
def test_ratio_clamp_on_degenerate_sets(backend, dim, points, expected):
    """Collinear/coplanar neighbors: short axes are clamped to h_max / ratio_max."""
    H, axes, _, _ = _hmat(backend, np.array(points, dtype=np.float32), ratio_max=10.0)
    np.testing.assert_allclose(axes[0], expected, rtol=1e-5)
    assert np.all(np.isfinite(H))
    assert abs(np.linalg.det(H[0])) > 0


def test_min_axis_floor(backend):
    """min_axis raises short axes and leaves longer ones untouched."""
    points = np.array([[1, 0], [-1, 0], [0.5, 0]], dtype=np.float32)
    _, axes, _, _ = _hmat(backend, points, min_axis=0.3)
    np.testing.assert_allclose(axes[0], [1.0, 0.3], rtol=1e-5)
    # a floor below every axis changes nothing
    _, axes_ref, _, _ = _hmat(backend, points)
    _, axes_low, _, _ = _hmat(backend, points, min_axis=1e-3)
    np.testing.assert_array_equal(axes_low, axes_ref)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_volume_preserving(backend, dim):
    """volume_preserving: det H = hsml^D with the hull tensor's shape."""
    rel = _random_offsets(dim, scales=[1.0, 0.3, 0.1])
    hsml = np.linspace(0.5, 2.0, rel.shape[0]).astype(np.float32)
    H, axes, _, _ = _hmat(backend, rel, volume_preserving=True, hsml=hsml)
    np.testing.assert_allclose(np.linalg.det(H), hsml**dim, rtol=1e-4)
    np.testing.assert_allclose(np.prod(axes, axis=1), hsml**dim, rtol=1e-4)
    # shape unchanged: same axis ratios as the hull-scaled tensor
    _, axes_hull, _, _ = _hmat(backend, rel)
    np.testing.assert_allclose(
        axes / axes[:, :1], axes_hull / axes_hull[:, :1], rtol=1e-4
    )


def test_input_validation(backend):
    """Invalid ratio_max and hsml arguments raise."""
    rel = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="ratio_max"):
        _hmat(backend, rel, ratio_max=0.5)
    with pytest.raises(ValueError, match="requires hsml"):
        _hmat(backend, rel, volume_preserving=True)
    with pytest.raises(ValueError, match="hsml must have shape"):
        _hmat(backend, rel, volume_preserving=True, hsml=np.ones(3))


def test_periodic_wrapping(backend):
    """Neighbors across the box edge give the same H as their unwrapped images."""
    boxsize = np.ones(2, dtype=np.float32)
    query = np.array([[0.02, 0.97]], dtype=np.float32)
    rel = np.array(
        [[[-0.05, 0.01], [0.03, 0.06], [-0.04, -0.02], [0.02, 0.05], [0.0, 0.0]]],
        dtype=np.float32,
    )
    wrapped = np.mod(query[:, np.newaxis, :] + rel, 1.0)
    weights = np.ones(rel.shape[:2], dtype=np.float32)
    H_pbc, _, _, rel_pbc = backend.compute_hmat(query, wrapped, weights, boxsize)
    H_ref, _, _, _ = _hmat(backend, rel, query=query)
    np.testing.assert_allclose(rel_pbc, rel, atol=1e-6)
    np.testing.assert_allclose(H_pbc, H_ref, atol=1e-5)


def test_all_neighbors_at_query(backend):
    """Sigma = 0: H = 0 (no NaNs), and min_axis restores an invertible tensor."""
    rel = np.zeros((4, 3), dtype=np.float32)
    H, axes, _, _ = _hmat(backend, rel)
    np.testing.assert_array_equal(H, 0.0)
    np.testing.assert_array_equal(axes, 0.0)
    H, _, _, _ = _hmat(backend, rel, min_axis=0.1)
    np.testing.assert_allclose(H[0], 0.1 * np.eye(3), atol=1e-7)


# =============================================================================
# Invariants on random neighbor sets
# =============================================================================


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize(
    "scales", [[1.0, 1.0, 1.0], [1.0, 0.2, 0.05], [1.0, 1e-3, 1e-3]]
)
def test_hull_property(backend, dim, scales):
    """Every neighbor lies inside the support ellipsoid, the farthest on it."""
    rel = _random_offsets(dim, scales=scales)
    H, _, _, _ = _hmat(backend, rel)
    xi = _mahalanobis(H, rel)
    assert xi.max() <= 1.0 + 1e-4
    np.testing.assert_allclose(xi.max(axis=1), 1.0, rtol=1e-4)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_eigendecomposition_invariants(backend, dim):
    """H = V diag(axes) V^T with orthonormal, sign-normalized V and sorted axes."""
    rel = _random_offsets(dim, scales=[1.0, 0.5, 0.2])
    weights = np.random.default_rng(2).uniform(0.5, 2.0, rel.shape[:2])
    H, axes, V, _ = _hmat(backend, rel, weights=weights.astype(np.float32))
    eye = np.broadcast_to(np.eye(dim), V.shape)
    np.testing.assert_allclose(np.swapaxes(V, 1, 2) @ V, eye, atol=1e-5)
    np.testing.assert_allclose(
        H, np.einsum("nij,nj,nkj->nik", V, axes, V), atol=1e-5 * axes.max()
    )
    np.testing.assert_allclose(H, np.swapaxes(H, 1, 2), atol=1e-6 * axes.max())
    assert np.all(np.diff(axes, axis=1) <= 0), "axes must be sorted descending"
    # sign convention: largest-magnitude component of each eigenvector is positive
    idx = np.argmax(np.abs(V), axis=1)
    assert np.all(np.take_along_axis(V, idx[:, np.newaxis, :], axis=1) > 0)
    # H only depends on the shape of the weights, not their overall scale
    H_scaled, _, _, _ = _hmat(backend, rel, weights=(7.3 * weights).astype(np.float32))
    np.testing.assert_allclose(H_scaled, H, rtol=1e-5, atol=1e-6 * axes.max())


# =============================================================================
# Numpy vs Taichi
# =============================================================================


def _filament_positions(dim, n, thickness, n_filaments=4, seed=42):
    """Uniform in x; the other coordinates cluster around a few filament axes.

    thickness=None gives a uniform cloud. Thin filaments make the neighborhoods
    strongly anisotropic: 3e-3 stays below the default ratio clamp, 1e-4 hits it.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n, dim))
    if thickness is not None and dim > 1:
        centers = rng.uniform(0, 1, (n_filaments, dim - 1))
        members = rng.integers(n_filaments, size=n)
        x[:, 1:] = centers[members] + rng.normal(0, thickness, (n, dim - 1))
        x[:, 1:] %= 1.0
    return x.astype(np.float32)


def _knn_cloud(dim, thickness, periodic, n=3000, k=16):
    x = _filament_positions(dim, n, thickness)
    boxsize = np.ones(dim, dtype=np.float32) if periodic else None
    dists, idx = cKDTree(x, boxsize=boxsize).query(x, k)
    w = np.random.default_rng(0).uniform(0.5, 1.5, n).astype(np.float32)
    return x, x[idx], w[idx], boxsize, dists[:, -1].astype(np.float32)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize(
    "thickness", [None, 3e-3, 1e-4], ids=["uniform", "filament", "thin_filament"]
)
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("volume_preserving", [False, True])
def test_backends_agree(dim, thickness, periodic, volume_preserving):
    """Numpy and Taichi agree on H, its axes and (separated) eigenvectors."""
    x, nb_pos, nb_w, boxsize, hsml = _knn_cloud(dim, thickness, periodic)
    kwargs = dict(
        query_positions=x,
        neighbor_positions=nb_pos,
        neighbor_weights=nb_w,
        boxsize=boxsize,
        volume_preserving=volume_preserving,
        hsml=hsml if volume_preserving else None,
    )
    H_np, ax_np, V_np, rel_np = np_smoothing.compute_hmat(**kwargs)
    H_ti, ax_ti, V_ti, rel_ti = ti_smoothing.compute_hmat(**kwargs)

    np.testing.assert_allclose(rel_ti, rel_np, atol=1e-6)
    # rtol 5e-4: float32 relative coordinates can differ by an ulp between the
    # backends' PBC arithmetic, which the clamped metric amplifies to ~1e-4
    np.testing.assert_allclose(ax_ti, ax_np, rtol=5e-4)
    scale = ax_np[:, :1, np.newaxis]  # largest axis of each tensor
    np.testing.assert_array_less(
        np.abs(H_ti - H_np), np.broadcast_to(5e-4 * scale + 1e-12, H_np.shape)
    )

    # eigenvectors are only unique where their eigenvalue is well separated
    if dim > 1:
        gaps = np.abs(np.diff(ax_np, axis=1)) / ax_np[:, :1]
        sep = np.ones_like(ax_np, dtype=bool)
        sep[:, :-1] &= gaps > 1e-2
        sep[:, 1:] &= gaps > 1e-2
        diff = np.abs(V_ti - V_np).max(axis=1)
        np.testing.assert_array_less(diff[sep], 1e-3)
    else:
        np.testing.assert_array_equal(V_ti, V_np)
