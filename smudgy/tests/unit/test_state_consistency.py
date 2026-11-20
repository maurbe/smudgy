"""Regression tests for PointCloud state handling and input validation.

Each test pins a bug found in the post-Quadrants-port review: stale or
mismatched smoothing/density state, rejected-but-valid inputs, and inputs that
used to fail with an unhelpful error.
"""

import numpy as np
import pytest

from smudgy import PointCloud


def _cloud(rng, n=300, dim=3, periodic=True):
    pos = rng.uniform(0, 1, (n, dim))
    pc = PointCloud(pos, boxsize=1.0 if periodic else None, verbose=False)
    pc.add_fields("f", rng.standard_normal((n, dim)))
    return pc


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# -- integer inputs ------------------------------------------------------------
def test_numpy_integer_num_neighbors_accepted(rng):
    pc = _cloud(rng)
    pc.global_setup(num_neighbors=np.int64(8), structure="isotropic")
    pc.compute_smoothing(num_neighbors=np.int32(8))
    assert pc.smoothing.num_neighbors == 8


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_num_neighbors_rejected(rng, bad):
    pc = _cloud(rng)
    with pytest.raises(ValueError):
        pc.global_setup(num_neighbors=bad)


# -- smoothing at query positions ------------------------------------------------
def test_query_smoothing_blocks_particle_operations(rng):
    pc = _cloud(rng).global_setup(num_neighbors=8, structure="isotropic",
                                  kernel_name="cubic_spline")
    pc.compute_smoothing(query_positions=rng.uniform(0, 1, (50, 3)))
    assert pc.smoothing.smoothing_lengths.shape == (50,)
    with pytest.raises(ValueError, match="query positions"):
        pc.compute_density()
    with pytest.raises(ValueError, match="query positions"):
        pc.deposit("f", averaged=False, gridnums=8)
    pc.compute_smoothing()
    pc.compute_density()  # usable again


# -- stale densities / kernels -------------------------------------------------
def test_recomputed_smoothing_invalidates_density(rng):
    pc = _cloud(rng).global_setup(num_neighbors=8, structure="isotropic",
                                  kernel_name="cubic_spline")
    pc.compute_smoothing()
    pc.compute_density()
    pc.find_neighbors(num_neighbors=16)
    pc.compute_smoothing(num_neighbors=16)
    with pytest.raises(AttributeError, match="density"):
        pc.interpolate("f")


def test_interpolate_uses_kernel_of_its_structures_density(rng):
    pc = _cloud(rng).global_setup(num_neighbors=16, kernel_name="cubic_spline")
    pc.compute_smoothing(structure="isotropic")
    pc.compute_smoothing(structure="covariant")
    pc.compute_density(structure="isotropic", kernel_name="cubic_spline")
    ref = pc.interpolate("f", structure="isotropic")
    pc.compute_density(structure="covariant", kernel_name="wendland_c2")
    np.testing.assert_array_equal(pc.interpolate("f", structure="isotropic"), ref)


def test_interpolate_without_global_num_neighbors(rng):
    pc = _cloud(rng)
    pc.compute_smoothing(num_neighbors=8, structure="isotropic")
    pc.compute_density(kernel_name="cubic_spline", structure="isotropic")
    assert pc.interpolate("f", structure="isotropic").shape == (300, 3)
    q = rng.uniform(0, 1, (20, 3))
    assert pc.interpolate("f", query_positions=q, structure="isotropic").shape == (20, 3)


def test_interpolate_rejects_mismatched_neighbor_lists(rng):
    pc = _cloud(rng).global_setup(num_neighbors=8, structure="isotropic",
                                  kernel_name="cubic_spline")
    pc.compute_smoothing()
    pc.compute_density()
    pc.find_neighbors(num_neighbors=12)
    with pytest.raises(ValueError, match="neighbor lists"):
        pc.interpolate("f")


# -- set_smoothing ---------------------------------------------------------------
def test_set_smoothing_tensors_without_eigensystem(rng):
    pc = _cloud(rng).global_setup(num_neighbors=16, structure="covariant",
                                  kernel_name="cubic_spline")
    pc.compute_smoothing()
    kw = dict(fields="f", averaged=False, gridnums=12)
    ref = pc.deposit(**kw)
    fresh = _cloud(np.random.default_rng(0)).global_setup(
        structure="covariant", kernel_name="cubic_spline")
    fresh.set_smoothing(structure="covariant",
                        smoothing_tensors=pc.smoothing.smoothing_tensors)
    np.testing.assert_allclose(fresh.deposit(**kw), ref, rtol=1e-4, atol=1e-5)


def test_set_smoothing_is_deposit_only(rng):
    pc = _cloud(rng).global_setup(structure="isotropic", kernel_name="cubic_spline")
    pc.set_smoothing(structure="isotropic", smoothing_lengths=np.full(300, 0.1))
    pc.deposit("f", averaged=False, gridnums=8)
    with pytest.raises(ValueError, match="set_smoothing"):
        pc.compute_density()


# -- deposit -------------------------------------------------------------------------
def test_plane_projection_input_forms(rng):
    pc = _cloud(rng).global_setup(kernel_name="cic")
    kw = dict(fields="f", averaged=False, gridnums=8, adaptive=False)
    ref = pc.deposit(plane_projection=[0, 2], **kw)
    np.testing.assert_array_equal(pc.deposit(plane_projection=np.array([0, 2]), **kw), ref)
    np.testing.assert_array_equal(pc.deposit(plane_projection=(0, 2), **kw), ref)
    for bad in ([0, 0], [0, 3], [0, 1, 2], [0]):
        with pytest.raises(ValueError, match="plane_projection"):
            pc.deposit(plane_projection=bad, **kw)
    pc2 = _cloud(rng, dim=2).global_setup(kernel_name="cic")
    with pytest.raises(ValueError, match="3D"):
        pc2.deposit(plane_projection=[0, 1], **kw)


@pytest.mark.parametrize("structure", ["separable", "isotropic", "covariant"])
def test_deposit_extent_without_particles(rng, structure):
    pc = PointCloud(rng.uniform(0, 0.5, (200, 2)), verbose=False)
    pc.global_setup(num_neighbors=8, kernel_name="tsc")
    pc.compute_smoothing(structure="isotropic" if structure == "separable" else structure)
    grid = pc.deposit(None, averaged=False, gridnums=8, extent=[[0.6, 0.9]] * 2,
                      structure=structure)
    assert grid.shape == (8, 8) and not grid.any()


# -- fields --------------------------------------------------------------------------
def test_field_names_cannot_shadow_attributes(rng):
    pc = _cloud(rng)
    for name in ("deposit", "positions", "smoothing"):
        with pytest.raises(ValueError, match="existing PointCloud attribute"):
            pc.add_fields(name, np.ones(300))
    pc.add_fields("f", np.zeros(300))  # overwriting an own field is fine
    pc.delete_fields("positions")  # not a field: left alone
    assert pc.positions.shape == (300, 3)
    pc.delete_fields("f")
    assert not hasattr(pc, "f")


def test_single_field_given_as_list(rng):
    pc = _cloud(rng, n=5)
    pc.add_fields("g", [1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(pc.g, np.arange(1, 6, dtype=np.float32))


def test_get_rejects_non_array(rng):
    pc = _cloud(rng).global_setup(num_neighbors=8, structure="isotropic",
                                  kernel_name="cubic_spline")
    pc.compute_smoothing()
    pc.compute_density()
    with pytest.raises(TypeError):
        pc.get("kernel_name")


def test_divergence_in_1d(rng):
    pc = _cloud(rng, dim=1).global_setup(num_neighbors=8, structure="isotropic",
                                         kernel_name="cubic_spline")
    pc.compute_smoothing()
    pc.compute_density()
    div = pc.interpolate("f", mode="divergence")
    grad = pc.interpolate("f", mode="gradient")
    np.testing.assert_allclose(div[:, 0], grad[:, 0, 0], rtol=1e-6)


# -- interpolation cache ---------------------------------------------------------
@pytest.mark.parametrize("structure", ["isotropic", "covariant"])
def test_interpolate_reuses_ghost_smoothing(rng, structure, monkeypatch):
    import smudgy.pointcloud as pointcloud_module

    pc = _cloud(rng).global_setup(num_neighbors=16, structure=structure,
                                  kernel_name="cubic_spline")
    pc.compute_smoothing()
    pc.compute_density()
    first = pc.interpolate("f", mode="gradient")

    calls = []
    real_push = pointcloud_module.push_to_ghosts
    monkeypatch.setattr(pointcloud_module, "push_to_ghosts",
                        lambda *a: calls.append(1) or real_push(*a))
    np.testing.assert_array_equal(pc.interpolate("f", mode="gradient"), first)
    assert len(calls) == 1  # the fields only; density and h are cached

    # new smoothing/density: the cache must not be reused
    pc.find_neighbors(num_neighbors=24)
    pc.compute_smoothing(num_neighbors=24)
    pc.compute_density()
    fresh = pc.interpolate("f", mode="gradient")
    ref = _cloud(np.random.default_rng(0)).global_setup(
        num_neighbors=24, structure=structure, kernel_name="cubic_spline")
    ref.compute_smoothing()
    ref.compute_density()
    np.testing.assert_array_equal(fresh, ref.interpolate("f", mode="gradient"))
