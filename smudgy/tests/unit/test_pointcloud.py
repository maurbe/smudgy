"""Unit tests for the PointCloud public API.

Focus: input/output shape & type consistency of the frontend entry points
(``__init__``, ``global_setup``, ``compute_smoothing``, ``compute_density``,
``interpolate``, ``deposit``, plus the small field-management helpers).

Heavy backend computation (``execution._dispatch``, taichi init, kd-tree
queries) is monkeypatched with lightweight fakes so these tests run fast and
without a working taichi/MPI environment. They check *contracts* (shapes,
dtypes, raised exceptions) rather than numerical correctness -- numerical
correctness and cross-backend agreement belong in separate test modules.

Adjust the import below to match your package layout.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Adjust this import to match the actual package name/path.
# ---------------------------------------------------------------------------
from smudgy.pointcloud import PointCloud, STRUCTURES, INTERPOLATION_MODES
from smudgy import execution


# =============================================================================
# Fixtures: fake backend so tests don't depend on taichi/MPI/kd-tree internals
# =============================================================================
@pytest.fixture(autouse=True)
def fake_backend(monkeypatch):
    """Stub out execution._dispatch, taichi init, and kd-tree calls.

    Returns shapes consistent with what real backends would produce, so the
    PointCloud-level shape/type contracts can be tested in isolation.
    """
    monkeypatch.setattr("smudgy.pointcloud.taichi_init", lambda **kwargs: None)

    class _FakeTree:
        def __init__(self, data):
            self.data = data

    def fake_build_kdtree(positions, boxsize=None):
        return _FakeTree(positions)

    def fake_query_kdtree(tree, qpos, k):
        n = qpos.shape[0]
        # fake distances/indices with the right shape; indices valid & in-bounds
        nn_dists = np.ones((n, k), dtype=np.float32)
        nn_inds = np.zeros((n, k), dtype=np.int64)
        return nn_dists, nn_inds

    monkeypatch.setattr("smudgy.pointcloud.build_kdtree", fake_build_kdtree)
    monkeypatch.setattr("smudgy.pointcloud.query_kdtree", fake_query_kdtree)

    def fake_dispatch(name, **kwargs):
        dim = kwargs.get("dim", 3)
        if name == "compute_hsml":
            n = kwargs["nn_dists"].shape[0]
            return np.ones(n, dtype=np.float32)
        if name == "compute_hmat":
            qpos = kwargs["query_positions"]
            n = qpos.shape[0]
            tensors = np.tile(np.eye(dim, dtype=np.float32), (n, 1, 1))
            eigvals = np.ones((n, dim), dtype=np.float32)
            eigvecs = np.tile(np.eye(dim, dtype=np.float32), (n, 1, 1))
            nn_dists_vec = np.zeros(
                (n, kwargs["neighbor_positions"].shape[1], dim), dtype=np.float32
            )
            return tensors, eigvals, eigvecs, nn_dists_vec
        if name == "compute_density":
            n = kwargs["neighbor_weights"].shape[0]
            return np.ones(n, dtype=np.float32)
        if name == "interpolate":
            fields = kwargs["fields"]
            mode = kwargs["mode"]
            m = fields.shape[0]
            num_fields = fields.shape[-1]
            if mode == "field":
                return np.ones((m, num_fields), dtype=np.float32)
            if mode == "gradient":
                return np.ones((m, num_fields, dim), dtype=np.float32)
            if mode == "divergence":
                return np.ones((m, 1), dtype=np.float32)
            if mode == "curl":
                comps = 1 if dim == 2 else 3
                return np.ones((m, comps), dtype=np.float32)
        if name == "deposit":
            gridnums = kwargs["gridnums"]
            fields = kwargs["particle_fields"]
            num_fields = fields.shape[-1]
            grid_shape = tuple(int(g) for g in gridnums)
            fields_grid = np.ones((num_fields, *grid_shape), dtype=np.float32)
            weights_grid = np.ones(grid_shape, dtype=np.float32)
            return fields_grid, weights_grid
        if name == "project_2d":
            h = kwargs["h_tensor"]
            n = h.shape[0]
            return (
                np.ones((n, 2, 2), dtype=np.float32),
                np.ones((n, 2), dtype=np.float32),
                np.tile(np.eye(2, dtype=np.float32), (n, 1, 1)),
            )
        raise AssertionError(f"Unexpected dispatch call: {name}")

    monkeypatch.setattr(execution, "_dispatch", fake_dispatch)
    monkeypatch.setattr("smudgy.pointcloud.execution", execution)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def make_positions(rng, n=50, dim=3):
    return rng.uniform(0, 1, size=(n, dim)).astype(np.float32)


# =============================================================================
# __init__
# =============================================================================
class TestInit:
    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_valid_dims_accepted(self, rng, dim):
        pos = make_positions(rng, n=20, dim=dim)
        pc = PointCloud(pos, verbose=False)
        assert pc.dim == dim
        assert pc.positions.shape == (20, dim)
        assert pc.positions.dtype == np.float32

    def test_invalid_dim_rejected(self, rng):
        pos = rng.uniform(0, 1, size=(20, 4)).astype(np.float32)
        with pytest.raises(AssertionError):
            PointCloud(pos, verbose=False)

    def test_default_weights_are_uniform_and_correct_shape(self, rng):
        pos = make_positions(rng, n=30)
        pc = PointCloud(pos, verbose=False)
        assert pc.weights.shape == (30,)
        assert pc.weights.dtype == np.float32
        assert np.all(pc.weights == 1.0)

    def test_explicit_weights_cast_and_shape_checked(self, rng):
        pos = make_positions(rng, n=30)
        weights = rng.uniform(0.1, 2.0, size=30).astype(np.float64)
        pc = PointCloud(pos, weights=weights, verbose=False)
        assert pc.weights.dtype == np.float32
        assert pc.weights.shape == (30,)

    def test_weights_shape_mismatch_raises(self, rng):
        pos = make_positions(rng, n=30)
        weights = rng.uniform(0.1, 2.0, size=25).astype(np.float32)
        with pytest.raises(AssertionError):
            PointCloud(pos, weights=weights, verbose=False)

    def test_positions_input_dtype_is_cast_to_float32(self, rng):
        pos = rng.uniform(0, 1, size=(10, 3)).astype(np.float64)
        pc = PointCloud(pos, verbose=False)
        assert pc.positions.dtype == np.float32

    def test_boxsize_none_sets_non_periodic(self, rng):
        pos = make_positions(rng, n=10)
        pc = PointCloud(pos, boxsize=None, verbose=False)
        assert pc.periodic is False
        assert pc.boxsize is None

    def test_scalar_boxsize_broadcast_to_dim(self, rng):
        pos = make_positions(rng, n=10, dim=3)
        pc = PointCloud(pos, boxsize=1.0, verbose=False)
        assert pc.periodic is True
        assert pc.boxsize.shape == (3,)
        assert np.all(pc.boxsize == 1.0)

    def test_vector_boxsize_correct_shape(self, rng):
        pos = make_positions(rng, n=10, dim=3)
        pc = PointCloud(pos, boxsize=[1.0, 2.0, 3.0], verbose=False)
        assert pc.boxsize.shape == (3,)
        np.testing.assert_array_equal(pc.boxsize, [1.0, 2.0, 3.0])

    def test_vector_boxsize_wrong_length_raises(self, rng):
        pos = make_positions(rng, n=10, dim=3)
        with pytest.raises(AssertionError):
            PointCloud(pos, boxsize=[1.0, 2.0], verbose=False)

    def test_backend_default_is_taichi(self, rng):
        pos = make_positions(rng, n=10)
        pc = PointCloud(pos, verbose=False)
        assert pc.backend == "taichi"

    def test_invalid_backend_raises(self, rng):
        pos = make_positions(rng, n=10)
        with pytest.raises(AssertionError):
            PointCloud(pos, backend="not_a_backend", verbose=False)

    def test_backend_non_string_raises(self, rng):
        pos = make_positions(rng, n=10)
        with pytest.raises(AssertionError):
            PointCloud(pos, backend=123, verbose=False)


# =============================================================================
# set_backend / global_setup / _set_property
# =============================================================================
class TestSetupProperties:
    def test_set_backend_valid(self, rng):
        pc = PointCloud(make_positions(rng), verbose=False)
        pc.set_backend("numpy")
        assert pc.backend == "numpy"

    def test_set_backend_invalid_raises(self, rng):
        pc = PointCloud(make_positions(rng), verbose=False)
        with pytest.raises(AssertionError):
            pc.set_backend("bogus")

    def test_global_setup_returns_self_for_chaining(self, rng):
        pc = PointCloud(make_positions(rng), verbose=False)
        result = pc.global_setup(structure="isotropic", kernel_name="cubic_spline", num_neighbors=16)
        assert result is pc
        assert pc.structure == "isotropic"
        assert pc.kernel_name == "cubic_spline"
        assert pc.num_neighbors == 16

    @pytest.mark.parametrize("structure", STRUCTURES)
    def test_valid_structures_accepted(self, rng, structure):
        pc = PointCloud(make_positions(rng), verbose=False)
        pc.global_setup(structure=structure)
        assert pc.structure == structure

    def test_invalid_structure_raises_value_error(self, rng):
        pc = PointCloud(make_positions(rng), verbose=False)
        with pytest.raises(ValueError):
            pc.global_setup(structure="not_a_structure")

    def test_structure_non_string_raises_type_error(self, rng):
        pc = PointCloud(make_positions(rng), verbose=False)
        with pytest.raises(TypeError):
            pc._set_property("structure", 123)

    def test_num_neighbors_non_int_raises_type_error(self, rng):
        pc = PointCloud(make_positions(rng), verbose=False)
        with pytest.raises(TypeError):
            pc._set_property("num_neighbors", 16.0)

    def test_num_neighbors_non_positive_raises_value_error(self, rng):
        pc = PointCloud(make_positions(rng), verbose=False)
        with pytest.raises(ValueError):
            pc._set_property("num_neighbors", 0)

    def test_unsupported_property_raises_value_error(self, rng):
        pc = PointCloud(make_positions(rng), verbose=False)
        with pytest.raises(ValueError):
            pc._set_property("unsupported_prop", "x")

    def test_property_not_set_raises_attribute_error(self, rng):
        pc = PointCloud(make_positions(rng), verbose=False)
        with pytest.raises(AttributeError):
            pc._check_property("structure")


# =============================================================================
# compute_smoothing
# =============================================================================
class TestComputeSmoothing:
    @pytest.mark.parametrize("structure", ["separable", "isotropic"])
    def test_isotropic_smoothing_lengths_shape(self, rng, structure):
        n = 40
        pc = PointCloud(make_positions(rng, n=n), verbose=False)
        pc.global_setup(structure=structure, num_neighbors=8)
        pc.compute_smoothing()
        assert pc.smoothing.smoothing_lengths.shape == (n,)
        assert pc.smoothing.smoothing_lengths.dtype == np.float32

    def test_covariant_smoothing_tensor_shapes(self, rng):
        n, dim = 40, 3
        pc = PointCloud(make_positions(rng, n=n, dim=dim), verbose=False)
        pc.global_setup(structure="covariant", num_neighbors=8)
        pc.compute_smoothing()
        assert pc.smoothing.smoothing_tensors.shape == (n, dim, dim)
        assert pc.smoothing.smoothing_tensors_eigvals.shape == (n, dim)
        assert pc.smoothing.smoothing_tensors_eigvecs.shape == (n, dim, dim)

    def test_num_neighbors_out_of_range_raises(self, rng):
        pc = PointCloud(make_positions(rng), verbose=False)
        pc.global_setup(structure="isotropic")
        with pytest.raises(AssertionError):
            pc.compute_smoothing(num_neighbors=-3)

    def test_neighbors_out_of_bounds_raises_index_error(self, rng, monkeypatch):
        n = 10
        pc = PointCloud(make_positions(rng, n=n), verbose=False)
        pc.global_setup(structure="isotropic", num_neighbors=4)

        def bad_query_kdtree(tree, qpos, k):
            nn_dists = np.ones((qpos.shape[0], k), dtype=np.float32)
            nn_inds = np.full((qpos.shape[0], k), n, dtype=np.int64)  # out of bounds
            return nn_dists, nn_inds

        monkeypatch.setattr("smudgy.pointcloud.query_kdtree", bad_query_kdtree)
        with pytest.raises(IndexError):
            pc.compute_smoothing()


# =============================================================================
# compute_density
# =============================================================================
class TestComputeDensity:
    def test_density_shape_matches_num_particles(self, rng):
        n = 25
        pc = PointCloud(make_positions(rng, n=n), verbose=False)
        pc.global_setup(structure="isotropic", kernel_name="cubic_spline", num_neighbors=8)
        pc.compute_smoothing()
        pc.compute_density()
        assert pc.smoothing.density_isotropic.shape == (n,)
        assert pc.smoothing.density_isotropic.dtype == np.float32

    def test_density_before_smoothing_raises_attribute_error(self, rng):
        pc = PointCloud(make_positions(rng), verbose=False)
        pc.global_setup(structure="isotropic", kernel_name="cubic_spline")
        with pytest.raises(AttributeError):
            pc.compute_density()

    def test_density_without_kernel_or_structure_raises_attribute_error(self, rng):
        pc = PointCloud(make_positions(rng), verbose=False)
        with pytest.raises(AttributeError):
            pc.compute_density()


# =============================================================================
# add_fields / delete_fields
# =============================================================================
class TestFieldManagement:
    def test_add_scalar_field_shape_and_dtype(self, rng):
        n = 20
        pc = PointCloud(make_positions(rng, n=n), verbose=False)
        values = rng.uniform(size=n)
        pc.add_fields("temperature", values)
        assert pc.temperature.shape == (n,)
        assert pc.temperature.dtype == np.float32

    def test_add_vector_field_shape(self, rng):
        n, dim = 20, 3
        pc = PointCloud(make_positions(rng, n=n, dim=dim), verbose=False)
        values = rng.uniform(size=(n, dim))
        pc.add_fields("velocity", values)
        assert pc.velocity.shape == (n, dim)

    def test_add_field_shape_mismatch_raises(self, rng):
        pc = PointCloud(make_positions(rng, n=20), verbose=False)
        with pytest.raises(ValueError):
            pc.add_fields("bad_field", rng.uniform(size=15))

    def test_add_multiple_fields_length_mismatch_raises(self, rng):
        pc = PointCloud(make_positions(rng, n=20), verbose=False)
        with pytest.raises(ValueError):
            pc.add_fields(["a", "b"], [rng.uniform(size=20)])

    def test_add_multiple_fields_type_mismatch_raises(self, rng):
        pc = PointCloud(make_positions(rng, n=20), verbose=False)
        with pytest.raises(ValueError):
            pc.add_fields(["a"], rng.uniform(size=20))  # values not list/tuple

    def test_delete_existing_field(self, rng):
        pc = PointCloud(make_positions(rng, n=20), verbose=False)
        pc.add_fields("temp", rng.uniform(size=20))
        assert hasattr(pc, "temp")
        pc.delete_fields("temp")
        assert not hasattr(pc, "temp")

    def test_delete_nonexistent_field_does_not_raise(self, rng):
        pc = PointCloud(make_positions(rng, n=20), verbose=False)
        pc.delete_fields("does_not_exist")  # should just print, not raise


# =============================================================================
# interpolate
# =============================================================================
class TestInterpolate:
    def _setup_ready_pc(self, rng, n=30, dim=3, structure="isotropic"):
        pc = PointCloud(make_positions(rng, n=n, dim=dim), verbose=False)
        pc.global_setup(structure=structure, kernel_name="cubic_spline", num_neighbors=8)
        pc.compute_smoothing()
        pc.compute_density()
        return pc

    def test_field_mode_output_shape(self, rng):
        n, dim = 30, 3
        pc = self._setup_ready_pc(rng, n=n, dim=dim)
        pc.add_fields("scalar_field", rng.uniform(size=n))
        out = pc.interpolate("scalar_field", mode="field")
        assert out.shape == (n, 1)

    def test_gradient_mode_output_shape(self, rng):
        n, dim = 30, 3
        pc = self._setup_ready_pc(rng, n=n, dim=dim)
        pc.add_fields("scalar_field", rng.uniform(size=n))
        out = pc.interpolate("scalar_field", mode="gradient")
        assert out.shape == (n, 1, dim)

    def test_divergence_requires_vector_field_with_dim_components(self, rng):
        n, dim = 30, 3
        pc = self._setup_ready_pc(rng, n=n, dim=dim)
        pc.add_fields("scalar_field", rng.uniform(size=n))
        with pytest.raises(ValueError):
            pc.interpolate("scalar_field", mode="divergence")

    def test_divergence_output_shape_for_valid_vector_field(self, rng):
        n, dim = 30, 3
        pc = self._setup_ready_pc(rng, n=n, dim=dim)
        pc.add_fields("velocity", rng.uniform(size=(n, dim)))
        out = pc.interpolate("velocity", mode="divergence")
        assert out.shape == (n, 1)

    def test_curl_output_shape_3d(self, rng):
        n, dim = 30, 3
        pc = self._setup_ready_pc(rng, n=n, dim=dim)
        pc.add_fields("velocity", rng.uniform(size=(n, dim)))
        out = pc.interpolate("velocity", mode="curl")
        assert out.shape == (n, 3)

    def test_curl_output_shape_2d(self, rng):
        n, dim = 30, 2
        pc = self._setup_ready_pc(rng, n=n, dim=dim)
        pc.add_fields("velocity", rng.uniform(size=(n, dim)))
        out = pc.interpolate("velocity", mode="curl")
        assert out.shape == (n, 1)

    def test_invalid_mode_raises_value_error(self, rng):
        pc = self._setup_ready_pc(rng)
        pc.add_fields("scalar_field", rng.uniform(size=pc.positions.shape[0]))
        with pytest.raises(ValueError):
            pc.interpolate("scalar_field", mode="not_a_mode")

    def test_interpolate_before_density_raises_attribute_error(self, rng):
        n, dim = 30, 3
        pc = PointCloud(make_positions(rng, n=n, dim=dim), verbose=False)
        pc.global_setup(structure="isotropic", kernel_name="cubic_spline", num_neighbors=8)
        pc.add_fields("scalar_field", rng.uniform(size=n))
        with pytest.raises(AttributeError):
            pc.interpolate("scalar_field")

    def test_interpolate_at_custom_query_positions(self, rng):
        n, dim, m = 30, 3, 12
        pc = self._setup_ready_pc(rng, n=n, dim=dim)
        pc.add_fields("scalar_field", rng.uniform(size=n))
        qpos = make_positions(rng, n=m, dim=dim)
        out = pc.interpolate("scalar_field", query_positions=qpos, mode="field")
        assert out.shape == (m, 1)

    @pytest.mark.parametrize("mode", INTERPOLATION_MODES)
    def test_all_documented_modes_are_accepted(self, rng, mode):
        n, dim = 30, 3
        pc = self._setup_ready_pc(rng, n=n, dim=dim)
        field_name = "velocity" if mode in ("divergence", "curl") else "scalar_field"
        shape = (n, dim) if mode in ("divergence", "curl") else (n,)
        pc.add_fields(field_name, rng.uniform(size=shape))
        out = pc.interpolate(field_name, mode=mode)
        assert isinstance(out, np.ndarray)


# =============================================================================
# deposit
# =============================================================================
class TestDeposit:
    def _setup_ready_pc(self, rng, n=40, dim=3, structure="isotropic", boxsize=1.0):
        pc = PointCloud(make_positions(rng, n=n, dim=dim), boxsize=boxsize, verbose=False)
        pc.global_setup(structure=structure, kernel_name="cubic_spline", num_neighbors=8)
        pc.compute_smoothing()
        return pc

    def test_deposit_output_shape_matches_gridnums(self, rng):
        n, dim = 40, 3
        pc = self._setup_ready_pc(rng, n=n, dim=dim)
        pc.add_fields("scalar_field", rng.uniform(size=n))
        grid = pc.deposit("scalar_field", averaged=False, gridnums=16, adaptive=True)
        assert grid.shape == (1, 16, 16, 16)
        assert grid.dtype == np.float32

    def test_deposit_return_weights_true_returns_tuple(self, rng):
        n, dim = 40, 3
        pc = self._setup_ready_pc(rng, n=n, dim=dim)
        pc.add_fields("scalar_field", rng.uniform(size=n))
        result = pc.deposit(
            "scalar_field", averaged=False, gridnums=8, adaptive=True, return_weights=True
        )
        assert isinstance(result, tuple)
        fields_grid, weights_grid = result
        assert fields_grid.shape == (1, 8, 8, 8)
        assert weights_grid.shape == (8, 8, 8)

    def test_deposit_return_weights_false_returns_array_only(self, rng):
        n, dim = 40, 3
        pc = self._setup_ready_pc(rng, n=n, dim=dim)
        pc.add_fields("scalar_field", rng.uniform(size=n))
        result = pc.deposit(
            "scalar_field", averaged=False, gridnums=8, adaptive=True, return_weights=False
        )
        assert isinstance(result, np.ndarray)

    def test_deposit_no_boxsize_no_extent_raises_value_error(self, rng):
        n, dim = 40, 3
        pc = PointCloud(make_positions(rng, n=n, dim=dim), boxsize=None, verbose=False)
        pc.global_setup(structure="isotropic", kernel_name="cubic_spline", num_neighbors=8)
        pc.compute_smoothing()
        pc.add_fields("scalar_field", rng.uniform(size=n))
        with pytest.raises(ValueError):
            pc.deposit("scalar_field", averaged=False, gridnums=8, adaptive=True)

    def test_deposit_structure_with_non_adaptive_raises_value_error(self, rng):
        n, dim = 40, 3
        pc = self._setup_ready_pc(rng, n=n, dim=dim)
        pc.add_fields("scalar_field", rng.uniform(size=n))
        with pytest.raises(ValueError):
            pc.deposit(
                "scalar_field",
                averaged=False,
                gridnums=8,
                adaptive=False,
                structure="isotropic",
                kernel_name="ngp",
            )

    def test_deposit_plane_projection_output_shape_2d(self, rng):
        n = 40
        pc = self._setup_ready_pc(rng, n=n, dim=3, structure="covariant")
        pc.add_fields("scalar_field", rng.uniform(size=n))
        grid = pc.deposit(
            "scalar_field",
            averaged=False,
            gridnums=10,
            adaptive=True,
            plane_projection=[0, 1],
        )
        assert grid.shape == (1, 10, 10)

    def test_deposit_plane_projection_requires_3d(self, rng):
        n = 40
        pc = self._setup_ready_pc(rng, n=n, dim=2)
        pc.add_fields("scalar_field", rng.uniform(size=n))
        with pytest.raises(ValueError):
            pc.deposit(
                "scalar_field",
                averaged=False,
                gridnums=10,
                adaptive=True,
                plane_projection=[0, 1],
            )

    def test_deposit_plane_projection_basis_not_supported(self, rng):
        n = 40
        pc = self._setup_ready_pc(rng, n=n, dim=3)
        pc.add_fields("scalar_field", rng.uniform(size=n))
        with pytest.raises(ValueError):
            pc.deposit(
                "scalar_field",
                averaged=False,
                gridnums=10,
                adaptive=True,
                plane_projection_basis=[[1, 0, 0], [0, 1, 0]],
            )

    def test_deposit_averaged_length_mismatch_raises(self, rng):
        n = 40
        pc = self._setup_ready_pc(rng, n=n, dim=3)
        pc.add_fields("scalar_field", rng.uniform(size=n))
        with pytest.raises(ValueError):
            pc.deposit(
                "scalar_field", averaged=[True, False, True], gridnums=8, adaptive=True
            )

    def test_deposit_gridnums_scalar_broadcast(self, rng):
        n, dim = 40, 3
        pc = self._setup_ready_pc(rng, n=n, dim=dim)
        pc.add_fields("scalar_field", rng.uniform(size=n))
        grid = pc.deposit("scalar_field", averaged=False, gridnums=6, adaptive=True)
        assert grid.shape[1:] == (6, 6, 6)

    def test_deposit_gridnums_per_axis(self, rng):
        n, dim = 40, 3
        pc = self._setup_ready_pc(rng, n=n, dim=dim)
        pc.add_fields("scalar_field", rng.uniform(size=n))
        grid = pc.deposit("scalar_field", averaged=False, gridnums=[4, 6, 8], adaptive=True)
        assert grid.shape[1:] == (4, 6, 8)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))