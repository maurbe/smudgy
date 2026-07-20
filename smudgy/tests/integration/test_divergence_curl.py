"""Test divergence and curl computation features."""

import numpy as np
import pytest

from smudgy import PointCloud


class TestInterpolationModes:
    """Test new interpolation modes (gradient, divergence, curl)."""

    @pytest.fixture
    def point_cloud_3d(self):
        """Create a 3D point cloud with scalar and vector fields."""
        np.random.seed(42)
        N = 100
        D = 3

        positions = np.random.uniform(0, 1, size=(N, D)).astype(np.float32)
        weights = np.ones(N, dtype=np.float32)

        pc = PointCloud(positions, weights=weights, verbose=False)
        pc.global_setup(
            kernel_name="cubic_spline",
            structure="isotropic",
            num_neighbors=32,
        )
        pc.compute_smoothing()
        pc.compute_density()

        # Add fields
        scalar = np.random.uniform(-1, 1, size=N).astype(np.float32)
        vector = np.random.uniform(-1, 1, size=(N, D)).astype(np.float32)

        pc.add_fields("scalar", scalar)
        pc.add_fields("vector", vector)

        return pc, D

    @pytest.fixture
    def point_cloud_2d(self):
        """Create a 2D point cloud with scalar and vector fields."""
        np.random.seed(42)
        N = 100
        D = 2

        positions = np.random.uniform(0, 1, size=(N, D)).astype(np.float32)
        weights = np.ones(N, dtype=np.float32)

        pc = PointCloud(positions, weights=weights, verbose=False)
        pc.global_setup(
            kernel_name="cubic_spline",
            structure="isotropic",
            num_neighbors=32,
        )
        pc.compute_smoothing()
        pc.compute_density()

        # Add fields
        scalar = np.random.uniform(-1, 1, size=N).astype(np.float32)
        vector = np.random.uniform(-1, 1, size=(N, D)).astype(np.float32)

        pc.add_fields("scalar", scalar)
        pc.add_fields("vector", vector)

        return pc, D

    @pytest.fixture
    def query_positions_3d(self):
        """Create query positions for 3D."""
        np.random.seed(123)
        M = 20
        D = 3
        return np.random.uniform(0, 1, size=(M, D)).astype(np.float32)

    @pytest.fixture
    def query_positions_2d(self):
        """Create query positions for 2D."""
        np.random.seed(123)
        M = 20
        D = 2
        return np.random.uniform(0, 1, size=(M, D)).astype(np.float32)

    # ===== Shape tests =====

    def test_field_mode_shape_scalar(self, point_cloud_3d, query_positions_3d):
        """Test that mode='field' returns correct shape for scalar field."""
        pc, D = point_cloud_3d
        M = query_positions_3d.shape[0]

        result = pc.interpolate_fields("scalar", query_positions_3d, mode="field")
        assert result.shape == (M, 1), f"Expected ({M}, 1), got {result.shape}"
        assert np.all(np.isfinite(result))

    def test_field_mode_shape_vector(self, point_cloud_3d, query_positions_3d):
        """Test that mode='field' returns correct shape for vector field."""
        pc, D = point_cloud_3d
        M = query_positions_3d.shape[0]

        result = pc.interpolate_fields("vector", query_positions_3d, mode="field")
        assert result.shape == (M, D), f"Expected ({M}, {D}), got {result.shape}"
        assert np.all(np.isfinite(result))

    def test_gradient_mode_shape_scalar(self, point_cloud_3d, query_positions_3d):
        """Test that mode='gradient' returns correct shape for scalar field."""
        pc, D = point_cloud_3d
        M = query_positions_3d.shape[0]

        result = pc.interpolate_fields("scalar", query_positions_3d, mode="gradient")
        assert result.shape == (M, 1, D), f"Expected ({M}, 1, {D}), got {result.shape}"
        assert np.all(np.isfinite(result))

    def test_gradient_mode_shape_vector(self, point_cloud_3d, query_positions_3d):
        """Test that mode='gradient' returns correct shape for vector field."""
        pc, D = point_cloud_3d
        M = query_positions_3d.shape[0]

        result = pc.interpolate_fields("vector", query_positions_3d, mode="gradient")
        assert result.shape == (
            M,
            D,
            D,
        ), f"Expected ({M}, {D}, {D}), got {result.shape}"
        assert np.all(np.isfinite(result))

    def test_divergence_mode_shape(self, point_cloud_3d, query_positions_3d):
        """Test that mode='divergence' returns correct shape."""
        pc, D = point_cloud_3d
        M = query_positions_3d.shape[0]

        result = pc.interpolate_fields("vector", query_positions_3d, mode="divergence")
        assert result.shape == (M, 1), f"Expected ({M}, 1), got {result.shape}"
        assert np.all(np.isfinite(result))

    def test_curl_mode_shape_3d(self, point_cloud_3d, query_positions_3d):
        """Test that mode='curl' returns correct shape for 3D."""
        pc, D = point_cloud_3d
        M = query_positions_3d.shape[0]

        result = pc.interpolate_fields("vector", query_positions_3d, mode="curl")
        assert result.shape == (M, 3), f"Expected ({M}, 3), got {result.shape}"
        assert np.all(np.isfinite(result))

    def test_curl_mode_shape_2d(self, point_cloud_2d, query_positions_2d):
        """Test that mode='curl' returns correct shape for 2D (scalar curl)."""
        pc, D = point_cloud_2d
        M = query_positions_2d.shape[0]

        result = pc.interpolate_fields("vector", query_positions_2d, mode="curl")
        assert result.shape == (M, 1), f"Expected ({M}, 1), got {result.shape}"
        assert np.all(np.isfinite(result))

    # ===== Field type validation tests =====

    def test_divergence_rejects_scalar(self, point_cloud_3d, query_positions_3d):
        """Test that divergence raises error for scalar fields."""
        pc, _ = point_cloud_3d

        with pytest.raises(ValueError, match="divergence requires vector"):
            pc.interpolate_fields("scalar", query_positions_3d, mode="divergence")

    def test_curl_rejects_scalar(self, point_cloud_3d, query_positions_3d):
        """Test that curl raises error for scalar fields."""
        pc, _ = point_cloud_3d

        with pytest.raises(ValueError, match="curl requires vector"):
            pc.interpolate_fields("scalar", query_positions_3d, mode="curl")

    def test_gradient_accepts_scalar(self, point_cloud_3d, query_positions_3d):
        """Test that gradient accepts scalar fields."""
        pc, _ = point_cloud_3d

        result = pc.interpolate_fields("scalar", query_positions_3d, mode="gradient")
        assert result.shape[0] > 0  # Should not raise

    def test_gradient_accepts_vector(self, point_cloud_3d, query_positions_3d):
        """Test that gradient accepts vector fields."""
        pc, _ = point_cloud_3d

        result = pc.interpolate_fields("vector", query_positions_3d, mode="gradient")
        assert result.shape[0] > 0  # Should not raise

    # ===== Backward compatibility tests =====

    def test_backward_compat_compute_gradients(
        self, point_cloud_3d, query_positions_3d
    ):
        """Test that deprecated compute_gradients parameter still works."""
        pc, D = point_cloud_3d

        result_new = pc.interpolate_fields(
            "scalar", query_positions_3d, mode="gradient"
        )
        with pytest.warns(DeprecationWarning):
            result_old = pc.interpolate_fields(
                "scalar", query_positions_3d, compute_gradients=True
            )

        assert np.allclose(result_new, result_old)

    def test_backward_compat_interpolate_gradient_fields(
        self, point_cloud_3d, query_positions_3d
    ):
        """Test that interpolate_gradient_fields method still works."""
        pc, D = point_cloud_3d

        result_new = pc.interpolate_fields(
            "scalar", query_positions_3d, mode="gradient"
        )
        result_old = pc.interpolate_gradient_fields("scalar", query_positions_3d)

        assert np.allclose(result_new, result_old)

    # ===== Divergence property tests =====

    def test_divergence_of_radial_field(self, query_positions_3d):
        """Test divergence of a radial field (analytical solution known)."""
        # For a radial field f(r) = r/|r|, divergence should be ~2/|r|
        np.random.seed(42)
        N = 200
        D = 3

        positions = np.random.uniform(0.1, 0.9, size=(N, D)).astype(np.float32)
        center = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        # Radial field
        r_vec = positions - center
        r_mag = np.linalg.norm(r_vec, axis=1, keepdims=True)
        radial_field = r_vec / (r_mag + 1e-6)

        weights = np.ones(N, dtype=np.float32)

        pc = PointCloud(positions, weights=weights, verbose=False)
        pc.global_setup(
            kernel_name="cubic_spline", structure="isotropic", num_neighbors=32
        )
        pc.compute_smoothing()
        pc.compute_density()
        pc.add_fields("radial", radial_field)

        div = pc.interpolate_fields("radial", query_positions_3d, mode="divergence")

        # Divergence should be positive everywhere for radial field
        assert np.all(
            div > -0.5
        ), "Radial field divergence should be generally positive"
        assert np.all(np.isfinite(div))

    # ===== Curl property tests =====

    def test_curl_of_vortex_field_2d(self, query_positions_2d):
        """Test curl of a 2D vortex field."""
        np.random.seed(42)
        N = 200
        D = 2

        positions = np.random.uniform(0.1, 0.9, size=(N, D)).astype(np.float32)
        center = np.array([0.5, 0.5], dtype=np.float32)

        # Vortex field: v = (-y, x) / r²
        r_vec = positions - center
        r_mag_sq = np.sum(r_vec**2, axis=1, keepdims=True) + 1e-6

        vortex_field = np.zeros_like(positions)
        vortex_field[:, 0] = -r_vec[:, 1] / r_mag_sq.ravel()
        vortex_field[:, 1] = r_vec[:, 0] / r_mag_sq.ravel()

        weights = np.ones(N, dtype=np.float32)

        pc = PointCloud(positions, weights=weights, verbose=False)
        pc.global_setup(
            kernel_name="cubic_spline", structure="isotropic", num_neighbors=32
        )
        pc.compute_smoothing()
        pc.compute_density()
        pc.add_fields("vortex", vortex_field)

        curl = pc.interpolate_fields("vortex", query_positions_2d, mode="curl")

        # Curl should be nonzero and finite
        assert np.any(np.abs(curl) > 0.01), "Curl of vortex should be nonzero"
        assert np.all(np.isfinite(curl))

    # ===== Invalid mode test =====

    def test_invalid_mode(self, point_cloud_3d, query_positions_3d):
        """Test that invalid mode raises ValueError."""
        pc, _ = point_cloud_3d

        with pytest.raises(ValueError, match="must be one of"):
            pc.interpolate_fields("scalar", query_positions_3d, mode="invalid")  # type: ignore


class TestInterpolationModeEdgeCases:
    """Test edge cases and special scenarios."""

    def test_mode_with_multiple_fields(self):
        """Test that mode works with multiple fields (though each computed separately)."""
        np.random.seed(42)
        N = 100
        D = 3

        positions = np.random.uniform(0, 1, size=(N, D)).astype(np.float32)
        weights = np.ones(N, dtype=np.float32)

        pc = PointCloud(positions, weights=weights, verbose=False)
        pc.global_setup(
            kernel_name="cubic_spline", structure="isotropic", num_neighbors=32
        )
        pc.compute_smoothing()
        pc.compute_density()

        # Multiple vector fields
        v1 = np.random.uniform(-1, 1, size=(N, D)).astype(np.float32)
        v2 = np.random.uniform(-1, 1, size=(N, D)).astype(np.float32)

        pc.add_fields(["v1", "v2"], [v1, v2])

        M = 10
        query = np.random.uniform(0, 1, size=(M, D)).astype(np.float32)

        # Divergence of concatenated fields should give divergence of both
        result = pc.interpolate_fields(["v1", "v2"], query, mode="divergence")
        assert result.shape == (M, 2), f"Expected ({M}, 2), got {result.shape}"

    def test_anisotropic_structure_with_modes(self):
        """Test that different structures work with new modes."""
        np.random.seed(42)
        N = 100
        D = 3

        positions = np.random.uniform(0, 1, size=(N, D)).astype(np.float32)
        weights = np.ones(N, dtype=np.float32)

        pc = PointCloud(positions, weights=weights, verbose=False)
        pc.global_setup(
            kernel_name="cubic_spline", structure="anisotropic", num_neighbors=32
        )
        pc.compute_smoothing()
        pc.compute_density()

        v = np.random.uniform(-1, 1, size=(N, D)).astype(np.float32)
        pc.add_fields("v", v)

        M = 10
        query = np.random.uniform(0, 1, size=(M, D)).astype(np.float32)

        # Should work with anisotropic structure
        result = pc.interpolate_fields(
            "v", query, mode="divergence", structure="anisotropic"
        )
        assert result.shape == (M, 1)
        assert np.all(np.isfinite(result))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
