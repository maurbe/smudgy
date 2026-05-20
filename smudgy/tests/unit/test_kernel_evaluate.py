"""Tests evaluation and properties of all available kernel types in Kernel class."""

import numpy as np
import pytest

from smudgy.core.kernels import KERNEL_CLASSES, get_kernel


def get_all_kernel_names():
    """Return a list of all kernel names available in the KERNEL_CLASSES."""
    return list(KERNEL_CLASSES.keys())


def random_posdef_tensors(M, K, D):
    """Generate (M, K, D, D) random positive-definite tensors."""
    A = np.random.randn(M, K, D, D)
    # Generate A * A.T + D*I to ensure positive definiteness
    out = np.matmul(A, np.transpose(A, (0, 1, 3, 2))) + D * np.eye(D)[None, None, :, :]
    return out


@pytest.fixture
def test_data():
    """Fixture to provide consistent test dimensions."""
    return {"M": 4, "K": 5}


@pytest.mark.parametrize("name", get_all_kernel_names())
@pytest.mark.parametrize("dim", [1, 2, 3])
class TestKernelProperties:
    """Test suite for kernel evaluation and gradients across all types and dimensions."""

    def test_isotropic_evaluate_shape_and_dtype(self, name, dim, test_data):
        """Test isotropic evaluation shapes and dtypes."""
        k = get_kernel(name, dim)
        M, K = test_data["M"], test_data["K"]

        r_ij = np.random.randn(M, K)
        h = np.random.uniform(
            0.1, 1.0, size=M
        )  # 1D array should be normalized to (M, K)

        for dtype in (np.float32, np.float64):
            out = k.evaluate(r_ij.astype(dtype), h.astype(dtype), mode="isotropic")
            assert out.shape == (M, K)
            assert out.dtype == dtype
            assert np.all(out >= 0), f"Kernel {name} produced negative values"

    def test_anisotropic_evaluate_shape_and_dtype(self, name, dim, test_data):
        """Test anisotropic evaluation shapes and dtypes."""
        k = get_kernel(name, dim)
        M, K = test_data["M"], test_data["K"]

        r_ij = np.random.randn(M, K, dim)
        H = random_posdef_tensors(M, K, dim)

        for dtype in (np.float32, np.float64):
            out = k.evaluate(r_ij.astype(dtype), h=H.astype(dtype), mode="anisotropic")
            assert out.shape == (M, K)
            assert out.dtype == dtype
            assert np.all(
                out >= 0
            ), f"Kernel {name} produced negative values (anisotropic)"

    def test_evaluate_symmetry(self, name, dim, test_data):
        """Test W(r) == W(-r) for both isotropic and anisotropic cases."""
        k = get_kernel(name, dim)
        M, K = test_data["M"], test_data["K"]

        # Isotropic
        r_ij = np.random.randn(M, K)
        h = np.ones(M)
        out1 = k.evaluate(r_ij, h, mode="isotropic")
        out2 = k.evaluate(-r_ij, h, mode="isotropic")
        np.testing.assert_allclose(out1, out2, rtol=1e-5, atol=1e-8)

        # Anisotropic
        r_vec = np.random.randn(M, K, dim)
        H = random_posdef_tensors(M, K, dim)
        out_a1 = k.evaluate(r_vec, h=H, mode="anisotropic")
        out_a2 = k.evaluate(-r_vec, h=H, mode="anisotropic")
        np.testing.assert_allclose(out_a1, out_a2, rtol=1e-5, atol=1e-8)

    def test_compact_support(self, name, dim, test_data):
        """Test that kernels return 0 outside their support radius."""
        k = get_kernel(name, dim)
        M, K = test_data["M"], test_data["K"]
        # r > support * h
        r_ij = np.full((M, K), (k.support + 1.0))
        h = np.ones((M, K))
        print(r_ij, h)
        out = k.evaluate(r_ij, h, mode="isotropic")
        assert np.all(out == 0), f"Kernel {name} did not return 0 outside support"

    def test_gradient_shape_and_dtype(self, name, dim, test_data):
        """Test evaluation of kernel gradients."""
        k = get_kernel(name, dim)
        M, K = test_data["M"], test_data["K"]

        r_vec = np.random.randn(M, K, dim)

        # Isotropic gradient
        h_iso = np.random.uniform(0.1, 1.0, size=(M, K))
        for dtype in (np.float32, np.float64):
            grad = k.evaluate_gradient(
                r_vec.astype(dtype), h_iso.astype(dtype), mode="isotropic"
            )
            assert grad.shape == (M, K, dim)
            assert grad.dtype == dtype

        # Anisotropic gradient
        H_aniso = random_posdef_tensors(M, K, dim)
        for dtype in (np.float32, np.float64):
            grad = k.evaluate_gradient(
                r_vec.astype(dtype), H_aniso.astype(dtype), mode="anisotropic"
            )
            assert grad.shape == (M, K, dim)
            assert grad.dtype == dtype


def test_invalid_initialization():
    """Test that BaseKernelClass raises errors for invalid dim."""
    with pytest.raises(ValueError):
        get_kernel("cubic_spline", dim=4)
    with pytest.raises(ValueError):
        get_kernel("cubic_spline", dim="3")
