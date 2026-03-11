"""Tests evaluation and properties of all available kernel types in Kernel class."""

import numpy as np
import pytest

from smudgy.core.kernels import Kernel

def get_all_kernel_names():
    # Extract allowed kernels from Kernel class
    return [
        "lucy",
        "gaussian",
        "cubic_spline",
        "quintic_spline",
        "wendland_c2",
        "wendland_c4",
        "wendland_c6",
    ]

def test_kernel_evaluate_output_shape_and_type():
    """Test that evaluate_kernel returns correct shape and dtype for all kernels and dims."""
    kernel_names = get_all_kernel_names()
    for dim in (1, 2, 3):
        for name in kernel_names:
            k = Kernel(name, dim)
            # Create input data: (N, K) for isotropic
            N, K = 4, 5
            r_ij = np.abs(np.random.randn(N, K))  # distances >= 0
            h = np.abs(np.random.rand(N)) + 0.1  # avoid zero
            out = k.evaluate_kernel(r_ij, smoothing_lengths=h)
            assert out.shape == (N, K)
            assert np.issubdtype(out.dtype, np.floating)

def test_kernel_evaluate_output_nonnegative():
    """Test that all kernel evaluations are >= 0 for all kernels and dims."""
    kernel_names = get_all_kernel_names()
    for dim in (1, 2, 3):
        for name in kernel_names:
            k = Kernel(name, dim)
            N, K = 4, 5
            r_ij = np.abs(np.random.randn(N, K))
            h = np.abs(np.random.rand(N)) + 0.1
            out = k.evaluate_kernel(r_ij, smoothing_lengths=h)
            assert np.all(out >= 0), f"Kernel {name} dim {dim} produced negative values!"

def test_kernel_evaluate_dtype_preserved():
    """Test that output dtype matches input dtype (float32/float64)."""
    kernel_names = get_all_kernel_names()
    for dim in (1, 2, 3):
        for name in kernel_names:
            k = Kernel(name, dim)
            N, K = 3, 4
            for dtype in (np.float32, np.float64):
                r_ij = np.abs(np.random.randn(N, K)).astype(dtype)
                h = (np.abs(np.random.rand(N)) + 0.1).astype(dtype)
                out = k.evaluate_kernel(r_ij, smoothing_lengths=h)
                assert out.dtype == dtype

def test_kernel_evaluate_zero_for_large_r():
    """Test that kernels with compact support return 0 for r > support radius."""
    # Only applies to kernels with compact support (not gaussian/super_gaussian)
    compact_kernels = [
        "cubic_spline",
        "quintic_spline",
        "wendland_c2",
        "wendland_c4",
        "wendland_c6",
    ]
    for dim in (1, 2, 3):
        for name in compact_kernels:
            k = Kernel(name, dim)
            N, K = 2, 3
            h = np.ones(N)
            # r_ij > 3h for all entries
            r_ij = np.full((N, K), 4.0)
            out = k.evaluate_kernel(r_ij, smoothing_lengths=h)
            assert np.all(out == 0), f"Kernel {name} dim {dim} did not return 0 for r > support."

def test_kernel_evaluate_symmetry():
    """Test that kernel is symmetric: W(r) == W(-r) for isotropic kernels."""
    kernel_names = get_all_kernel_names()
    for dim in (1, 2, 3):
        for name in kernel_names:
            k = Kernel(name, dim)
            N, K = 32, 8
            r_ij = np.random.randn(N, K)
            h = np.abs(np.random.rand(N)) + 0.1
            out1 = k.evaluate_kernel(r_ij, smoothing_lengths=h)
            out2 = k.evaluate_kernel(-r_ij, smoothing_lengths=h)
            mask = (np.abs(out1) > 1e-6) | (np.abs(out2) > 1e-6)
            np.testing.assert_allclose(out1[mask], out2[mask])
