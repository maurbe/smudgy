"""Tests input validation and error handling for the Kernel class."""

import numpy as np
import pytest

from smudgy.core.kernels import get_kernel


def test_invalid_kernel_name():
    """Test invalid kernel name raises ValueError."""
    with pytest.raises(ValueError):
        get_kernel(kernel_name="invalid", dim=2)


def test_invalid_dim_type():
    """Test invalid dim type raises ValueError."""
    with pytest.raises(ValueError):
        get_kernel(kernel_name="gaussian", dim="2")


def test_invalid_dim_value():
    """Test invalid dim value raises ValueError."""
    with pytest.raises(ValueError):
        get_kernel(kernel_name="gaussian", dim=4)


def test_evaluate_kernel_invalid_r_ij_type():
    """Test invalid r_ij type in evaluate_kernel."""
    k = get_kernel(kernel_name="gaussian", dim=2)
    with pytest.raises(ValueError):
        k.evaluate(r_ij=[[0, 1]], h=np.ones(1), mode="anisotropic")


def test_evaluate_kernel_invalid_h_type():
    """Test invalid h type in evaluate_kernel."""
    k = get_kernel(kernel_name="gaussian", dim=2)
    r_ij = np.zeros((1, 1))
    with pytest.raises(ValueError):
        k.evaluate(r_ij=r_ij, h=[1.0], mode="anisotropic")


def test_evaluate_kernel_anisotropic_missing_H():
    """Test missing H in anisotropic kernel evaluation."""
    k = get_kernel(kernel_name="gaussian", dim=2)
    r_ij = np.zeros((1, 1, 2))
    with pytest.raises(ValueError):
        k.evaluate(r_ij=r_ij, h=None, mode="anisotropic")


def test_evaluate_kernel_anisotropic_invalid_H_type():
    """Test invalid H type in anisotropic kernel evaluation."""
    k = get_kernel(kernel_name="gaussian", dim=2)
    r_ij = np.zeros((1, 1, 2))
    with pytest.raises(ValueError):
        k.evaluate(r_ij=r_ij, h=[[[1.0, 0.0], [0.0, 1.0]]], mode="anisotropic")


def test_evaluate_kernel_anisotropic_invalid_H_shape():
    """Test invalid H shape in anisotropic kernel evaluation."""
    k = get_kernel(kernel_name="gaussian", dim=2)
    r_ij = np.zeros((1, 1, 2))
    H = np.ones((1, 2, 3))
    with pytest.raises(ValueError):
        k.evaluate(r_ij=r_ij, h=H, mode="anisotropic")


def test_evaluate_gradient_invalid_r_ij_type():
    """Test invalid r_ij type in evaluate_gradient."""
    k = get_kernel(kernel_name="gaussian", dim=2)
    with pytest.raises(ValueError):
        k.evaluate_gradient(r_ij=[[0, 1]], h=np.ones(1), mode="anisotropic")


def test_evaluate_gradient_invalid_h_type():
    """Test invalid h type in evaluate_gradient."""
    k = get_kernel(kernel_name="gaussian", dim=2)
    r_ij = np.zeros((1, 1, 2))
    with pytest.raises(ValueError):
        k.evaluate_gradient(r_ij=r_ij, h=[1.0], mode="anisotropic")


def test_evaluate_gradient_anisotropic_missing_H():
    """Test missing H in anisotropic gradient evaluation."""
    k = get_kernel(kernel_name="gaussian", dim=2)
    r_ij = np.zeros((1, 1, 2))
    with pytest.raises(ValueError):
        k.evaluate_gradient(r_ij=r_ij, h=None, mode="anisotropic")


def test_evaluate_gradient_anisotropic_invalid_H_type():
    """Test invalid H type in anisotropic gradient evaluation."""
    k = get_kernel(kernel_name="gaussian", dim=2)
    r_ij = np.zeros((1, 1, 2))
    with pytest.raises(ValueError):
        k.evaluate_gradient(r_ij=r_ij, h=[[[1.0, 0.0], [0.0, 1.0]]], mode="anisotropic")


def test_evaluate_gradient_anisotropic_invalid_H_shape():
    """Test invalid H shape in anisotropic gradient evaluation."""
    k = get_kernel(kernel_name="gaussian", dim=2)
    r_ij = np.zeros((1, 1, 2))
    H = np.ones((1, 2, 3))
    with pytest.raises(ValueError):
        k.evaluate_gradient(r_ij=r_ij, h=H, mode="anisotropic")
