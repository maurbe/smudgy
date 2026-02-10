"""Tests input validation and error handling for the Kernel class."""

import numpy as np
import pytest
from sph_lib.kernels import Kernel

def test_invalid_kernel_name():
    with pytest.raises(AssertionError):
        Kernel(kernel_name="invalid", dim=2)

def test_invalid_dim_type():
    with pytest.raises(AssertionError):
        Kernel(kernel_name="gaussian", dim="2")

def test_invalid_dim_value():
    with pytest.raises(AssertionError):
        Kernel(kernel_name="gaussian", dim=4)

def test_evaluate_kernel_invalid_r_ij_type():
    k = Kernel("gaussian", 2)
    with pytest.raises(AssertionError):
        k.evaluate_kernel(r_ij=[[0, 1]], h=np.ones(1))

def test_evaluate_kernel_invalid_h_type():
    k = Kernel("gaussian", 2)
    r_ij = np.zeros((1, 1))
    with pytest.raises(AssertionError):
        k.evaluate_kernel(r_ij=r_ij, h=[1.0])

def test_evaluate_kernel_invalid_h_shape():
    k = Kernel("gaussian", 2)
    r_ij = np.zeros((1, 1))
    h = np.ones((1, 1, 1))
    with pytest.raises(AssertionError):
        k.evaluate_kernel(r_ij=r_ij, h=h)

def test_evaluate_kernel_anisotropic_missing_H():
    k = Kernel("gaussian", 2)
    r_ij = np.zeros((1, 1, 2))
    with pytest.raises(AssertionError):
        k.evaluate_kernel(r_ij=r_ij, h=None, H=None)

def test_evaluate_kernel_anisotropic_invalid_H_type():
    k = Kernel("gaussian", 2)
    r_ij = np.zeros((1, 1, 2))
    with pytest.raises(AssertionError):
        k.evaluate_kernel(r_ij=r_ij, h=None, H=[[[1.0, 0.0], [0.0, 1.0]]])

def test_evaluate_kernel_anisotropic_invalid_H_shape():
    k = Kernel("gaussian", 2)
    r_ij = np.zeros((1, 1, 2))
    H = np.ones((1, 2, 3))
    with pytest.raises(AssertionError):
        k.evaluate_kernel(r_ij=r_ij, h=None, H=H)

def test_evaluate_gradient_invalid_r_ij_vec_type():
    k = Kernel("gaussian", 2)
    with pytest.raises(AssertionError):
        k.evaluate_gradient(r_ij_vec=[[0, 1]], h=np.ones(1))

def test_evaluate_gradient_invalid_h_type():
    k = Kernel("gaussian", 2)
    r_ij_vec = np.zeros((1, 1, 2))
    with pytest.raises(AssertionError):
        k.evaluate_gradient(r_ij_vec=r_ij_vec, h=[1.0])

def test_evaluate_gradient_invalid_h_shape():
    k = Kernel("gaussian", 2)
    r_ij_vec = np.zeros((1, 1, 2))
    h = np.ones((1, 1, 1))
    with pytest.raises(AssertionError):
        k.evaluate_gradient(r_ij_vec=r_ij_vec, h=h)

def test_evaluate_gradient_anisotropic_missing_H():
    k = Kernel("gaussian", 2)
    r_ij_vec = np.zeros((1, 1, 2))
    with pytest.raises(AssertionError):
        k.evaluate_gradient(r_ij_vec=r_ij_vec, h=None, H=None)

def test_evaluate_gradient_anisotropic_invalid_H_type():
    k = Kernel("gaussian", 2)
    r_ij_vec = np.zeros((1, 1, 2))
    with pytest.raises(AssertionError):
        k.evaluate_gradient(r_ij_vec=r_ij_vec, h=None, H=[[[1.0, 0.0], [0.0, 1.0]]])

def test_evaluate_gradient_anisotropic_invalid_H_shape():
    k = Kernel("gaussian", 2)
    r_ij_vec = np.zeros((1, 1, 2))
    H = np.ones((1, 2, 3))
    with pytest.raises(AssertionError):
        k.evaluate_gradient(r_ij_vec=r_ij_vec, h=None, H=H)
