"""Taichi port of kernels.h / kernels.cpp.

Design notes
------------
C++ used runtime polymorphism (SeparableKernel / SphericalKernel base classes,
std::shared_ptr, virtual calls) to pick a kernel at runtime and dispatch on
`dim_` inside each method.

Taichi has no vtables, so both axes of that dispatch are moved to Python /
compile time instead:

  * "which kernel" -> resolved once in Python via the KERNELS_* registries
    below, then the chosen ti.func objects are passed into call sites as
    ti.template() arguments. Taichi fully inlines them -- zero indirection,
    strictly cheaper than the C++ virtual-call version.

  * "which dim" -> `dim` is also passed as ti.template() (a compile-time
    Python int, not a runtime field), so the `if (dim_ == 1) ...` chains
    become `ti.static(dim == 1)` and are constant-folded away entirely.
    Each (kernel, dim) combination gets its own specialized, branch-free
    compiled kernel.

Per-kernel constants (SUPPORT, NODE_1, NODE_2, EPS) are plain Python module
level floats, closed over by the ti.funcs -- Taichi inlines them as literals,
same mechanism as LUCY_SUPPORT in the earlier port.

Piecewise branches *on q* (the running coordinate, not on dim) are genuine
runtime data-dependent ifs, exactly as in the C++ -- there is nothing to fold
away there, so they're left as ordinary `if`/`elif`/`else` inside the funcs.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import pi

import numpy as np
import taichi as ti


# =============================================================================
# Helper functions for projections onto canonical coordinates
# Used for density computation and interpolation workflow (deposit does it internally)
# =============================================================================
@ti.func
def prepare_isotropic_inputs(
    r_vec, h: ti.f32, dim: ti.template(), sigma_val: ti.f32, eps: ti.f32
):
    r_mag = r_vec.norm()
    q = r_mag / (h + eps)
    scale = sigma_val / (h**dim)

    grad_q = ti.Vector.zero(ti.f32, dim)
    if r_mag > 0.0:
        grad_q = r_vec / ((r_mag + eps) * (h + eps))

    return q, grad_q, scale


@ti.func
def prepare_covariant_inputs(
    r_vec, H, dim: ti.template(), sigma_val: ti.f32, eps: ti.f32
):
    H_inv = H.inverse()
    det_H = H.determinant()

    xi = H_inv @ r_vec
    q = xi.norm()
    scale = sigma_val / det_H

    grad_q = ti.Vector.zero(ti.f32, dim)
    if q > 0.0:
        grad_q = (H_inv.transpose() @ xi) / (q + eps)

    return q, grad_q, scale


# =============================================================================
# Shared low-level helpers
# =============================================================================
@ti.func
def _erf(x: ti.f32) -> ti.f32:
    """Abramowitz & Stegun 7.1.26 approximation, max abs error ~1.5e-7.
    Taichi has no built-in erf, needed for the Gaussian kernels' F(q).
    """
    sign = 1.0
    if x < 0.0:
        sign = -1.0
    ax = ti.abs(x)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * ti.exp(-ax * ax)
    return sign * y


@ti.func
def _spherical_evaluate_integral_default(
    F_fn: ti.template(), support: ti.f32, dim: ti.template(), q1: ti.f32, q2_in: ti.f32
) -> ti.f32:
    """Default SphericalKernel::evaluate_integral (used by kernels that don't
    override it: Tophat, Lucy, Gaussian, WendlandC2/C4/C6).
    """
    result = 0.0
    q2 = q2_in
    if q2 > q1:
        if q2 >= support:
            q2 = support
        qq1 = q1
        if qq1 <= 0.0:
            qq1 = 0.0
        result = F_fn(q2, dim) - F_fn(qq1, dim)
    return result


@ti.func
def _separable_integrate_1d(
    F_1d_fn: ti.template(),
    support: ti.f32,
    dim: ti.template(),
    q0_in: ti.f32,
    q1_in: ti.f32,
) -> ti.f32:
    """SeparableKernel::integrate_1d -- 1D integral between two bounds,
    handling the kernel's even symmetry (evaluated via F_1d on |q|).
    """
    result = 0.0
    q0 = q0_in
    q1 = q1_in
    if q1 > q0:
        q0 = ti.max(q0, -support)
        q1 = ti.min(q1, support)
        if q0 >= 0.0:
            result = F_1d_fn(q1, dim) - F_1d_fn(q0, dim)
        elif q1 <= 0.0:
            result = F_1d_fn(-q0, dim) - F_1d_fn(-q1, dim)
        else:
            result = F_1d_fn(-q0, dim) + F_1d_fn(q1, dim)
    return result


# =============================================================================
# Separable kernels (SeparableKernel subclasses)
# =============================================================================
# --- TophatRect ---------------------------------------------------------
TOPHAT_RECT_SUPPORT = 0.5


@ti.func
def tophat_sep_evaluate_1d(q_in: ti.f32, dim: ti.template()) -> ti.f32:
    q = ti.abs(q_in)
    result = 0.0
    if q <= TOPHAT_RECT_SUPPORT:
        result = 1.0
    return result


@ti.func
def tophat_sep_sigma(dim: ti.template()) -> ti.f32:
    return 1.0


@ti.func
def tophat_sep_F_1d(q_in: ti.f32, dim: ti.template()) -> ti.f32:
    q = ti.max(q_in, 0.0)
    q = ti.min(q, TOPHAT_RECT_SUPPORT)
    return q


# --- TSCRect -------------------------------------------------------------
TSC_RECT_SUPPORT = 1.5


@ti.func
def tsc_sep_evaluate_1d(q_in: ti.f32, dim: ti.template()) -> ti.f32:
    q_abs = ti.abs(q_in)
    result = 0.0
    if q_abs < TSC_RECT_SUPPORT:
        result = 1.0 - q_abs / TSC_RECT_SUPPORT
    return result


@ti.func
def tsc_sep_sigma(dim: ti.template()) -> ti.f32:
    return 1.0 / (TSC_RECT_SUPPORT**dim)


@ti.func
def tsc_sep_F_1d(q_in: ti.f32, dim: ti.template()) -> ti.f32:
    q = ti.max(q_in, 0.0)
    q = ti.min(q, TSC_RECT_SUPPORT)
    return q - (q * q) / (2.0 * TSC_RECT_SUPPORT)


# --- GaussianRect ----------------------------------------------------------
GAUSSIAN_RECT_SUPPORT = 3.0


@ti.func
def gaussian_sep_evaluate_1d(q_in: ti.f32, dim: ti.template()) -> ti.f32:
    q = ti.abs(q_in)
    result = 0.0
    if q < GAUSSIAN_RECT_SUPPORT:
        result = ti.exp(-q * q)
    return result


@ti.func
def gaussian_sep_sigma(dim: ti.template()) -> ti.f32:
    result = 0.0
    if ti.static(dim == 1):
        result = 1.0 / ti.sqrt(pi)
    elif ti.static(dim == 2):
        result = 1.0 / pi
    elif ti.static(dim == 3):
        result = 1.0 / (pi**1.5)
    return result


@ti.func
def gaussian_sep_F_1d(q_in: ti.f32, dim: ti.template()) -> ti.f32:
    q = ti.max(q_in, 0.0)
    q = ti.min(q, GAUSSIAN_RECT_SUPPORT)
    return 0.5 * ti.sqrt(pi) * _erf(q)


# =============================================================================
# Spherical kernels (SphericalKernel subclasses)
# =============================================================================
# --- Tophat ----------------------------------------------------------------
TOPHAT_SUPPORT = 0.5


@ti.func
def tophat_evaluate(q: ti.f32, dim: ti.template()) -> ti.f32:
    result = 0.0
    if q <= TOPHAT_SUPPORT:
        result = 1.0
    return result


@ti.func
def tophat_sigma(dim: ti.template()) -> ti.f32:
    result = 0.0
    if ti.static(dim == 1):
        result = 1.0
    elif ti.static(dim == 2):
        result = 4.0 / pi
    elif ti.static(dim == 3):
        result = 6.0 / pi
    return result


@ti.func
def tophat_F(q_in: ti.f32, dim: ti.template()) -> ti.f32:
    q = ti.max(q_in, 0.0)
    q = ti.min(q, TOPHAT_SUPPORT)
    result = 0.0
    if ti.static(dim == 1):
        result = q
    elif ti.static(dim == 2):
        result = 0.5 * q * q
    elif ti.static(dim == 3):
        result = (1.0 / 3.0) * q * q * q
    return result


@ti.func
def tophat_evaluate_integral(q1: ti.f32, q2: ti.f32, dim: ti.template()) -> ti.f32:
    return _spherical_evaluate_integral_default(tophat_F, TOPHAT_SUPPORT, dim, q1, q2)


@ti.func
def tophat_gradient(q: ti.f32, grad_q, dim: ti.template()):
    return 0.0 * grad_q


# --- TSC ---------------------------------------------------------------
TSC_SUPPORT = 1.5


@ti.func
def tsc_evaluate(q_in: ti.f32, dim: ti.template()) -> ti.f32:
    q = ti.abs(q_in)
    result = 0.0
    if q < TSC_SUPPORT:
        result = 1.0 - q / TSC_SUPPORT
    return result


@ti.func
def tsc_sigma(dim: ti.template()) -> ti.f32:
    result = 0.0
    if ti.static(dim == 1):
        result = 1.0 / TSC_SUPPORT
    elif ti.static(dim == 2):
        result = 3.0 / (pi * TSC_SUPPORT * TSC_SUPPORT)
    elif ti.static(dim == 3):
        result = 3.0 / (pi * TSC_SUPPORT * TSC_SUPPORT * TSC_SUPPORT)
    return result


@ti.func
def tsc_F(q_in: ti.f32, dim: ti.template()) -> ti.f32:
    q = ti.max(q_in, 0.0)
    q = ti.min(q, TSC_SUPPORT)
    h = TSC_SUPPORT
    result = 0.0
    if ti.static(dim == 1):
        result = q - (q * q) / (2.0 * h)
    elif ti.static(dim == 2):
        result = 0.5 * q * q - (q * q * q) / (3.0 * h)
    elif ti.static(dim == 3):
        result = (1.0 / 3.0) * q * q * q - (q * q * q * q) / (4.0 * h)
    return result


@ti.func
def tsc_evaluate_integral(q1_in: ti.f32, q2_in: ti.f32, dim: ti.template()) -> ti.f32:
    result = 0.0
    if not (q2_in <= 0.0 or q1_in >= TSC_SUPPORT):
        q1 = ti.max(0.0, q1_in)
        q2 = ti.min(TSC_SUPPORT, q2_in)
        result = tsc_F(q2, dim) - tsc_F(q1, dim)
    return result


@ti.func
def tsc_gradient(q: ti.f32, grad_q, dim: ti.template()):
    q = ti.abs(q)
    dWdq = 0.0
    if q <= TSC_SUPPORT:
        dWdq = -(1.0 / TSC_SUPPORT)
    return dWdq * grad_q


# --- Gaussian ------------------------------------------------------------
GAUSSIAN_SUPPORT = 3.0


@ti.func
def gaussian_evaluate(q: ti.f32, dim: ti.template()) -> ti.f32:
    result = 0.0
    if q < GAUSSIAN_SUPPORT:
        result = ti.exp(-q * q)
    return result


@ti.func
def gaussian_sigma(dim: ti.template()) -> ti.f32:
    result = 0.0
    if ti.static(dim == 1):
        result = 1.0 / ti.sqrt(pi)
    elif ti.static(dim == 2):
        result = 1.0 / pi
    elif ti.static(dim == 3):
        result = 1.0 / (pi**1.5)
    return result


@ti.func
def gaussian_F(q_in: ti.f32, dim: ti.template()) -> ti.f32:
    q = ti.max(q_in, 0.0)
    q = ti.min(q, GAUSSIAN_SUPPORT)
    result = 0.0
    if ti.static(dim == 1):
        result = 0.5 * ti.sqrt(pi) * _erf(q)
    elif ti.static(dim == 2):
        result = -0.5 * ti.exp(-q * q)
    elif ti.static(dim == 3):
        result = 0.25 * (ti.sqrt(pi) * _erf(q) - 2.0 * q * ti.exp(-q * q))
    return result


@ti.func
def gaussian_evaluate_integral(q1: ti.f32, q2: ti.f32, dim: ti.template()) -> ti.f32:
    return _spherical_evaluate_integral_default(
        gaussian_F, GAUSSIAN_SUPPORT, dim, q1, q2
    )


@ti.func
def gaussian_gradient(q: ti.f32, grad_q, dim: ti.template()):
    dWdq = 0.0
    if q <= GAUSSIAN_SUPPORT:
        dWdq = -2.0 * q * ti.exp(-(q * q))
    return dWdq * grad_q


# --- Lucy --------------------------------------------------------------
LUCY_SUPPORT = 1.0


@ti.func
def lucy_evaluate(q: ti.f32, dim: ti.template()) -> ti.f32:
    result = 0.0
    if q <= LUCY_SUPPORT:
        result = (1.0 + 3.0 * q) * (1.0 - q) ** 3
    return result


@ti.func
def lucy_sigma(dim: ti.template()) -> ti.f32:
    result = 0.0
    if ti.static(dim == 1):
        result = 5.0 / 4.0
    elif ti.static(dim == 2):
        result = 5.0 / pi
    elif ti.static(dim == 3):
        result = 105.0 / (16.0 * pi)
    return result


@ti.func
def lucy_F(q_in: ti.f32, dim: ti.template()) -> ti.f32:
    q = ti.max(q_in, 0.0)
    q = ti.min(q, LUCY_SUPPORT)
    result = 0.0
    if ti.static(dim == 1):
        result = q - 2.0 * q**3 + 2.0 * q**4 - 0.6 * q**5
    elif ti.static(dim == 2):
        result = 0.5 * q**2 - 1.5 * q**4 + 1.6 * q**5 - 0.5 * q**6
    elif ti.static(dim == 3):
        result = (
            (1.0 / 3.0) * q**3
            - (6.0 / 5.0) * q**5
            + (4.0 / 3.0) * q**6
            - (3.0 / 7.0) * q**7
        )
    return result


@ti.func
def lucy_evaluate_integral(q1: ti.f32, q2: ti.f32, dim: ti.template()) -> ti.f32:
    return _spherical_evaluate_integral_default(lucy_F, LUCY_SUPPORT, dim, q1, q2)


@ti.func
def lucy_gradient(q: ti.f32, grad_q, dim: ti.template()):
    dWdq = 0.0
    if q <= LUCY_SUPPORT:
        dWdq = -12.0 * q * (1.0 - q) ** 2
    return dWdq * grad_q


# --- CubicSpline -----------------------------------------------------------
CUBIC_SPLINE_SUPPORT = 1.0
CUBIC_SPLINE_NODE_1 = 0.5
CUBIC_SPLINE_EPS = 1e-6


@ti.func
def cubic_spline_evaluate(q: ti.f32, dim: ti.template()) -> ti.f32:
    result = 0.0
    if q <= CUBIC_SPLINE_NODE_1:
        result = 1.0 - 6.0 * q * q + 6.0 * q * q * q
    elif q <= CUBIC_SPLINE_SUPPORT:
        r = 1.0 - q
        result = 2.0 * r * r * r
    return result


@ti.func
def cubic_spline_sigma(dim: ti.template()) -> ti.f32:
    result = 0.0
    if ti.static(dim == 1):
        result = 4.0 / 3.0
    elif ti.static(dim == 2):
        result = 40.0 / (7.0 * pi)
    elif ti.static(dim == 3):
        result = 8.0 / pi
    return result


@ti.func
def cubic_spline_F(q_in: ti.f32, dim: ti.template()) -> ti.f32:
    q = ti.max(q_in, 0.0)
    q = ti.min(q, CUBIC_SPLINE_SUPPORT)
    result = 0.0
    if ti.static(dim == 1):
        if q <= CUBIC_SPLINE_NODE_1:
            result = q - 2.0 * q**3 + 1.5 * q**4
        else:
            result = 0.375 - 0.5 * (1.0 - q) ** 4
    elif ti.static(dim == 2):
        if q <= CUBIC_SPLINE_NODE_1:
            result = 0.5 * q**2 - 1.5 * q**4 + 1.2 * q**5
        else:
            result = 0.0875 - 0.5 * (1.0 - q) ** 4 + 0.4 * (1.0 - q) ** 5
    elif ti.static(dim == 3):
        if q <= CUBIC_SPLINE_NODE_1:
            result = q**3 / 3.0 - 1.2 * q**5 + q**6
        else:
            result = (
                0.03125
                - 0.5 * (1.0 - q) ** 4
                + 0.8 * (1.0 - q) ** 5
                - (1.0 / 3.0) * (1.0 - q) ** 6
            )
    return result


@ti.func
def cubic_spline_evaluate_integral(
    q1_in: ti.f32, q2_in: ti.f32, dim: ti.template()
) -> ti.f32:
    q2 = ti.min(q2_in, CUBIC_SPLINE_SUPPORT)
    q1 = ti.max(q1_in, 0.0)
    result = 0.0
    if q1 <= CUBIC_SPLINE_NODE_1 and CUBIC_SPLINE_NODE_1 < q2:
        result = (
            cubic_spline_F(CUBIC_SPLINE_NODE_1, dim)
            - cubic_spline_F(q1, dim)
            + cubic_spline_F(q2, dim)
            - cubic_spline_F(CUBIC_SPLINE_NODE_1 + CUBIC_SPLINE_EPS, dim)
        )
    else:
        result = cubic_spline_F(q2, dim) - cubic_spline_F(q1, dim)
    return result


@ti.func
def cubic_spline_gradient(q: ti.f32, grad_q, dim: ti.template()):
    dWdq = 0.0
    if q <= CUBIC_SPLINE_NODE_1:
        dWdq = -6.0 * q * (2.0 - 3.0 * q)
    elif q <= CUBIC_SPLINE_SUPPORT:
        dWdq = -6.0 * (1.0 - q) ** 2
    return dWdq * grad_q


# --- QuinticSpline ---------------------------------------------------------
QUINTIC_SPLINE_SUPPORT = 3.0
QUINTIC_SPLINE_NODE_1 = 1.0
QUINTIC_SPLINE_NODE_2 = 2.0
QUINTIC_SPLINE_EPS = 1e-6


@ti.func
def quintic_spline_evaluate(q: ti.f32, dim: ti.template()) -> ti.f32:
    result = 0.0
    if q < QUINTIC_SPLINE_SUPPORT:
        f = 3.0 - q
        s = 2.0 - q
        t = 1.0 - q
        f5 = f**5
        s5 = s**5
        t5 = t**5
        if q < QUINTIC_SPLINE_NODE_1:
            result = f5 - 6.0 * s5 + 15.0 * t5
        elif q < QUINTIC_SPLINE_NODE_2:
            result = f5 - 6.0 * s5
        else:
            result = f5
    return result


@ti.func
def quintic_spline_sigma(dim: ti.template()) -> ti.f32:
    result = 0.0
    if ti.static(dim == 1):
        result = 1.0 / 120.0
    elif ti.static(dim == 2):
        result = 7.0 / (478.0 * pi)
    elif ti.static(dim == 3):
        result = 1.0 / (120.0 * pi)
    return result


@ti.func
def quintic_spline_F(q_in: ti.f32, dim: ti.template()) -> ti.f32:
    q = ti.max(q_in, 0.0)
    q = ti.min(q, QUINTIC_SPLINE_SUPPORT)
    result = 0.0

    if ti.static(dim == 1):
        if q <= QUINTIC_SPLINE_NODE_1:
            result = 66.0 * q - 20.0 * q**3 + 6.0 * q**5 - (5.0 / 3.0) * q**6
        elif q <= QUINTIC_SPLINE_NODE_2:
            result = (
                51.0 * q
                + 37.5 * q**2
                - 70.0 * q**3
                + 37.5 * q**4
                - 9.0 * q**5
                + (5.0 / 6.0) * q**6
            )
        else:
            result = -(1.0 / 6.0) * (3.0 - q) ** 6

    elif ti.static(dim == 2):
        if q <= QUINTIC_SPLINE_NODE_1:
            result = q**2 * (33.0 - 15.0 * q**2 + 5.0 * q**4 - (10.0 / 7.0) * q**5)
        elif q <= QUINTIC_SPLINE_NODE_2:
            result = q**2 * (
                25.5
                + 25.0 * q
                - 52.5 * q**2
                + 30.0 * q**3
                - 7.5 * q**4
                + (5.0 / 7.0) * q**5
            )
        else:
            result = q**2 * (
                121.5 - 135.0 * q + 67.5 * q**2 - 18.0 * q**3 + 2.5 * q**4 - q**5 / 7.0
            )

    elif ti.static(dim == 3):
        if q <= QUINTIC_SPLINE_NODE_1:
            result = q**3 * (22.0 - 12.0 * q**2 + (30.0 / 7.0) * q**4 - 1.25 * q**5)
        elif q <= QUINTIC_SPLINE_NODE_2:
            result = q**3 * (
                17.0
                + 18.75 * q
                - 42.0 * q**2
                + 25.0 * q**3
                - (45.0 / 7.0) * q**4
                + 0.625 * q**5
            )
        else:
            result = q**3 * (
                81.0
                - 101.25 * q
                + 54.0 * q**2
                - 15.0 * q**3
                + (15.0 / 7.0) * q**4
                - q**5 / 8.0
            )

    return result


@ti.func
def quintic_spline_evaluate_integral(
    q1_in: ti.f32, q2_in: ti.f32, dim: ti.template()
) -> ti.f32:
    q2 = ti.min(q2_in, QUINTIC_SPLINE_SUPPORT)
    q1 = ti.max(q1_in, 0.0)
    result = 0.0

    if q1 <= QUINTIC_SPLINE_NODE_1 and QUINTIC_SPLINE_NODE_2 < q2:
        result = (
            quintic_spline_F(QUINTIC_SPLINE_NODE_1, dim)
            - quintic_spline_F(q1, dim)
            + (
                quintic_spline_F(QUINTIC_SPLINE_NODE_2, dim)
                - quintic_spline_F(QUINTIC_SPLINE_NODE_1 + QUINTIC_SPLINE_EPS, dim)
            )
            + (
                quintic_spline_F(q2, dim)
                - quintic_spline_F(QUINTIC_SPLINE_NODE_2 + QUINTIC_SPLINE_EPS, dim)
            )
        )
    elif q1 <= QUINTIC_SPLINE_NODE_1 and QUINTIC_SPLINE_NODE_1 < q2:
        result = (
            quintic_spline_F(QUINTIC_SPLINE_NODE_1, dim)
            - quintic_spline_F(q1, dim)
            + quintic_spline_F(q2, dim)
            - quintic_spline_F(QUINTIC_SPLINE_NODE_1 + QUINTIC_SPLINE_EPS, dim)
        )
    elif q1 <= QUINTIC_SPLINE_NODE_2 and QUINTIC_SPLINE_NODE_2 < q2:
        result = (
            quintic_spline_F(QUINTIC_SPLINE_NODE_2, dim)
            - quintic_spline_F(q1, dim)
            + quintic_spline_F(q2, dim)
            - quintic_spline_F(QUINTIC_SPLINE_NODE_2 + QUINTIC_SPLINE_EPS, dim)
        )
    else:
        result = quintic_spline_F(q2, dim) - quintic_spline_F(q1, dim)

    return result


@ti.func
def quintic_spline_gradient(q: ti.f32, grad_q, dim: ti.template()):
    dWdq = 0.0
    if q <= 1.0:
        dWdq = -5.0 * (3.0 - q) ** 4 + 30.0 * (2.0 - q) ** 4 - 75.0 * (1.0 - q) ** 4
    elif q <= 2.0:
        dWdq = -5.0 * (3.0 - q) ** 4 + 30.0 * (2.0 - q) ** 4
    elif q <= 3.0:
        dWdq = -5.0 * (3.0 - q) ** 4
    return dWdq * grad_q


# --- WendlandC2 ------------------------------------------------------------
WENDLAND_C2_SUPPORT = 2.0


@ti.func
def wendland_c2_evaluate(q: ti.f32, dim: ti.template()) -> ti.f32:
    result = 0.0
    if q < WENDLAND_C2_SUPPORT:
        z = 1.0 - 0.5 * q
        if ti.static(dim == 1):
            result = z**3 * (1.5 * q + 1.0)
        else:
            result = z**4 * (2.0 * q + 1.0)
    return result


@ti.func
def wendland_c2_sigma(dim: ti.template()) -> ti.f32:
    result = 0.0
    if ti.static(dim == 1):
        result = 5.0 / 8.0
    elif ti.static(dim == 2):
        result = 7.0 / (4.0 * pi)
    elif ti.static(dim == 3):
        result = 21.0 / (16.0 * pi)
    return result


@ti.func
def wendland_c2_F(q_in: ti.f32, dim: ti.template()) -> ti.f32:
    q = ti.max(q_in, 0.0)
    q = ti.min(q, WENDLAND_C2_SUPPORT)
    result = 0.0
    if ti.static(dim == 1):
        result = q - 0.5 * q**3 + 0.25 * q**4 - (3.0 / 80.0) * q**5
    elif ti.static(dim == 2):
        result = (q**2 / 16.0) * (
            8.0 - 10.0 * q**2 + 8.0 * q**3 - 2.5 * q**4 + (2.0 / 7.0) * q**5
        )
    elif ti.static(dim == 3):
        result = (q**3 / 16.0) * (
            16.0 / 3.0
            - 8.0 * q**2
            + (20.0 / 3.0) * q**3
            - (15.0 / 7.0) * q**4
            + 0.25 * q**5
        )
    return result


@ti.func
def wendland_c2_evaluate_integral(q1: ti.f32, q2: ti.f32, dim: ti.template()) -> ti.f32:
    return _spherical_evaluate_integral_default(
        wendland_c2_F, WENDLAND_C2_SUPPORT, dim, q1, q2
    )


@ti.func
def wendland_c2_gradient(q: ti.f32, grad_q, dim: ti.template()):
    dWdq = 0.0
    if q <= WENDLAND_C2_SUPPORT:
        z = 1.0 - 0.5 * q
        if ti.static(dim == 1):
            dWdq = -1.5 * z * z * (1.5 * q + 1.0) + 1.5 * z**3
        else:
            dWdq = -2.0 * z**3 * (2.0 * q + 1.0) + 2.0 * z**4
    return dWdq * grad_q


# --- WendlandC4 ------------------------------------------------------------
WENDLAND_C4_SUPPORT = 2.0


@ti.func
def wendland_c4_evaluate(q: ti.f32, dim: ti.template()) -> ti.f32:
    result = 0.0
    if q < WENDLAND_C4_SUPPORT:
        z = 1.0 - 0.5 * q
        if ti.static(dim == 1):
            result = z**5 * (2.0 * q * q + 2.5 * q + 1.0)
        else:
            result = z**6 * ((35.0 / 12.0) * q * q + 3.0 * q + 1.0)
    return result


@ti.func
def wendland_c4_sigma(dim: ti.template()) -> ti.f32:
    result = 0.0
    if ti.static(dim == 1):
        result = 3.0 / 4.0
    elif ti.static(dim == 2):
        result = 9.0 / (4.0 * pi)
    elif ti.static(dim == 3):
        result = 495.0 / (256.0 * pi)
    return result


@ti.func
def wendland_c4_F(q_in: ti.f32, dim: ti.template()) -> ti.f32:
    q = ti.max(q_in, 0.0)
    q = ti.min(q, WENDLAND_C4_SUPPORT)
    result = 0.0
    if ti.static(dim == 1):
        result = (q / 64.0) * (
            64.0
            - (112.0 / 3.0) * q**2
            + 28.0 * q**4
            - (56.0 / 3.0) * q**5
            + 5.0 * q**6
            - 0.5 * q**7
        )
    elif ti.static(dim == 2):
        result = (q**2 / 768.0) * (
            384.0
            - 448.0 * q**2
            + 560.0 * q**4
            - 512.0 * q**5
            + 210.0 * q**6
            - (128.0 / 3.0) * q**7
            + 3.5 * q**8
        )
    elif ti.static(dim == 3):
        result = (q**3 / 768.0) * (
            256.0
            - (1792.0 / 5.0) * q**2
            + 480.0 * q**4
            - 448.0 * q**5
            + (560.0 / 3.0) * q**6
            - (192.0 / 5.0) * q**7
            + (35.0 / 11.0) * q**8
        )
    return result


@ti.func
def wendland_c4_evaluate_integral(q1: ti.f32, q2: ti.f32, dim: ti.template()) -> ti.f32:
    return _spherical_evaluate_integral_default(
        wendland_c4_F, WENDLAND_C4_SUPPORT, dim, q1, q2
    )


@ti.func
def wendland_c4_gradient(q: ti.f32, grad_q, dim: ti.template()):
    dWdq = 0.0
    if q <= WENDLAND_C4_SUPPORT:
        z = 1.0 - 0.5 * q
        f, df, g, dg = 0.0, 0.0, 0.0, 0.0
        if ti.static(dim == 1):
            f = z**5
            df = -2.5 * z**4
            g = 2.0 * q * q + 2.5 * q + 1.0
            dg = 4.0 * q + 2.5
        else:
            f = z**6
            df = -3.0 * z**5
            g = (35.0 / 12.0) * q * q + 3.0 * q + 1.0
            dg = (35.0 / 6.0) * q + 3.0
        dWdq = df * g + f * dg
    return dWdq * grad_q


# --- WendlandC6 ------------------------------------------------------------
WENDLAND_C6_SUPPORT = 2.0


@ti.func
def wendland_c6_evaluate(q: ti.f32, dim: ti.template()) -> ti.f32:
    result = 0.0
    if q < WENDLAND_C6_SUPPORT:
        z = 1.0 - 0.5 * q
        if ti.static(dim == 1):
            result = z**7 * ((21.0 / 8.0) * q**3 + (19.0 / 4.0) * q**2 + 3.5 * q + 1.0)
        else:
            result = z**8 * (4.0 * q**3 + 6.25 * q**2 + 4.0 * q + 1.0)
    return result


@ti.func
def wendland_c6_sigma(dim: ti.template()) -> ti.f32:
    result = 0.0
    if ti.static(dim == 1):
        result = 55.0 / 64.0
    elif ti.static(dim == 2):
        result = 39.0 / (14.0 * pi)
    elif ti.static(dim == 3):
        result = 1365.0 / (512.0 * pi)
    return result


@ti.func
def wendland_c6_F(q_in: ti.f32, dim: ti.template()) -> ti.f32:
    q = ti.max(q_in, 0.0)
    q = ti.min(q, WENDLAND_C6_SUPPORT)
    result = 0.0
    if ti.static(dim == 1):
        result = (1.0 / 1024.0) * (
            1024.0 * q
            - 768.0 * q**3
            + (2688.0 / 5.0) * q**5
            - 480.0 * q**7
            + 384.0 * q**8
            - 140.0 * q**9
            + (128.0 / 5.0) * q**10
            - (21.0 / 11.0) * q**11
        )
    elif ti.static(dim == 2):
        result = (
            0.5
            * q**2
            * (
                1.0
                - (11.0 / 8.0) * q**2
                + (11.0 / 8.0) * q**4
                - (231.0 / 128.0) * q**6
                + (11.0 / 6.0) * q**7
                - (231.0 / 256.0) * q**8
                + 0.25 * q**9
                - (77.0 / 2048.0) * q**10
                + (1.0 / 416.0) * q**11
            )
        )
    elif ti.static(dim == 3):
        result = q**3 * (
            1.0 / 3.0
            - (11.0 / 20.0) * q**2
            + (33.0 / 56.0) * q**4
            - (77.0 / 96.0) * q**6
            + (33.0 / 40.0) * q**7
            - (105.0 / 256.0) * q**8
            + (11.0 / 96.0) * q**9
            - (231.0 / 13312.0) * q**10
            + (1.0 / 896.0) * q**11
        )
    return result


@ti.func
def wendland_c6_evaluate_integral(q1: ti.f32, q2: ti.f32, dim: ti.template()) -> ti.f32:
    return _spherical_evaluate_integral_default(
        wendland_c6_F, WENDLAND_C6_SUPPORT, dim, q1, q2
    )


@ti.func
def wendland_c6_gradient(q: ti.f32, grad_q, dim: ti.template()):
    dWdq = 0.0
    if q <= WENDLAND_C6_SUPPORT:
        z = 1.0 - 0.5 * q
        f, df, g, dg = 0.0, 0.0, 0.0, 0.0
        if ti.static(dim == 1):
            f = z**7
            df = -3.5 * z**6
            g = (21.0 / 8.0) * q**3 + (19.0 / 4.0) * q * q + 3.5 * q + 1.0
            dg = (63.0 / 8.0) * q * q + (19.0 / 2.0) * q + 3.5
        else:
            f = z**8
            df = -4.0 * z**7
            g = 4.0 * q**3 + 6.25 * q * q + 4.0 * q + 1.0
            dg = 12.0 * q * q + 12.5 * q + 4.0
        dWdq = df * g + f * dg
    return dWdq * grad_q


# =============================================================================
# Registries
# =============================================================================
@dataclass(frozen=True)
class SeparableKernelSpec:
    """Specification for a separable kernel.

    Parameters
    ----------
    evaluate_1d : Callable
        A function that evaluates the kernel at a given distance in 1D.
    F_1d : Callable
        A function that evaluates the kernel function at a given distance in 1D.
    sigma : Callable
        A function that returns the scale parameter for a given dimension.
    support : float
        The support radius of the kernel.

    """

    evaluate_1d: Callable  # ti.func(q, dim) -> f32
    F_1d: Callable  # ti.func(q, dim) -> f32
    sigma: Callable  # ti.func(dim) -> f32
    support: float


@dataclass(frozen=True)
class SphericalKernelSpec:
    """Specification for a spherical kernel.

    Parameters
    ----------
    evaluate : Callable
        A function that evaluates the kernel at a given distance.
    F : Callable
        A function that evaluates the kernel function at a given distance.
    sigma : Callable
        A function that returns the scale parameter for a given dimension.
    evaluate_integral : Callable
        A function that evaluates the integral of the kernel over a given range.
    gradient : Callable
        A function that evaluates the gradient of the kernel at a given distance.
    support : float
        The support radius of the kernel.

    """

    evaluate: Callable  # ti.func(q, dim) -> f32
    F: Callable  # ti.func(q, dim) -> f32
    sigma: Callable  # ti.func(dim) -> f32
    evaluate_integral: Callable  # ti.func(q1, q2, dim) -> f32
    gradient: Callable  # ti.func(q, grad_q, dim) -> ti.Vector(dim)  [dW/dq * grad_q]
    support: float


SEPARABLE_KERNELS = {
    "tophat": SeparableKernelSpec(
        evaluate_1d=tophat_sep_evaluate_1d,
        F_1d=tophat_sep_F_1d,
        sigma=tophat_sep_sigma,
        support=TOPHAT_RECT_SUPPORT,
    ),
    "tsc": SeparableKernelSpec(
        evaluate_1d=tsc_sep_evaluate_1d,
        F_1d=tsc_sep_F_1d,
        sigma=tsc_sep_sigma,
        support=TSC_RECT_SUPPORT,
    ),
    "gaussian": SeparableKernelSpec(
        evaluate_1d=gaussian_sep_evaluate_1d,
        F_1d=gaussian_sep_F_1d,
        sigma=gaussian_sep_sigma,
        support=GAUSSIAN_RECT_SUPPORT,
    ),
}

SPHERICAL_KERNELS = {
    "tophat": SphericalKernelSpec(
        evaluate=tophat_evaluate,
        F=tophat_F,
        sigma=tophat_sigma,
        evaluate_integral=tophat_evaluate_integral,
        gradient=tophat_gradient,
        support=TOPHAT_SUPPORT,
    ),
    "tsc": SphericalKernelSpec(
        evaluate=tsc_evaluate,
        F=tsc_F,
        sigma=tsc_sigma,
        evaluate_integral=tsc_evaluate_integral,
        gradient=tsc_gradient,
        support=TSC_SUPPORT,
    ),
    "lucy": SphericalKernelSpec(
        evaluate=lucy_evaluate,
        F=lucy_F,
        sigma=lucy_sigma,
        evaluate_integral=lucy_evaluate_integral,
        gradient=lucy_gradient,
        support=LUCY_SUPPORT,
    ),
    "gaussian": SphericalKernelSpec(
        evaluate=gaussian_evaluate,
        F=gaussian_F,
        sigma=gaussian_sigma,
        evaluate_integral=gaussian_evaluate_integral,
        gradient=gaussian_gradient,
        support=GAUSSIAN_SUPPORT,
    ),
    "cubic_spline": SphericalKernelSpec(
        evaluate=cubic_spline_evaluate,
        F=cubic_spline_F,
        sigma=cubic_spline_sigma,
        evaluate_integral=cubic_spline_evaluate_integral,
        gradient=cubic_spline_gradient,
        support=CUBIC_SPLINE_SUPPORT,
    ),
    "quintic_spline": SphericalKernelSpec(
        evaluate=quintic_spline_evaluate,
        F=quintic_spline_F,
        sigma=quintic_spline_sigma,
        evaluate_integral=quintic_spline_evaluate_integral,
        gradient=quintic_spline_gradient,
        support=QUINTIC_SPLINE_SUPPORT,
    ),
    "wendland_c2": SphericalKernelSpec(
        evaluate=wendland_c2_evaluate,
        F=wendland_c2_F,
        sigma=wendland_c2_sigma,
        evaluate_integral=wendland_c2_evaluate_integral,
        gradient=wendland_c2_gradient,
        support=WENDLAND_C2_SUPPORT,
    ),
    "wendland_c4": SphericalKernelSpec(
        evaluate=wendland_c4_evaluate,
        F=wendland_c4_F,
        sigma=wendland_c4_sigma,
        evaluate_integral=wendland_c4_evaluate_integral,
        gradient=wendland_c4_gradient,
        support=WENDLAND_C4_SUPPORT,
    ),
    "wendland_c6": SphericalKernelSpec(
        evaluate=wendland_c6_evaluate,
        F=wendland_c6_F,
        sigma=wendland_c6_sigma,
        evaluate_integral=wendland_c6_evaluate_integral,
        gradient=wendland_c6_gradient,
        support=WENDLAND_C6_SUPPORT,
    ),
}


def create_separable_kernel(name: str) -> SeparableKernelSpec:
    """Create a SeparableKernelSpec from a kernel name.

    Parameters
    ----------
    name : str
        The name of the kernel. Must be one of the keys in SEPARABLE_KERNELS.

    Returns
    -------
    SeparableKernelSpec
        The kernel spec corresponding to the given name.

    Raises
    ------
    ValueError
        If the kernel name is not found in SEPARABLE_KERNELS.

    """
    try:
        return SEPARABLE_KERNELS[name]
    except KeyError:
        raise ValueError(f"Unknown kernel: {name!r}") from None


def create_spherical_kernel(name: str) -> SphericalKernelSpec:
    """Create a SphericalKernelSpec from a kernel name.

    Parameters
    ----------
    name : str
        The name of the kernel. Must be one of the keys in SPHERICAL_KERNELS.

    Returns
    -------
    SphericalKernelSpec
        The kernel spec corresponding to the given name.

    Raises
    ------
    ValueError
        If the kernel name is not found in SPHERICAL_KERNELS.

    """
    try:
        return SPHERICAL_KERNELS[name]
    except KeyError:
        raise ValueError(f"Unknown kernel: {name!r}") from None


# =============================================================================
# Sample grid kernels and wrapper
# =============================================================================
# One ti.kernel per dimension, mirroring the dim==1/2/3 branches in the C++.
# `evaluate_integral_fn` and `sigma_fn` are passed as ti.template() so the
# chosen kernel's math is fully inlined -- no virtual call, no runtime
# dispatch on which kernel this is.


# --------------------------------------------------------------------------
# Sizing helpers -- purely functions of eta_crit (and nyquist_factor). eta is
# a DIAMETER ratio (eta = 2 * support * h / cell_size); nyquist_factor=2.0 is
# the theoretical Nyquist minimum for the point spacing (in cell-size units)
# needed to avoid aliasing when depositing the resulting samples onto a grid.
# --------------------------------------------------------------------------


def _compute_n_q(eta_crit: float, nyquist_factor: float = 2.0) -> int:
    """Number of positive-radial shells.

    eta_crit is a diameter ratio; Nyquist spacing applies to the radius, so
    the effective radial eta_crit is eta_crit / 2.
    """
    eta_crit_radius = eta_crit / 2.0
    return max(1, int(np.ceil(nyquist_factor * eta_crit_radius)))


def _n_points_circle_shell(
    u_outer: float, eta_crit_radius: float, nyquist_factor: float
) -> int:
    """2D: number of uniformly-spaced angular samples needed on a shell of
    (normalized) outer radius u_outer, to keep point spacing <= s/nyquist_factor
    in the worst case (eta -> eta_crit).
    """
    delta = 1.0 / nyquist_factor
    circumference = 2.0 * pi * eta_crit_radius * u_outer
    return max(1, int(np.ceil(circumference / delta)))


def _n_points_sphere_shell(
    u_outer: float, eta_crit_radius: float, nyquist_factor: float
) -> int:
    """3D: number of quasi-uniformly-spaced directional samples needed on a
    shell of (normalized) outer radius u_outer.
    """
    delta = 1.0 / nyquist_factor
    r = eta_crit_radius * u_outer
    area = 4.0 * pi * r * r
    return max(1, int(np.ceil(area / (delta * delta))))


def _fibonacci_sphere_points(n: int) -> np.ndarray:
    """N quasi-uniformly distributed unit vectors on the sphere (Fibonacci
    spiral point set). No pole clustering, deterministic, O(n).
    """
    i = np.arange(n, dtype=np.float64)
    golden_angle = pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - (2.0 * i + 1.0) / n  # uniform in z, range (-1, 1)
    r_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = golden_angle * i
    x = r_xy * np.cos(phi)
    y = r_xy * np.sin(phi)
    return np.stack([x, y, z], axis=1).astype(np.float32)  # (n, 3)


# --------------------------------------------------------------------------
# Host-side (numpy) ragged shell-layout builders. These only produce bin
# boundaries / directions -- no kernel evaluation happens here, so they
# don't need Taichi scope.
# --------------------------------------------------------------------------


def _build_shell_layout_1d(
    support: float, eta_crit: float, nyquist_factor: float = 2.0
):
    """1D: each radial shell contributes exactly two points, at +q and -q."""
    n_q = _compute_n_q(eta_crit, nyquist_factor)
    dq = support / n_q
    shell_q0 = (np.arange(n_q, dtype=np.float32)) * dq
    shell_q1 = shell_q0 + dq

    shell_id = np.repeat(np.arange(n_q, dtype=np.int32), 2)
    sign = np.tile(np.array([1.0, -1.0], dtype=np.float32), n_q)
    n_points_per_shell = np.full(n_q, 2, dtype=np.int32)

    return n_q, shell_q0, shell_q1, shell_id, sign, n_points_per_shell


def _build_shell_layout_2d(
    support: float, eta_crit: float, nyquist_factor: float = 2.0
):
    """2D: uniform angular samples per shell (no pole-clustering issue in
    2D -- a circle has no poles -- so uniform angle spacing is already
    spherically-uniform).
    """
    n_q = _compute_n_q(eta_crit, nyquist_factor)
    eta_crit_radius = eta_crit / 2.0
    dq = support / n_q
    shell_q0 = np.arange(n_q, dtype=np.float32) * dq
    shell_q1 = shell_q0 + dq

    shell_id_chunks, theta_chunks, n_points_per_shell = [], [], []
    for iq in range(n_q):
        u_outer = (iq + 1) / n_q
        n_pts = _n_points_circle_shell(u_outer, eta_crit_radius, nyquist_factor)
        dtheta = 2.0 * pi / n_pts
        theta = (np.arange(n_pts, dtype=np.float32) + 0.5) * dtheta  # bin midpoints

        shell_id_chunks.append(np.full(n_pts, iq, dtype=np.int32))
        theta_chunks.append(theta)
        n_points_per_shell.append(n_pts)

    shell_id = np.concatenate(shell_id_chunks)
    theta = np.concatenate(theta_chunks)
    n_points_per_shell = np.array(n_points_per_shell, dtype=np.int32)

    return n_q, shell_q0, shell_q1, shell_id, theta, n_points_per_shell


def _build_shell_layout_3d(
    support: float, eta_crit: float, nyquist_factor: float = 2.0
):
    """3D: Fibonacci-sphere directional samples per shell -- quasi-uniform
    coverage of the sphere, no pole clustering.
    """
    n_q = _compute_n_q(eta_crit, nyquist_factor)
    eta_crit_radius = eta_crit / 2.0
    dq = support / n_q
    shell_q0 = np.arange(n_q, dtype=np.float32) * dq
    shell_q1 = shell_q0 + dq

    shell_id_chunks, dirs_chunks, n_points_per_shell = [], [], []
    for iq in range(n_q):
        u_outer = (iq + 1) / n_q
        n_pts = _n_points_sphere_shell(u_outer, eta_crit_radius, nyquist_factor)
        dirs = _fibonacci_sphere_points(n_pts)

        shell_id_chunks.append(np.full(n_pts, iq, dtype=np.int32))
        dirs_chunks.append(dirs)
        n_points_per_shell.append(n_pts)

    shell_id = np.concatenate(shell_id_chunks)
    dirs = np.concatenate(dirs_chunks, axis=0)  # (total, 3) unit vectors
    n_points_per_shell = np.array(n_points_per_shell, dtype=np.int32)

    return n_q, shell_q0, shell_q1, shell_id, dirs, n_points_per_shell


# --------------------------------------------------------------------------
# Taichi kernel: radial integral, computed ONCE per shell (n_q evaluations
# total, shared across every direction/angle sampled within that shell).
# --------------------------------------------------------------------------


@ti.kernel
def _compute_radial_integrals(
    evaluate_integral_fn: ti.template(),
    dim: ti.template(),
    n_q: ti.i32,
    shell_q0: ti.types.ndarray(dtype=ti.f32, ndim=1),
    shell_q1: ti.types.ndarray(dtype=ti.f32, ndim=1),
    radial_integral_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for iq in range(n_q):
        radial_integral_out[iq] = evaluate_integral_fn(shell_q0[iq], shell_q1[iq], dim)


# --------------------------------------------------------------------------
# Taichi kernels: populate coords / q / integrals from the precomputed shell
# layout + radial integrals. No calls to evaluate_integral_fn here.
# --------------------------------------------------------------------------


@ti.kernel
def _build_kernel_sample_grid_1d(
    sigma_fn: ti.template(),
    dim: ti.template(),
    n_pts: ti.i32,
    shell_id_in: ti.types.ndarray(dtype=ti.i32, ndim=1),
    shell_q0: ti.types.ndarray(dtype=ti.f32, ndim=1),
    shell_q1: ti.types.ndarray(dtype=ti.f32, ndim=1),
    radial_integral_in: ti.types.ndarray(dtype=ti.f32, ndim=1),
    n_points_per_shell_in: ti.types.ndarray(dtype=ti.i32, ndim=1),
    sign_in: ti.types.ndarray(dtype=ti.f32, ndim=1),
    coords_out: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (n_pts, 1)
    q_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
    integrals_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    sig = sigma_fn(dim)
    for i in range(n_pts):
        iq = shell_id_in[i]
        q0, q1 = shell_q0[iq], shell_q1[iq]
        q = 0.5 * (q0 + q1)

        coords_out[i, 0] = sign_in[i] * q
        q_out[i] = q
        # evaluate_integral_fn is a pure radial integral; the angular
        # measure for a 1D "shell" (two points, +/-q) is 2. sigma * (2 *
        # radial_integral) is the shell's total mass, split evenly across
        # this shell's points (spherically symmetric -> every point in a
        # shell carries an equal fraction of that shell's mass).
        integrals_out[i] = (
            sig * 2.0 * radial_integral_in[iq] / ti.f32(n_points_per_shell_in[iq])
        )


@ti.kernel
def _build_kernel_sample_grid_2d(
    sigma_fn: ti.template(),
    dim: ti.template(),
    n_pts: ti.i32,
    shell_id_in: ti.types.ndarray(dtype=ti.i32, ndim=1),
    shell_q0: ti.types.ndarray(dtype=ti.f32, ndim=1),
    shell_q1: ti.types.ndarray(dtype=ti.f32, ndim=1),
    radial_integral_in: ti.types.ndarray(dtype=ti.f32, ndim=1),
    n_points_per_shell_in: ti.types.ndarray(dtype=ti.i32, ndim=1),
    theta_in: ti.types.ndarray(dtype=ti.f32, ndim=1),
    coords_out: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (n_pts, 2)
    q_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
    integrals_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    sig = sigma_fn(dim)
    for i in range(n_pts):
        iq = shell_id_in[i]
        q0, q1 = shell_q0[iq], shell_q1[iq]
        q = 0.5 * (q0 + q1)
        theta = theta_in[i]

        coords_out[i, 0] = q * ti.cos(theta)
        coords_out[i, 1] = q * ti.sin(theta)
        q_out[i] = q
        # evaluate_integral_fn is a pure radial integral; the angular
        # measure for a full 2D shell (circle) is 2*pi.
        integrals_out[i] = (
            sig
            * (2.0 * pi)
            * radial_integral_in[iq]
            / ti.f32(n_points_per_shell_in[iq])
        )


@ti.kernel
def _build_kernel_sample_grid_3d(
    sigma_fn: ti.template(),
    dim: ti.template(),
    n_pts: ti.i32,
    shell_id_in: ti.types.ndarray(dtype=ti.i32, ndim=1),
    shell_q0: ti.types.ndarray(dtype=ti.f32, ndim=1),
    shell_q1: ti.types.ndarray(dtype=ti.f32, ndim=1),
    radial_integral_in: ti.types.ndarray(dtype=ti.f32, ndim=1),
    n_points_per_shell_in: ti.types.ndarray(dtype=ti.i32, ndim=1),
    dirs_in: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (n_pts, 3) unit vectors
    coords_out: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (n_pts, 3)
    q_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
    integrals_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    sig = sigma_fn(dim)
    for i in range(n_pts):
        iq = shell_id_in[i]
        q0, q1 = shell_q0[iq], shell_q1[iq]
        q = 0.5 * (q0 + q1)

        coords_out[i, 0] = q * dirs_in[i, 0]
        coords_out[i, 1] = q * dirs_in[i, 1]
        coords_out[i, 2] = q * dirs_in[i, 2]
        q_out[i] = q
        # evaluate_integral_fn is a pure radial integral; the angular
        # measure for a full 3D shell (sphere) is 4*pi.
        integrals_out[i] = (
            sig
            * (4.0 * pi)
            * radial_integral_in[iq]
            / ti.f32(n_points_per_shell_in[iq])
        )


# --------------------------------------------------------------------------
# Public orchestrator
# --------------------------------------------------------------------------


def build_kernel_sample_grid(
    kernel_name: str,
    dim: int,
    eta_crit: float,
    nyquist_factor: float = 2.0,
):
    """Build a Nyquist-safe, spherically-uniform kernel sample table.

    The table geometry (n_q, per-shell point counts, directions) is sized
    purely from `eta_crit` -- independent of any specific particle's h, the
    grid's cell size, or the kernel's own support multiple -- so the
    returned table is safe (by construction) for every particle whose
    actual eta < eta_crit.

    Parameters
    ----------
    kernel_name : str
        Name of the kernel to sample (looked up via create_spherical_kernel).
    dim : int
        Dimension of the kernel (1, 2, or 3).
    eta_crit : float
        Critical eta (diameter convention: eta = 2 * support * h / cell_size)
        at which the deposition loop switches from tabulated samples to
        numerical quadrature.
    nyquist_factor : float
        Samples-per-cell-width safety margin. 2.0 is the Nyquist minimum;
        values > 2.0 add extra safety margin at the cost of more points.

    Returns
    -------
    dict
        - "dim": int
        - "count": int, total number of samples
        - "n_q": int, number of positive-radial shells (diagnostic)
        - "coords": np.ndarray (count, dim), sample coordinates in units of
          q = r/h (i.e. NOT yet scaled by any particle's physical h)
        - "q": np.ndarray (count,), radial distance in units of q = r/h
        - "integrals": np.ndarray (count,), normalized kernel mass
          contribution of each sample (sums to ~1.0 over the full table)

    """
    if eta_crit < 0:
        raise ValueError("eta_crit must be >= 0")
    if dim not in (1, 2, 3):
        raise ValueError("build_kernel_sample_grid supports only dim = 1, 2 or 3")

    kspec = create_spherical_kernel(kernel_name)
    support = kspec.support  # kernel's own cutoff in q = r/h units

    if dim == 1:
        n_q, shell_q0, shell_q1, shell_id, sign, n_points_per_shell = (
            _build_shell_layout_1d(support, eta_crit, nyquist_factor)
        )
        count = shell_id.shape[0]
        radial_integral = np.zeros(n_q, dtype=np.float32)
        _compute_radial_integrals(
            kspec.evaluate_integral, dim, n_q, shell_q0, shell_q1, radial_integral
        )

        coords = np.zeros((count, 1), dtype=np.float32)
        q = np.zeros(count, dtype=np.float32)
        integrals = np.zeros(count, dtype=np.float32)
        _build_kernel_sample_grid_1d(
            kspec.sigma,
            dim,
            count,
            shell_id,
            shell_q0,
            shell_q1,
            radial_integral,
            n_points_per_shell,
            sign,
            coords,
            q,
            integrals,
        )

    elif dim == 2:
        n_q, shell_q0, shell_q1, shell_id, theta, n_points_per_shell = (
            _build_shell_layout_2d(support, eta_crit, nyquist_factor)
        )
        count = shell_id.shape[0]
        radial_integral = np.zeros(n_q, dtype=np.float32)
        _compute_radial_integrals(
            kspec.evaluate_integral, dim, n_q, shell_q0, shell_q1, radial_integral
        )

        coords = np.zeros((count, 2), dtype=np.float32)
        q = np.zeros(count, dtype=np.float32)
        integrals = np.zeros(count, dtype=np.float32)
        _build_kernel_sample_grid_2d(
            kspec.sigma,
            dim,
            count,
            shell_id,
            shell_q0,
            shell_q1,
            radial_integral,
            n_points_per_shell,
            theta,
            coords,
            q,
            integrals,
        )

    else:  # dim == 3
        n_q, shell_q0, shell_q1, shell_id, dirs, n_points_per_shell = (
            _build_shell_layout_3d(support, eta_crit, nyquist_factor)
        )
        count = shell_id.shape[0]
        radial_integral = np.zeros(n_q, dtype=np.float32)
        _compute_radial_integrals(
            kspec.evaluate_integral, dim, n_q, shell_q0, shell_q1, radial_integral
        )

        coords = np.zeros((count, 3), dtype=np.float32)
        q = np.zeros(count, dtype=np.float32)
        integrals = np.zeros(count, dtype=np.float32)
        _build_kernel_sample_grid_3d(
            kspec.sigma,
            dim,
            count,
            shell_id,
            shell_q0,
            shell_q1,
            radial_integral,
            n_points_per_shell,
            dirs,
            coords,
            q,
            integrals,
        )

    return {
        "dim": dim,
        "count": count,
        "n_q": n_q,
        "coords": coords,
        "q": q,
        "integrals": integrals,
    }


# =============================================================================
# Utility functions (debugging / testing parity with the C++ helpers)
# =============================================================================
def compute_total_integral_separable(kernel_name: str, dim: int) -> float:
    """Sigma * evaluate_integral(bounds) over the box [-support, support]^dim.

    Parameters
    ----------
    kernel_name : str
        Name of the kernel to integrate.
    dim : int
        Dimension of the kernel (1, 2, or 3).

    Returns
    -------
    float
        Total integral of the kernel over the box [-support, support]^dim.

    """
    ti.init(arch=ti.cpu)
    kspec = create_separable_kernel(kernel_name)

    @ti.kernel
    def _compute(
        F_1d_fn: ti.template(), support: ti.f32, dim_: ti.template()
    ) -> ti.f32:
        return _separable_integrate_1d(F_1d_fn, support, dim_, -support, support)

    single_axis_integral = _compute(kspec.F_1d, kspec.support, dim)

    @ti.kernel
    def _sigma(sigma_fn: ti.template(), dim_: ti.template()) -> ti.f32:
        return sigma_fn(dim_)

    sigma = _sigma(kspec.sigma, dim)
    return float(sigma * (single_axis_integral**dim))


def compute_total_integral_spherical(
    kernel_name: str, dim: int, eta_crit: float
) -> float:
    """Sigma * evaluate_integral(bounds) over the box [-support, support]^dim.

    Parameters
    ----------
    kernel_name : str
        Name of the kernel to integrate.
    dim : int
        Dimension of the kernel (1, 2, or 3).
    eta_crit : float
        Critical eta (diameter convention) used to size the sample grid --
        see `build_kernel_sample_grid`.

    Returns
    -------
    float
        Total integral of the kernel over the box [-support, support]^dim.

    """
    ti.init(arch=ti.cpu)
    grid = build_kernel_sample_grid(kernel_name, dim, eta_crit)
    return float(np.sum(grid["integrals"]))


@ti.kernel
def _sample_spherical_1d(
    evaluate_fn: ti.template(),
    sigma_fn: ti.template(),
    support: ti.f32,
    dim: ti.template(),
    num_samples: ti.i32,
    q_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
    values_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    dq = 2.0 * support / num_samples
    sig = sigma_fn(dim)
    for i in range(num_samples):
        q_current = -support + i * dq
        value = sig * evaluate_fn(ti.abs(q_current), dim)
        q_out[i] = q_current
        values_out[i] = value


@ti.kernel
def _sample_separable_1d(
    evaluate_1d_fn: ti.template(),
    sigma_fn: ti.template(),
    support: ti.f32,
    dim: ti.template(),
    num_samples: ti.i32,
    q_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
    values_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    dq = 2.0 * support / num_samples
    sig = sigma_fn(dim)
    for i in range(num_samples):
        q_current = -support + i * dq
        value = sig * evaluate_1d_fn(ti.abs(q_current), dim)
        q_out[i] = q_current
        values_out[i] = value


def get_spherical_kernel_values_1D(kernel_name: str):
    """Get 1D values of a separable kernel for testing and plotting.

    Parameters
    ----------
    kernel_name : str
        Name of the separable kernel to sample.

    Returns
    -------
    q : np.ndarray
        1D array of sample points in the range [-support, support].
    values : np.ndarray
        1D array of kernel values corresponding to the sample points.

    """
    ti.init(arch=ti.cpu)
    kspec = create_spherical_kernel(kernel_name)
    num_samples = 100
    q = np.zeros(num_samples, dtype=np.float32)
    values = np.zeros(num_samples, dtype=np.float32)
    _sample_spherical_1d(
        kspec.evaluate, kspec.sigma, kspec.support, 1, num_samples, q, values
    )
    return q, values


def get_separable_kernel_values_1D(kernel_name: str):
    """Get 1D values of a separable kernel for testing and plotting.

    Parameters
    ----------
    kernel_name : str
        Name of the separable kernel to sample.

    Returns
    -------
    q : np.ndarray
        1D array of sample points in the range [-support, support].
    values : np.ndarray
        1D array of kernel values corresponding to the sample points.

    """
    ti.init(arch=ti.cpu)
    kspec = create_separable_kernel(kernel_name)
    num_samples = 100
    q = np.zeros(num_samples, dtype=np.float32)
    values = np.zeros(num_samples, dtype=np.float32)
    _sample_separable_1d(
        kspec.evaluate_1d, kspec.sigma, kspec.support, 1, num_samples, q, values
    )
    return q, values
