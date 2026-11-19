"""Quadrants implementations of the separable and spherical smoothing kernels.

Kernels are looked up by name in the ``SEPARABLE_KERNELS`` and
``SPHERICAL_KERNELS`` registries; the dimension is a compile-time parameter,
so every (kernel, dimension) pair is compiled into its own specialized code.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from math import pi

import numpy as np
import quadrants as qd

from ..kernel_table import SEPARABLE, SPHERICAL


def _make_sigma(values):
    """qd.func(dim) -> normalisation constant, from a kernel_table entry."""
    s1, s2, s3 = values

    @qd.func
    def sigma(dim: qd.template()) -> qd.f32:
        result = 0.0
        if qd.static(dim == 1):
            result = s1
        elif qd.static(dim == 2):
            result = s2
        else:
            result = s3
        return result

    return sigma


# =============================================================================
# Helper functions for projections onto canonical coordinates
# Used for density computation and interpolation workflow (deposit does it internally)
# =============================================================================
@qd.func
def prepare_isotropic_inputs(
    r_vec, h: qd.f32, dim: qd.template(), sigma_val: qd.f32, eps: qd.f32
):
    r_mag = r_vec.norm()
    q = r_mag / (h + eps)
    scale = sigma_val / (h**dim)

    grad_q = qd.Vector.zero(qd.f32, dim)
    if r_mag > 0.0:
        grad_q = r_vec / ((r_mag + eps) * (h + eps))

    return q, grad_q, scale


@qd.func
def covariant_inputs_from_inverse(
    r_vec, H_inv, inv_det: qd.f32, dim: qd.template(), sigma_val: qd.f32, eps: qd.f32
):
    """Kernel coordinate q = |H^-1 r|, its gradient and the prefactor
    sigma / det(H), from the precomputed H^-1 and 1/det(H) of one particle."""
    xi = H_inv @ r_vec
    q = xi.norm()
    scale = sigma_val * inv_det

    grad_q = qd.Vector.zero(qd.f32, dim)
    if q > 0.0:
        grad_q = (H_inv.transpose() @ xi) / (q + eps)

    return q, grad_q, scale


# =============================================================================
# Shared low-level helpers
# =============================================================================
@qd.func
def _erf(x: qd.f32) -> qd.f32:
    """Abramowitz & Stegun 7.1.26 approximation, max abs error ~1.5e-7.
    Quadrants has no built-in erf, needed for the Gaussian kernels' F(q).
    """
    sign = 1.0
    if x < 0.0:
        sign = -1.0
    ax = qd.abs(x)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * qd.exp(-ax * ax)
    return sign * y


@qd.func
def _spherical_evaluate_integral_default(
    F_fn: qd.template(), support: qd.f32, dim: qd.template(), q1: qd.f32, q2_in: qd.f32
) -> qd.f32:
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


@qd.func
def _separable_integrate_1d(
    F_1d_fn: qd.template(),
    support: qd.f32,
    dim: qd.template(),
    q0_in: qd.f32,
    q1_in: qd.f32,
) -> qd.f32:
    """SeparableKernel::integrate_1d -- 1D integral between two bounds,
    handling the kernel's even symmetry (evaluated via F_1d on |q|).
    """
    result = 0.0
    q0 = q0_in
    q1 = q1_in
    if q1 > q0:
        q0 = qd.max(q0, -support)
        q1 = qd.min(q1, support)
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
TOPHAT_RECT_SUPPORT = SEPARABLE["tophat"].support


@qd.func
def tophat_sep_evaluate_1d(q_in: qd.f32, dim: qd.template()) -> qd.f32:
    q = qd.abs(q_in)
    result = 0.0
    if q <= TOPHAT_RECT_SUPPORT:
        result = 1.0
    return result


tophat_sep_sigma = _make_sigma(SEPARABLE["tophat"].sigma)


@qd.func
def tophat_sep_F_1d(q_in: qd.f32, dim: qd.template()) -> qd.f32:
    q = qd.max(q_in, 0.0)
    q = qd.min(q, TOPHAT_RECT_SUPPORT)
    return q


# --- TSCRect -------------------------------------------------------------
TSC_RECT_SUPPORT = SEPARABLE["tsc"].support


@qd.func
def tsc_sep_evaluate_1d(q_in: qd.f32, dim: qd.template()) -> qd.f32:
    q_abs = qd.abs(q_in)
    result = 0.0
    if q_abs <= TSC_RECT_SUPPORT:
        result = 1.0 - q_abs / TSC_RECT_SUPPORT
    return result


tsc_sep_sigma = _make_sigma(SEPARABLE["tsc"].sigma)


@qd.func
def tsc_sep_F_1d(q_in: qd.f32, dim: qd.template()) -> qd.f32:
    q = qd.max(q_in, 0.0)
    q = qd.min(q, TSC_RECT_SUPPORT)
    return q - (q * q) / (2.0 * TSC_RECT_SUPPORT)


# --- GaussianRect ----------------------------------------------------------
GAUSSIAN_RECT_SUPPORT = SEPARABLE["gaussian"].support


@qd.func
def gaussian_sep_evaluate_1d(q_in: qd.f32, dim: qd.template()) -> qd.f32:
    q = qd.abs(q_in)
    result = 0.0
    if q <= GAUSSIAN_RECT_SUPPORT:
        result = qd.exp(-q * q)
    return result


gaussian_sep_sigma = _make_sigma(SEPARABLE["gaussian"].sigma)


@qd.func
def gaussian_sep_F_1d(q_in: qd.f32, dim: qd.template()) -> qd.f32:
    q = qd.max(q_in, 0.0)
    q = qd.min(q, GAUSSIAN_RECT_SUPPORT)
    return 0.5 * qd.sqrt(pi) * _erf(q)


# =============================================================================
# Spherical kernels (SphericalKernel subclasses)
# =============================================================================
# --- Tophat ----------------------------------------------------------------
TOPHAT_SUPPORT = SPHERICAL["tophat"].support


@qd.func
def tophat_evaluate(q: qd.f32, dim: qd.template()) -> qd.f32:
    result = 0.0
    if q <= TOPHAT_SUPPORT:
        result = 1.0
    return result


tophat_sigma = _make_sigma(SPHERICAL["tophat"].sigma)


@qd.func
def tophat_F(q_in: qd.f32, dim: qd.template()) -> qd.f32:
    q = qd.max(q_in, 0.0)
    q = qd.min(q, TOPHAT_SUPPORT)
    result = 0.0
    if qd.static(dim == 1):
        result = q
    elif qd.static(dim == 2):
        result = 0.5 * q * q
    elif qd.static(dim == 3):
        result = (1.0 / 3.0) * q * q * q
    return result


@qd.func
def tophat_evaluate_integral(q1: qd.f32, q2: qd.f32, dim: qd.template()) -> qd.f32:
    return _spherical_evaluate_integral_default(tophat_F, TOPHAT_SUPPORT, dim, q1, q2)


@qd.func
def tophat_gradient(q: qd.f32, grad_q, dim: qd.template()):
    return 0.0 * grad_q


# --- TSC ---------------------------------------------------------------
TSC_SUPPORT = SPHERICAL["tsc"].support


@qd.func
def tsc_evaluate(q_in: qd.f32, dim: qd.template()) -> qd.f32:
    q = qd.abs(q_in)
    result = 0.0
    if q <= TSC_SUPPORT:
        result = 1.0 - q / TSC_SUPPORT
    return result


tsc_sigma = _make_sigma(SPHERICAL["tsc"].sigma)


@qd.func
def tsc_F(q_in: qd.f32, dim: qd.template()) -> qd.f32:
    q = qd.max(q_in, 0.0)
    q = qd.min(q, TSC_SUPPORT)
    h = TSC_SUPPORT
    result = 0.0
    if qd.static(dim == 1):
        result = q - (q * q) / (2.0 * h)
    elif qd.static(dim == 2):
        result = 0.5 * q * q - (q * q * q) / (3.0 * h)
    elif qd.static(dim == 3):
        result = (1.0 / 3.0) * q * q * q - (q * q * q * q) / (4.0 * h)
    return result


@qd.func
def tsc_evaluate_integral(q1_in: qd.f32, q2_in: qd.f32, dim: qd.template()) -> qd.f32:
    result = 0.0
    if not (q2_in <= 0.0 or q1_in >= TSC_SUPPORT):
        q1 = qd.max(0.0, q1_in)
        q2 = qd.min(TSC_SUPPORT, q2_in)
        result = tsc_F(q2, dim) - tsc_F(q1, dim)
    return result


@qd.func
def tsc_gradient(q: qd.f32, grad_q, dim: qd.template()):
    q = qd.abs(q)
    dWdq = 0.0
    if q <= TSC_SUPPORT:
        dWdq = -(1.0 / TSC_SUPPORT)
    return dWdq * grad_q


# --- Gaussian ------------------------------------------------------------
GAUSSIAN_SUPPORT = SPHERICAL["gaussian"].support


@qd.func
def gaussian_evaluate(q: qd.f32, dim: qd.template()) -> qd.f32:
    result = 0.0
    if q <= GAUSSIAN_SUPPORT:
        result = qd.exp(-q * q)
    return result


gaussian_sigma = _make_sigma(SPHERICAL["gaussian"].sigma)


@qd.func
def gaussian_F(q_in: qd.f32, dim: qd.template()) -> qd.f32:
    q = qd.max(q_in, 0.0)
    q = qd.min(q, GAUSSIAN_SUPPORT)
    result = 0.0
    if qd.static(dim == 1):
        result = 0.5 * qd.sqrt(pi) * _erf(q)
    elif qd.static(dim == 2):
        result = -0.5 * qd.exp(-q * q)
    elif qd.static(dim == 3):
        result = 0.25 * (qd.sqrt(pi) * _erf(q) - 2.0 * q * qd.exp(-q * q))
    return result


@qd.func
def gaussian_evaluate_integral(q1: qd.f32, q2: qd.f32, dim: qd.template()) -> qd.f32:
    return _spherical_evaluate_integral_default(
        gaussian_F, GAUSSIAN_SUPPORT, dim, q1, q2
    )


@qd.func
def gaussian_gradient(q: qd.f32, grad_q, dim: qd.template()):
    dWdq = 0.0
    if q <= GAUSSIAN_SUPPORT:
        dWdq = -2.0 * q * qd.exp(-(q * q))
    return dWdq * grad_q


# --- Lucy --------------------------------------------------------------
LUCY_SUPPORT = SPHERICAL["lucy"].support


@qd.func
def lucy_evaluate(q: qd.f32, dim: qd.template()) -> qd.f32:
    result = 0.0
    if q <= LUCY_SUPPORT:
        result = (1.0 + 3.0 * q) * (1.0 - q) ** 3
    return result


lucy_sigma = _make_sigma(SPHERICAL["lucy"].sigma)


@qd.func
def lucy_F(q_in: qd.f32, dim: qd.template()) -> qd.f32:
    q = qd.max(q_in, 0.0)
    q = qd.min(q, LUCY_SUPPORT)
    result = 0.0
    if qd.static(dim == 1):
        result = q - 2.0 * q**3 + 2.0 * q**4 - 0.6 * q**5
    elif qd.static(dim == 2):
        result = 0.5 * q**2 - 1.5 * q**4 + 1.6 * q**5 - 0.5 * q**6
    elif qd.static(dim == 3):
        result = (
            (1.0 / 3.0) * q**3
            - (6.0 / 5.0) * q**5
            + (4.0 / 3.0) * q**6
            - (3.0 / 7.0) * q**7
        )
    return result


@qd.func
def lucy_evaluate_integral(q1: qd.f32, q2: qd.f32, dim: qd.template()) -> qd.f32:
    return _spherical_evaluate_integral_default(lucy_F, LUCY_SUPPORT, dim, q1, q2)


@qd.func
def lucy_gradient(q: qd.f32, grad_q, dim: qd.template()):
    dWdq = 0.0
    if q <= LUCY_SUPPORT:
        dWdq = -12.0 * q * (1.0 - q) ** 2
    return dWdq * grad_q


# --- CubicSpline -----------------------------------------------------------
CUBIC_SPLINE_SUPPORT = SPHERICAL["cubic_spline"].support
CUBIC_SPLINE_NODE_1 = 0.5
CUBIC_SPLINE_EPS = 1e-6


@qd.func
def cubic_spline_evaluate(q: qd.f32, dim: qd.template()) -> qd.f32:
    result = 0.0
    if q <= CUBIC_SPLINE_NODE_1:
        result = 1.0 - 6.0 * q * q + 6.0 * q * q * q
    elif q <= CUBIC_SPLINE_SUPPORT:
        r = 1.0 - q
        result = 2.0 * r * r * r
    return result


cubic_spline_sigma = _make_sigma(SPHERICAL["cubic_spline"].sigma)


@qd.func
def cubic_spline_F(q_in: qd.f32, dim: qd.template()) -> qd.f32:
    q = qd.max(q_in, 0.0)
    q = qd.min(q, CUBIC_SPLINE_SUPPORT)
    result = 0.0
    if qd.static(dim == 1):
        if q <= CUBIC_SPLINE_NODE_1:
            result = q - 2.0 * q**3 + 1.5 * q**4
        else:
            result = 0.375 - 0.5 * (1.0 - q) ** 4
    elif qd.static(dim == 2):
        if q <= CUBIC_SPLINE_NODE_1:
            result = 0.5 * q**2 - 1.5 * q**4 + 1.2 * q**5
        else:
            result = 0.0875 - 0.5 * (1.0 - q) ** 4 + 0.4 * (1.0 - q) ** 5
    elif qd.static(dim == 3):
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


@qd.func
def cubic_spline_evaluate_integral(
    q1_in: qd.f32, q2_in: qd.f32, dim: qd.template()
) -> qd.f32:
    q2 = qd.min(q2_in, CUBIC_SPLINE_SUPPORT)
    q1 = qd.max(q1_in, 0.0)
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


@qd.func
def cubic_spline_gradient(q: qd.f32, grad_q, dim: qd.template()):
    dWdq = 0.0
    if q <= CUBIC_SPLINE_NODE_1:
        dWdq = -6.0 * q * (2.0 - 3.0 * q)
    elif q <= CUBIC_SPLINE_SUPPORT:
        dWdq = -6.0 * (1.0 - q) ** 2
    return dWdq * grad_q


# --- QuinticSpline ---------------------------------------------------------
QUINTIC_SPLINE_SUPPORT = SPHERICAL["quintic_spline"].support
QUINTIC_SPLINE_NODE_1 = 1.0
QUINTIC_SPLINE_NODE_2 = 2.0
QUINTIC_SPLINE_EPS = 1e-6


@qd.func
def quintic_spline_evaluate(q: qd.f32, dim: qd.template()) -> qd.f32:
    result = 0.0
    if q <= QUINTIC_SPLINE_SUPPORT:
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


quintic_spline_sigma = _make_sigma(SPHERICAL["quintic_spline"].sigma)


@qd.func
def quintic_spline_F(q_in: qd.f32, dim: qd.template()) -> qd.f32:
    q = qd.max(q_in, 0.0)
    q = qd.min(q, QUINTIC_SPLINE_SUPPORT)
    result = 0.0

    if qd.static(dim == 1):
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

    elif qd.static(dim == 2):
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

    elif qd.static(dim == 3):
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


@qd.func
def quintic_spline_evaluate_integral(
    q1_in: qd.f32, q2_in: qd.f32, dim: qd.template()
) -> qd.f32:
    q2 = qd.min(q2_in, QUINTIC_SPLINE_SUPPORT)
    q1 = qd.max(q1_in, 0.0)
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


@qd.func
def quintic_spline_gradient(q: qd.f32, grad_q, dim: qd.template()):
    dWdq = 0.0
    if q <= QUINTIC_SPLINE_NODE_1:
        dWdq = -5.0 * (3.0 - q) ** 4 + 30.0 * (2.0 - q) ** 4 - 75.0 * (1.0 - q) ** 4
    elif q <= QUINTIC_SPLINE_NODE_2:
        dWdq = -5.0 * (3.0 - q) ** 4 + 30.0 * (2.0 - q) ** 4
    elif q <= QUINTIC_SPLINE_SUPPORT:
        dWdq = -5.0 * (3.0 - q) ** 4
    return dWdq * grad_q


# --- WendlandC2 ------------------------------------------------------------
WENDLAND_C2_SUPPORT = SPHERICAL["wendland_c2"].support


@qd.func
def wendland_c2_evaluate(q: qd.f32, dim: qd.template()) -> qd.f32:
    result = 0.0
    if q <= WENDLAND_C2_SUPPORT:
        z = 1.0 - 0.5 * q
        if qd.static(dim == 1):
            result = z**3 * (1.5 * q + 1.0)
        else:
            result = z**4 * (2.0 * q + 1.0)
    return result


wendland_c2_sigma = _make_sigma(SPHERICAL["wendland_c2"].sigma)


@qd.func
def wendland_c2_F(q_in: qd.f32, dim: qd.template()) -> qd.f32:
    q = qd.max(q_in, 0.0)
    q = qd.min(q, WENDLAND_C2_SUPPORT)
    result = 0.0
    if qd.static(dim == 1):
        result = q - 0.5 * q**3 + 0.25 * q**4 - (3.0 / 80.0) * q**5
    elif qd.static(dim == 2):
        result = (q**2 / 16.0) * (
            8.0 - 10.0 * q**2 + 8.0 * q**3 - 2.5 * q**4 + (2.0 / 7.0) * q**5
        )
    elif qd.static(dim == 3):
        result = (q**3 / 16.0) * (
            16.0 / 3.0
            - 8.0 * q**2
            + (20.0 / 3.0) * q**3
            - (15.0 / 7.0) * q**4
            + 0.25 * q**5
        )
    return result


@qd.func
def wendland_c2_evaluate_integral(q1: qd.f32, q2: qd.f32, dim: qd.template()) -> qd.f32:
    return _spherical_evaluate_integral_default(
        wendland_c2_F, WENDLAND_C2_SUPPORT, dim, q1, q2
    )


@qd.func
def wendland_c2_gradient(q: qd.f32, grad_q, dim: qd.template()):
    dWdq = 0.0
    if q <= WENDLAND_C2_SUPPORT:
        z = 1.0 - 0.5 * q
        if qd.static(dim == 1):
            dWdq = -1.5 * z * z * (1.5 * q + 1.0) + 1.5 * z**3
        else:
            dWdq = -2.0 * z**3 * (2.0 * q + 1.0) + 2.0 * z**4
    return dWdq * grad_q


# --- WendlandC4 ------------------------------------------------------------
WENDLAND_C4_SUPPORT = SPHERICAL["wendland_c4"].support


@qd.func
def wendland_c4_evaluate(q: qd.f32, dim: qd.template()) -> qd.f32:
    result = 0.0
    if q <= WENDLAND_C4_SUPPORT:
        z = 1.0 - 0.5 * q
        if qd.static(dim == 1):
            result = z**5 * (2.0 * q * q + 2.5 * q + 1.0)
        else:
            result = z**6 * ((35.0 / 12.0) * q * q + 3.0 * q + 1.0)
    return result


wendland_c4_sigma = _make_sigma(SPHERICAL["wendland_c4"].sigma)


@qd.func
def wendland_c4_F(q_in: qd.f32, dim: qd.template()) -> qd.f32:
    q = qd.max(q_in, 0.0)
    q = qd.min(q, WENDLAND_C4_SUPPORT)
    result = 0.0
    if qd.static(dim == 1):
        result = (q / 64.0) * (
            64.0
            - (112.0 / 3.0) * q**2
            + 28.0 * q**4
            - (56.0 / 3.0) * q**5
            + 5.0 * q**6
            - 0.5 * q**7
        )
    elif qd.static(dim == 2):
        result = (q**2 / 768.0) * (
            384.0
            - 448.0 * q**2
            + 560.0 * q**4
            - 512.0 * q**5
            + 210.0 * q**6
            - (128.0 / 3.0) * q**7
            + 3.5 * q**8
        )
    elif qd.static(dim == 3):
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


@qd.func
def wendland_c4_evaluate_integral(q1: qd.f32, q2: qd.f32, dim: qd.template()) -> qd.f32:
    return _spherical_evaluate_integral_default(
        wendland_c4_F, WENDLAND_C4_SUPPORT, dim, q1, q2
    )


@qd.func
def wendland_c4_gradient(q: qd.f32, grad_q, dim: qd.template()):
    dWdq = 0.0
    if q <= WENDLAND_C4_SUPPORT:
        z = 1.0 - 0.5 * q
        f, df, g, dg = 0.0, 0.0, 0.0, 0.0
        if qd.static(dim == 1):
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
WENDLAND_C6_SUPPORT = SPHERICAL["wendland_c6"].support


@qd.func
def wendland_c6_evaluate(q: qd.f32, dim: qd.template()) -> qd.f32:
    result = 0.0
    if q <= WENDLAND_C6_SUPPORT:
        z = 1.0 - 0.5 * q
        if qd.static(dim == 1):
            result = z**7 * ((21.0 / 8.0) * q**3 + (19.0 / 4.0) * q**2 + 3.5 * q + 1.0)
        else:
            result = z**8 * (4.0 * q**3 + 6.25 * q**2 + 4.0 * q + 1.0)
    return result


wendland_c6_sigma = _make_sigma(SPHERICAL["wendland_c6"].sigma)


@qd.func
def wendland_c6_F(q_in: qd.f32, dim: qd.template()) -> qd.f32:
    q = qd.max(q_in, 0.0)
    q = qd.min(q, WENDLAND_C6_SUPPORT)
    result = 0.0
    if qd.static(dim == 1):
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
    elif qd.static(dim == 2):
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
    elif qd.static(dim == 3):
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


@qd.func
def wendland_c6_evaluate_integral(q1: qd.f32, q2: qd.f32, dim: qd.template()) -> qd.f32:
    return _spherical_evaluate_integral_default(
        wendland_c6_F, WENDLAND_C6_SUPPORT, dim, q1, q2
    )


@qd.func
def wendland_c6_gradient(q: qd.f32, grad_q, dim: qd.template()):
    dWdq = 0.0
    if q <= WENDLAND_C6_SUPPORT:
        z = 1.0 - 0.5 * q
        f, df, g, dg = 0.0, 0.0, 0.0, 0.0
        if qd.static(dim == 1):
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

    evaluate_1d: Callable  # qd.func(q, dim) -> f32
    F_1d: Callable  # qd.func(q, dim) -> f32
    sigma: Callable  # qd.func(dim) -> f32
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

    evaluate: Callable  # qd.func(q, dim) -> f32
    F: Callable  # qd.func(q, dim) -> f32
    sigma: Callable  # qd.func(dim) -> f32
    evaluate_integral: Callable  # qd.func(q1, q2, dim) -> f32
    gradient: Callable  # qd.func(q, grad_q, dim) -> qd.Vector(dim)  [dW/dq * grad_q]
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
# One qd.kernel per dimension, mirroring the dim==1/2/3 branches in the C++.
# `evaluate_integral_fn` and `sigma_fn` are passed as qd.template() so the
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
# don't need Quadrants scope.
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
# Quadrants kernel: radial integral, computed ONCE per shell (n_q evaluations
# total, shared across every direction/angle sampled within that shell).
# --------------------------------------------------------------------------


@qd.kernel
def _compute_radial_integrals(
    evaluate_integral_fn: qd.template(),
    dim: qd.template(),
    n_q: qd.i32,
    shell_q0: qd.types.ndarray(dtype=qd.f32, ndim=1),
    shell_q1: qd.types.ndarray(dtype=qd.f32, ndim=1),
    radial_integral_out: qd.types.ndarray(dtype=qd.f32, ndim=1),
):
    for iq in range(n_q):
        radial_integral_out[iq] = evaluate_integral_fn(shell_q0[iq], shell_q1[iq], dim)


# --------------------------------------------------------------------------
# Quadrants kernel: sample coordinates and weights from the shell layout and
# the radial integrals. `layout` is the per-sample sign (1D), angle (2D) or
# unit direction (3D); every sample of a shell carries an equal share of the
# shell's mass sigma * (angular measure) * radial integral.
# --------------------------------------------------------------------------
_ANGULAR_MEASURE = {1: 2.0, 2: 2.0 * pi, 3: 4.0 * pi}


@qd.kernel
def _build_kernel_sample_grid_kernel(
    sigma_fn: qd.template(),
    dim: qd.template(),
    n_pts: qd.i32,
    shell_id_in: qd.types.ndarray(dtype=qd.i32, ndim=1),
    shell_q0: qd.types.ndarray(dtype=qd.f32, ndim=1),
    shell_q1: qd.types.ndarray(dtype=qd.f32, ndim=1),
    radial_integral_in: qd.types.ndarray(dtype=qd.f32, ndim=1),
    n_points_per_shell_in: qd.types.ndarray(dtype=qd.i32, ndim=1),
    layout: qd.types.ndarray(dtype=qd.f32),
    coords_out: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (n_pts, dim)
    integrals_out: qd.types.ndarray(dtype=qd.f32, ndim=1),
):
    sig = sigma_fn(dim)
    measure = qd.static(_ANGULAR_MEASURE[dim])
    for i in range(n_pts):
        iq = shell_id_in[i]
        q = 0.5 * (shell_q0[iq] + shell_q1[iq])
        if qd.static(dim == 1):
            coords_out[i, 0] = layout[i] * q
        elif qd.static(dim == 2):
            coords_out[i, 0] = q * qd.cos(layout[i])
            coords_out[i, 1] = q * qd.sin(layout[i])
        else:
            for a in qd.static(range(3)):
                coords_out[i, a] = q * layout[i, a]
        integrals_out[i] = (
            sig * measure * radial_integral_in[iq] / qd.f32(n_points_per_shell_in[iq])
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
    """Build the pre-computed kernel sample table used for unresolved kernels.

    The table is Nyquist-safe for every particle with ``eta < eta_crit``.

    Parameters
    ----------
    kernel_name : str
        Name of the (spherical) kernel to sample.
    dim : int
        Dimension of the kernel (1, 2, or 3).
    eta_crit : float
        Critical kernel diameter in units of the cell size, below which the
        sample table is used instead of numerical quadrature.
    nyquist_factor : float, default 2.0
        Samples per cell width; 2.0 is the Nyquist minimum.

    Returns
    -------
    dict
        Read-only, cached table with keys ``"dim"``, ``"count"`` (number of
        samples), ``"n_q"`` (number of radial shells), ``"coords"`` (sample
        coordinates in units of ``h``, shape (count, dim)) and ``"integrals"``
        (integral weight of each sample, summing to unity).
    """
    return dict(
        _build_kernel_sample_grid_cached(
            kernel_name, int(dim), float(eta_crit), float(nyquist_factor)
        )
    )


@lru_cache(maxsize=64)
def _build_kernel_sample_grid_cached(
    kernel_name: str,
    dim: int,
    eta_crit: float,
    nyquist_factor: float,
):
    if eta_crit < 0:
        raise ValueError("eta_crit must be >= 0")
    if dim not in (1, 2, 3):
        raise ValueError("build_kernel_sample_grid supports only dim = 1, 2 or 3")

    kspec = create_spherical_kernel(kernel_name)
    layout_fn = {1: _build_shell_layout_1d, 2: _build_shell_layout_2d,
                 3: _build_shell_layout_3d}[dim]
    n_q, shell_q0, shell_q1, shell_id, layout, n_points_per_shell = layout_fn(
        kspec.support, eta_crit, nyquist_factor
    )
    count = shell_id.shape[0]
    radial_integral = np.zeros(n_q, dtype=np.float32)
    _compute_radial_integrals(
        kspec.evaluate_integral, dim, n_q, shell_q0, shell_q1, radial_integral
    )

    coords = np.zeros((count, dim), dtype=np.float32)
    integrals = np.zeros(count, dtype=np.float32)
    _build_kernel_sample_grid_kernel(
        kspec.sigma,
        dim,
        count,
        shell_id,
        shell_q0,
        shell_q1,
        radial_integral,
        n_points_per_shell,
        layout,
        coords,
        integrals,
    )

    for arr in (coords, integrals):
        arr.flags.writeable = False
    return {
        "dim": dim,
        "count": count,
        "n_q": n_q,
        "coords": coords,
        "integrals": integrals,
    }


# =============================================================================
# Utility functions (debugging / testing parity with the C++ helpers)
# =============================================================================
@qd.kernel
def _separable_axis_integral(
    F_1d_fn: qd.template(), support: qd.f32, dim: qd.template()
) -> qd.f32:
    return _separable_integrate_1d(F_1d_fn, support, dim, -support, support)


@qd.kernel
def _sigma_value(sigma_fn: qd.template(), dim: qd.template()) -> qd.f32:
    return sigma_fn(dim)


def compute_total_integral_separable(kernel_name: str, dim: int) -> float:
    """Integrate a separable kernel over its support (should be unity).

    Parameters
    ----------
    kernel_name : str
        Name of the separable kernel.
    dim : int
        Dimension of the kernel (1, 2, or 3).

    Returns
    -------
    float
        Total integral of the kernel.
    """
    from . import ensure_initialized

    ensure_initialized()
    kspec = create_separable_kernel(kernel_name)
    single_axis_integral = _separable_axis_integral(kspec.F_1d, kspec.support, dim)
    sigma = _sigma_value(kspec.sigma, dim)
    return float(sigma * (single_axis_integral**dim))


def compute_total_integral_spherical(
    kernel_name: str, dim: int, eta_crit: float
) -> float:
    """Integrate a spherical kernel via its sample table (should be unity).

    Parameters
    ----------
    kernel_name : str
        Name of the spherical kernel.
    dim : int
        Dimension of the kernel (1, 2, or 3).
    eta_crit : float
        Critical eta used to size the sample table, see
        :func:`build_kernel_sample_grid`.

    Returns
    -------
    float
        Total integral of the kernel.
    """
    from . import ensure_initialized

    ensure_initialized()
    grid = build_kernel_sample_grid(kernel_name, dim, eta_crit)
    return float(np.sum(grid["integrals"]))


@qd.kernel
def _sample_profile_1d(
    evaluate_fn: qd.template(),
    sigma_fn: qd.template(),
    support: qd.f32,
    dim: qd.template(),
    num_samples: qd.i32,
    q_out: qd.types.ndarray(dtype=qd.f32, ndim=1),
    values_out: qd.types.ndarray(dtype=qd.f32, ndim=1),
):
    """sigma * W(|q|) on `num_samples` points spanning [-support, support)."""
    dq = 2.0 * support / num_samples
    sig = sigma_fn(dim)
    for i in range(num_samples):
        q_current = -support + i * dq
        q_out[i] = q_current
        values_out[i] = sig * evaluate_fn(qd.abs(q_current), dim)


def get_spherical_kernel_values_1D(kernel_name: str):
    """Sample the 1D profile of a spherical (radial) kernel.

    Parameters
    ----------
    kernel_name : str
        Name of the spherical kernel.

    Returns
    -------
    q : np.ndarray
        Sample points in units of the smoothing length.
    values : np.ndarray
        Kernel values at the sample points.
    """
    from . import ensure_initialized

    ensure_initialized()
    kspec = create_spherical_kernel(kernel_name)
    num_samples = 100
    q = np.zeros(num_samples, dtype=np.float32)
    values = np.zeros(num_samples, dtype=np.float32)
    _sample_profile_1d(
        kspec.evaluate, kspec.sigma, kspec.support, 1, num_samples, q, values
    )
    return q, values


def get_separable_kernel_values_1D(kernel_name: str):
    """Sample the 1D profile of a separable kernel.

    Parameters
    ----------
    kernel_name : str
        Name of the separable kernel.

    Returns
    -------
    q : np.ndarray
        Sample points in units of the smoothing length.
    values : np.ndarray
        Kernel values at the sample points.
    """
    from . import ensure_initialized

    ensure_initialized()
    kspec = create_separable_kernel(kernel_name)
    num_samples = 100
    q = np.zeros(num_samples, dtype=np.float32)
    values = np.zeros(num_samples, dtype=np.float32)
    _sample_profile_1d(
        kspec.evaluate_1d, kspec.sigma, kspec.support, 1, num_samples, q, values
    )
    return q, values
