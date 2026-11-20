"""Projected (line-of-sight integrated) spherical kernels, float64 on the host.

Integrating a 3D radial kernel w(r) along a line of sight at impact parameter
rho gives the projected profile

    F(rho) = 2 int_0^sqrt(R^2 - rho^2) w(sqrt(rho^2 + s^2)) ds,

so that sigma_3 F(|y| / h) / h^2 is the column density of the 3D kernel
sigma_3 w(|x| / h) / h^3 and integrates to one in the plane. Its cumulative
mass is

    M(q) = int_0^q F(rho) rho drho = 2 [C - int_{r > q} w(r) (r^2 - q^2) ds],
    C = int_0^R w(r) r^2 dr,

(the second term is the kernel mass outside the cylinder of radius q).

Both are evaluated in closed form, piece by piece of the piecewise
polynomial w(r) = sum_k c_k r^k, from I_k = int r^k ds:

    I_k = (s r^k + k rho^2 I_{k-2}) / (k + 1),  I_0 = s,  I_-1 = ln(s + r).

The recurrence cancels badly in float32 for high-degree kernels (4e-4 of the
peak for Wendland C6), but not in float64 (< 1e-12), so the device does not
evaluate it: it interpolates :func:`projected_table`, a table in x = q^2
built here once per kernel. The table is mass-exact: on every interval the
interpolant is the derivative of the cubic Hermite interpolant of the exact
cumulative mass, so it matches F at the nodes and the exact mass of every
interval (see :func:`projected_table`).
"""

from functools import cache
from math import erf, exp, pi, sqrt

import numpy as np
from numpy.polynomial import Polynomial

from .kernel_table import SPHERICAL

_r = Polynomial([0.0, 1.0])

# w(r) of every spherical kernel except the Gaussian, as (r_lo, r_hi, poly)
# pieces; must equal the 3D shapes (dim >= 2 forms) of the kernel backends.
PIECES = {
    "tophat": [(0.0, 0.5, Polynomial([1.0]))],
    "tsc": [(0.0, 1.5, 1.0 - _r / 1.5)],
    "lucy": [(0.0, 1.0, (1.0 + 3.0 * _r) * (1.0 - _r) ** 3)],
    "cubic_spline": [
        (0.0, 0.5, 1.0 - 6.0 * _r**2 + 6.0 * _r**3),
        (0.5, 1.0, 2.0 * (1.0 - _r) ** 3),
    ],
    "quintic_spline": [
        (0.0, 1.0, (3.0 - _r) ** 5 - 6.0 * (2.0 - _r) ** 5 + 15.0 * (1.0 - _r) ** 5),
        (1.0, 2.0, (3.0 - _r) ** 5 - 6.0 * (2.0 - _r) ** 5),
        (2.0, 3.0, (3.0 - _r) ** 5),
    ],
    "wendland_c2": [(0.0, 2.0, (1.0 - _r / 2.0) ** 4 * (2.0 * _r + 1.0))],
    "wendland_c4": [
        (0.0, 2.0, (1.0 - _r / 2.0) ** 6 * (35.0 / 12.0 * _r**2 + 3.0 * _r + 1.0))
    ],
    "wendland_c6": [
        (
            0.0,
            2.0,
            (1.0 - _r / 2.0) ** 8 * (4.0 * _r**3 + 6.25 * _r**2 + 4.0 * _r + 1.0),
        )
    ],
}

# Interior breakpoints of w (kinks of the profile); table nodes fall on them.
BREAKPOINTS = {
    name: [lo for lo, _, _ in pieces[1:]] for name, pieces in PIECES.items()
}
BREAKPOINTS["gaussian"] = []

# Table intervals in x = q^2. 1152 = 2^7 * 3^2 puts every breakpoint on a
# node (cubic spline x = 1/4 of R^2 = 1; quintic spline x = 1, 4 of R^2 = 9).
N_TABLE = 1152


def _support(name):
    try:
        return SPHERICAL[name].support
    except KeyError:
        raise ValueError(
            f"Unknown spherical kernel: {name!r}. Available: {list(SPHERICAL)}"
        ) from None


def _delta_I(a, b, rho, kmax):
    """{k: int r^k ds} over the part of [a, b] in r beyond rho, k = -1..kmax."""
    r2 = rho * rho
    ra = np.maximum(a, rho)
    rb = np.maximum(b, rho)
    sa = np.sqrt(np.maximum(ra * ra - r2, 0.0))
    sb = np.sqrt(np.maximum(rb * rb - r2, 0.0))
    # ra = 0 only at rho = 0, where I_-1 enters multiplied by rho^2 = 0
    Ik = {-1: np.log((sb + rb) / np.maximum(sa + ra, 1e-300)), 0: sb - sa}
    pa, pb = sa, sb  # s r^k at both ends
    for k in range(1, kmax + 1):
        pa, pb = pa * ra, pb * rb
        Ik[k] = ((pb - pa) + k * r2 * Ik[k - 2]) / (k + 1)
    return Ik


def _gaussian_G(S):
    """int_0^S s^2 exp(-s^2) ds."""
    return 0.25 * sqrt(pi) * erf(S) - 0.5 * S * exp(-S * S)


def projected_profile(name, rho):
    """F(rho), the 3D shape w integrated along the line of sight (unnormalised).

    ``sigma_3 * F(|y| / h) / h**2`` integrates to one in the plane, sigma_3
    being the kernel's 3D normalisation.
    """
    R = _support(name)
    rho = np.asarray(rho, dtype=np.float64)
    if name == "gaussian":
        S = np.sqrt(np.maximum(R * R - rho * rho, 0.0))
        return np.where(
            rho < R, sqrt(pi) * np.exp(-rho * rho) * np.vectorize(erf)(S), 0.0
        )
    F = np.zeros_like(rho)
    for a, b, poly in PIECES[name]:
        c = poly.coef
        Ik = _delta_I(a, b, rho, len(c) - 1)
        F += sum(ck * Ik[k] for k, ck in enumerate(c))
    return 2.0 * F


def projected_cumulative(name, q):
    """M(q) = int_0^q F(rho) rho drho; ``2 pi sigma_3 M(R)`` is one."""
    R = _support(name)
    q = np.minimum(np.asarray(q, dtype=np.float64), R)
    if name == "gaussian":
        S = np.sqrt(np.maximum(R * R - q * q, 0.0))
        G = np.vectorize(_gaussian_G)(S)
        return 2.0 * (_gaussian_G(R) - np.exp(-q * q) * G)
    total = 0.0
    outside = np.zeros_like(q)
    for a, b, poly in PIECES[name]:
        c = poly.coef
        total += (poly * _r**2).integ()(b) - (poly * _r**2).integ()(a)
        Ik = _delta_I(a, b, q, len(c) + 1)
        outside += sum(ck * (Ik[k + 2] - q * q * Ik[k]) for k, ck in enumerate(c))
    return 2.0 * (total - outside)


@cache
def projected_table(name, n=N_TABLE):
    """Mass-exact interpolation table of F in x = q^2, float64.

    Nodes x_i = i R^2 / n. Returns read-only arrays

    - ``F`` (n + 1,): F at the nodes,
    - ``F_mean`` (n,): mean of F over each interval in x, 2 (M_i+1 - M_i) / dx,
    - ``M`` (n + 1,): cumulative mass M at the nodes.

    With t in [0, 1] across interval i, the interpolant

        F~ = 6 t (1 - t) F_mean_i + (1 - t)(1 - 3 t) F_i + t (3 t - 2) F_i+1

    is twice the derivative (in x) of the cubic Hermite interpolant of M(x)
    with slopes dM/dx = F / 2: it equals F at both nodes and has mean F_mean_i,
    so every interval carries its exact mass. Its antiderivative is

        M~ = M_i + dx / 2 [F_mean_i (3t^2 - 2t^3) + F_i (t - 2t^2 + t^3)
                           + F_i+1 (t^3 - t^2)].
    """
    R = _support(name)
    x = np.linspace(0.0, R * R, n + 1)
    q = np.sqrt(x)
    q[-1] = R
    F = projected_profile(name, q)
    M = projected_cumulative(name, q)
    F_mean = 2.0 * np.diff(M) / (x[1] - x[0])
    for arr in (F, F_mean, M):
        arr.flags.writeable = False
    return F, F_mean, M


def interpolate_table(name, q, n=N_TABLE):
    """Evaluate the table interpolant F~(q) on the host (float64)."""
    R = _support(name)
    F, F_mean, _ = projected_table(name, n)
    x = np.asarray(q, dtype=np.float64) ** 2
    u = x * (n / (R * R))
    i = np.clip(u.astype(np.int64), 0, n - 1)
    t = u - i
    val = (
        6.0 * t * (1.0 - t) * F_mean[i]
        + (1.0 - t) * (1.0 - 3.0 * t) * F[i]
        + t * (3.0 * t - 2.0) * F[i + 1]
    )
    return np.where(x < R * R, val, 0.0)
