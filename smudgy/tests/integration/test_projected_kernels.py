"""Projected kernel profiles: float64 closed form, mass-exact table, device lookup."""

import numpy as np
import pytest
import quadrants as qd
from scipy.integrate import quad

from smudgy.backend.kernel_table import SPHERICAL
from smudgy.backend.numpy.kernels import get_kernel
from smudgy.backend.projected_kernels import (
    BREAKPOINTS,
    N_TABLE,
    PIECES,
    interpolate_table,
    projected_cumulative,
    projected_profile,
    projected_table,
)
from smudgy.backend.quadrants import ensure_initialized
from smudgy.backend.quadrants.kernels import (
    build_kernel_sample_grid,
    create_spherical_kernel,
)

KERNELS = list(SPHERICAL)


def _profile_quad(name, rho):
    """F(rho) by adaptive quadrature of the numpy backend's 3D shape."""
    k = get_kernel(name, 3)
    R = k.support
    s_max = np.sqrt(max(R * R - rho * rho, 0.0))
    pts = [np.sqrt(b * b - rho * rho) for b in BREAKPOINTS[name] if b > rho]
    val, _ = quad(
        lambda s: float(k._kernel_values(np.array(np.sqrt(rho * rho + s * s)))),
        0.0, s_max, points=pts or None, epsabs=1e-14, epsrel=1e-13, limit=200,
    )
    return 2.0 * val


@pytest.mark.parametrize("name", [n for n in KERNELS if n != "gaussian"])
def test_pieces_match_kernel_shapes(name):
    """The polynomial pieces equal the 3D kernel shapes."""
    r = np.linspace(0.0, SPHERICAL[name].support, 1001)[:-1]
    w = np.zeros_like(r)
    for a, b, poly in PIECES[name]:
        w = np.where((r >= a) & (r < b), poly(r), w)
    np.testing.assert_allclose(w, get_kernel(name, 3)._kernel_values(r), atol=1e-12)


@pytest.mark.parametrize("name", KERNELS)
def test_closed_form_matches_quadrature(name):
    """Closed-form F and M (float64) equal adaptive quadrature."""
    R = SPHERICAL[name].support
    rho = np.linspace(0.0, R, 61)[:-1]
    ref = np.array([_profile_quad(name, r) for r in rho])
    F = projected_profile(name, rho)
    assert np.abs(F - ref).max() < 1e-11 * ref[0]  # measured <= 6e-13 (wendland_c6)

    q = rho[1:]
    M_ref = [
        quad(lambda r: float(projected_profile(name, r)) * r, 0.0, qq, limit=200,
             points=[b for b in BREAKPOINTS[name] if b < qq] or None, epsabs=1e-14)[0]
        for qq in q
    ]
    M = projected_cumulative(name, q)
    assert np.abs(M - M_ref).max() < 1e-10 * M[-1]  # measured <= 2e-11


@pytest.mark.parametrize("name", KERNELS)
def test_projection_conserves_3d_mass(name):
    """2 pi M(R) = 4 pi int_0^R w r^2 dr: the projection keeps the 3D mass."""
    R = SPHERICAL[name].support
    k = get_kernel(name, 3)
    mass_3d = quad(
        lambda r: float(k._kernel_values(np.array(r))) * r * r, 0.0, R,
        points=BREAKPOINTS[name] or None, epsabs=1e-14,
    )[0]
    np.testing.assert_allclose(projected_cumulative(name, R), 2.0 * mass_3d, rtol=1e-11)


@pytest.mark.parametrize("name", KERNELS)
def test_table_is_mass_exact(name):
    """The interpolant matches F at the nodes and the exact mass of each interval."""
    R = SPHERICAL[name].support
    F, F_mean, M = projected_table(name)
    x_nodes = np.linspace(0.0, R * R, N_TABLE + 1)
    for b in BREAKPOINTS[name]:  # kinks of F fall on nodes
        assert np.isclose(x_nodes, b * b, rtol=0, atol=1e-12).any()
    np.testing.assert_allclose(
        interpolate_table(name, np.sqrt(x_nodes[:-1])), F[:-1], rtol=0, atol=1e-13 * F[0]
    )
    t, wt = np.polynomial.legendre.leggauss(4)  # exact for the quadratic in t
    t, wt = 0.5 * (t + 1.0), 0.5 * wt
    dx = x_nodes[1] - x_nodes[0]
    x = x_nodes[:-1, None] + dx * t[None, :]
    mean = (interpolate_table(name, np.sqrt(x)) * wt).sum(axis=1)
    np.testing.assert_allclose(mean, F_mean, rtol=0, atol=1e-12 * F[0])
    np.testing.assert_allclose(M[-1], projected_cumulative(name, R), rtol=1e-14)


@pytest.mark.parametrize("name", KERNELS)
def test_table_interpolation_error(name):
    """Within 5e-5 of the peak, but in the first and last interval.

    Measured (max / peak): <= 1e-7 for the smooth kernels; TSC 1.7e-6 (its
    x log x term at the centre), tophat 3.6e-5 (infinite slope at the edge).
    The first interval (TSC 4e-5) and the last (tophat 4e-3) hold the
    singular terms; their mass is still exact.
    """
    R = SPHERICAL[name].support
    q = np.random.default_rng(0).uniform(0.0, R, 50_000)
    x = q * q / (R * R)
    keep = (x > 1.0 / N_TABLE) & (x < 1.0 - 1.0 / N_TABLE)
    err = np.abs(interpolate_table(name, q) - projected_profile(name, q))[keep]
    assert err.max() < 5e-5 * projected_profile(name, 0.0)


@qd.kernel
def _eval_projected(
    evaluate_fn: qd.template(),
    F_fn: qd.template(),
    gradient_fn: qd.template(),
    sigma_fn: qd.template(),
    q: qd.types.ndarray(dtype=qd.f32, ndim=1),
    out: qd.types.ndarray(dtype=qd.f32, ndim=2),
):
    for i in range(q.shape[0]):
        out[i, 0] = evaluate_fn(q[i], 2)
        out[i, 1] = F_fn(q[i], 2)
        g = gradient_fn(q[i], qd.Vector([1.0, 0.0]), 2)
        out[i, 2] = g[0]
        out[i, 3] = sigma_fn(2)


@pytest.mark.parametrize("name", KERNELS)
def test_device_lookup(name):
    """Device evaluate / F / gradient / sigma match the host table."""
    ensure_initialized()
    spec = create_spherical_kernel(name, projected=True)
    R = SPHERICAL[name].support
    q = np.random.default_rng(1).uniform(0.0, 1.05 * R, 4000).astype(np.float32)
    out = np.zeros((q.size, 4), dtype=np.float32)
    _eval_projected(spec.evaluate, spec.F, spec.gradient, spec.sigma, q, out)
    q64 = q.astype(np.float64)
    peak = projected_profile(name, 0.0)
    # float32 t = q^2 / dx - i is good to ~1e-4 near the edge; that only
    # shows where F is steep there (tophat, measured 2.5e-6)
    atol = (1e-5 if name == "tophat" else 2e-6) * peak
    np.testing.assert_allclose(out[:, 0], interpolate_table(name, q64), rtol=0, atol=atol)
    total = projected_cumulative(name, R)
    np.testing.assert_allclose(out[:, 1], projected_cumulative(name, q64), rtol=0, atol=2e-6 * total)
    assert np.all(out[:, 3] == np.float32(SPHERICAL[name].sigma[2]))
    # gradient: central differences of the host interpolant, away from nodes
    # (dF/dq jumps there) and the singular end intervals
    u = q64**2 * N_TABLE / (R * R)
    t = u - np.floor(u)
    ok = (t > 0.05) & (t < 0.95) & (u > 1) & (u < N_TABLE - 1)
    eps = 1e-7 * R
    fd = (interpolate_table(name, q64 + eps) - interpolate_table(name, q64 - eps)) / (2 * eps)
    scale = np.abs(fd[ok]).max()
    np.testing.assert_allclose(out[ok, 2], fd[ok], rtol=0, atol=2e-3 * scale)


@pytest.mark.parametrize("eta_crit", [4.0, 30.0])
@pytest.mark.parametrize("name", ["tophat", "cubic_spline", "quintic_spline", "wendland_c6"])
def test_projected_sample_table_sums_to_one(name, eta_crit):
    """The projected sample table carries the projected kernel's mass."""
    ensure_initialized()
    grid = build_kernel_sample_grid(name, 2, eta_crit, projected=True)
    expected = 2 * np.pi * SPHERICAL[name].sigma[2] * projected_cumulative(name, SPHERICAL[name].support)
    np.testing.assert_allclose(grid["integrals"].sum(), expected, rtol=2e-6)
    plain = build_kernel_sample_grid(name, 2, eta_crit)
    assert not np.array_equal(plain["integrals"], grid["integrals"])


def test_projected_requires_2d():
    """Projected kernels are 2D only."""
    with pytest.raises(ValueError, match="2D"):
        build_kernel_sample_grid("cubic_spline", 3, 4.0, projected=True)
