"""Plane projection must equal the 3D deposit summed along the projection axis.

Integrating the 3D kernel W(x) = sigma_3 w(|H^-1 x|) / det H along the line of
sight gives, in the plane,

    Sigma(y) = sigma_3 F(|H_2^-1 y|) / det H_2,
    H_2 = (P H H^T P^T)^(1/2)                       (shadow of the ellipsoid)
    F(rho) = 2 int_0^sqrt(R^2 - rho^2) w(sqrt(rho^2 + s^2)) ds   (projected profile)

`_reference` evaluates this in float64, independently of smudgy's deposition
code, and integrates it over each pixel. Isotropic and covariant plane
projection do not do this yet (they use the 2D kernel and, for covariant,
H_2 = (P H^-1 P^T)^-1): those cases are marked xfail(strict) until the fix
(plan Phase P). The Gaussian is its own projection, so isotropic Gaussian
already passes; separable and fixed-grid kernels are exact already.

Tolerances are provisional (max abs error / max of the reference image); they
are set from measurement once the fix is in.
"""

import numpy as np
import pytest
from scipy.linalg import sqrtm

from smudgy import PointCloud
from smudgy.backend.numpy.kernels import get_kernel

SPHERICAL_KERNELS = [
    "tophat",
    "tsc",
    "gaussian",
    "lucy",
    "cubic_spline",
    "quintic_spline",
    "wendland_c2",
    "wendland_c4",
    "wendland_c6",
]
SEPARABLE_KERNELS = ["tophat", "tsc", "gaussian"]
FIXED_KERNELS = ["ngp", "cic", "tsc", "bspline4"]
# (2, 0) checks the output axis order follows `plane_projection`
PLANES = [(0, 1), (2, 0), (1, 2)]
GRID_3D = (40, 48, 56)  # non-cubic: catches axis mix-ups
N = 10
R_MIN, R_MAX = 0.10, 0.20  # physical support radius (largest axis); box = 1

# simpson, resolved particles; tophat has an infinite-slope edge in projection
TOL_REFERENCE = {"tophat": 1e-1}
TOL_REFERENCE_DEFAULT = 5e-3
# 3D deposit summed along the axis vs reference: measured <= 7e-3 (gaussian
# covariant), limited by the 3D cell quadrature; today's isotropic projection
# is off by 2.5-5e-2 (cubic spline, Lucy), covariant by 0.1-0.2
TOL_COLUMN = 1e-2
TOL_REFERENCE_GUARD = 1.5e-2


def _particles(kernel_name, seed=0):
    """Positions, weights, smoothing lengths and tensors (support radius in [R_MIN, R_MAX])."""
    rng = np.random.default_rng(seed)
    support = get_kernel(kernel_name if kernel_name not in FIXED_KERNELS else "cubic_spline", 3).support
    pos = rng.uniform(0, 1, (N, 3))
    weights = rng.uniform(0.5, 1.5, N)
    h = rng.uniform(R_MIN, R_MAX, N) / support
    Q, _ = np.linalg.qr(rng.normal(size=(N, 3, 3)))
    axes = h[:, None] * np.stack(
        [np.ones(N), rng.uniform(0.5, 0.8, N), rng.uniform(0.35, 0.5, N)], axis=1
    )
    H = np.einsum("nij,nj,nkj->nik", Q, axes, Q)
    return pos, weights, h, H


def _cloud(kernel_name, structure):
    pos, weights, h, H = _particles(kernel_name)
    pc = PointCloud(pos, weights=weights, boxsize=1.0, verbose=False)
    if structure == "covariant":
        pc.set_smoothing("covariant", smoothing_tensors=H)
    elif structure in ("isotropic", "separable"):
        pc.set_smoothing(structure, smoothing_lengths=h)
    return pc


def _projected_profile(kernel_name, n_rho=4001, n_s=2000):
    """sigma_3 * F(rho) on a fine grid, Gauss-Legendre along the line of sight."""
    kernel = get_kernel(kernel_name, 3)
    R = kernel.support
    rho = np.linspace(0.0, R, n_rho)
    x, wq = np.polynomial.legendre.leggauss(n_s)
    s_max = np.sqrt(np.maximum(R * R - rho * rho, 0.0))[:, None]
    s = 0.5 * (x + 1.0) * s_max
    ws = 0.5 * wq * s_max
    F = 2.0 * (ws * kernel._kernel_values(np.sqrt(rho[:, None] ** 2 + s**2))).sum(axis=1)
    return rho, kernel.sigma * F


def _reference(kernel_name, structure, plane, gridnums, n_sub=8):
    """Exact projected image, each particle's pixel integrals normalised to its weight.

    deposit() renormalises every particle to its weight as well, so the
    comparison tests the shape, not the kernel's truncated normalisation.
    """
    pos, weights, h, H = _particles(kernel_name)
    rho, F = _projected_profile(kernel_name)
    x, wq = np.polynomial.legendre.leggauss(n_sub)
    x, wq = 0.5 * (x + 1.0), 0.5 * wq
    centers = [
        (np.arange(n)[:, None] + x[None, :]) / n for n in gridnums
    ]  # (n_cells, n_sub) sub-point coordinates per axis
    image = np.zeros(gridnums)
    for i in range(N):
        y0 = pos[i, list(plane)]
        if structure == "covariant":
            H2 = sqrtm((H[i] @ H[i].T)[np.ix_(plane, plane)]).real
        else:
            H2 = h[i] * np.eye(2)
        Hi = np.linalg.inv(H2)
        dx = (centers[0] - y0[0] + 0.5) % 1.0 - 0.5  # minimum image
        dy = (centers[1] - y0[1] + 0.5) % 1.0 - 0.5
        DX, DY = dx[:, None, :, None], dy[None, :, None, :]
        u = Hi[0, 0] * DX + Hi[0, 1] * DY
        v = Hi[1, 0] * DX + Hi[1, 1] * DY
        val = np.interp(np.sqrt(u * u + v * v), rho, F, right=0.0)
        cell = (val * wq[None, None, :, None] * wq[None, None, None, :]).sum(axis=(2, 3))
        image += weights[i] * cell / cell.sum()
    return image


def _column(grid3d, plane):
    """Sum a 3D grid along the axis not in `plane`, axes ordered as `plane`."""
    axis = ({0, 1, 2} - set(plane)).pop()
    col = grid3d.sum(axis=axis)
    return col.T if plane[0] > plane[1] else col


def _rel_err(a, ref):
    return np.abs(a - ref).max() / np.abs(ref).max()


def _xfail_if_unfixed(request, structure, kernel_name):
    gaussian_isotropic = structure == "isotropic" and kernel_name == "gaussian"
    if structure in ("isotropic", "covariant") and not gaussian_isotropic:
        request.applymarker(pytest.mark.xfail(
            strict=True,
            reason="plane projection uses the 2D kernel (and the wrong covariant "
                   "H_2), not the projected 3D kernel (plan Phase P)",
        ))


@pytest.mark.parametrize("plane", PLANES)
@pytest.mark.parametrize("structure", ["isotropic", "covariant"])
@pytest.mark.parametrize("kernel_name", SPHERICAL_KERNELS)
def test_projection_matches_reference(request, kernel_name, structure, plane):
    """2D projected deposit == exact column density integrated over each pixel."""
    _xfail_if_unfixed(request, structure, kernel_name)
    gridnums = [GRID_3D[a] for a in plane]
    image = _cloud(kernel_name, structure).deposit(
        None, averaged=False, gridnums=gridnums, plane_projection=list(plane),
        kernel_name=kernel_name, structure=structure,
        integration_method="simpson", eta_crit=0.0,
    )
    ref = _reference(kernel_name, structure, plane, gridnums)
    tol = TOL_REFERENCE.get(kernel_name, TOL_REFERENCE_DEFAULT)
    assert _rel_err(image, ref) < tol


@pytest.mark.parametrize("structure", ["isotropic", "covariant"])
@pytest.mark.parametrize("kernel_name", ["gaussian", "cubic_spline", "wendland_c2"])
def test_reference_matches_3d_column(kernel_name, structure):
    """Guards the reference itself: the 3D deposit summed along z agrees with it."""
    plane = (2, 0)
    grid3d = _cloud(kernel_name, structure).deposit(
        None, averaged=False, gridnums=GRID_3D, kernel_name=kernel_name,
        structure=structure, integration_method="simpson", eta_crit=0.0,
    )
    ref = _reference(kernel_name, structure, plane, [GRID_3D[a] for a in plane])
    assert _rel_err(_column(grid3d, plane), ref) < TOL_REFERENCE_GUARD


def _column_and_projection(kernel_name, structure, plane, **kw):
    pc = _cloud(kernel_name, structure)
    kw = dict(averaged=False, kernel_name=kernel_name, **kw)
    if structure is None:
        kw["adaptive"] = False
    else:
        kw["structure"] = structure
    grid3d = pc.deposit(None, gridnums=GRID_3D, **kw)
    image = pc.deposit(
        None, gridnums=[GRID_3D[a] for a in plane], plane_projection=list(plane), **kw
    )
    return _column(grid3d, plane), image


@pytest.mark.parametrize("plane", PLANES)
@pytest.mark.parametrize("structure", ["isotropic", "covariant"])
@pytest.mark.parametrize("kernel_name", ["cubic_spline", "lucy"])
def test_projection_matches_3d_column(request, kernel_name, structure, plane):
    """2D projected deposit == 3D deposit summed along the projection axis."""
    _xfail_if_unfixed(request, structure, kernel_name)
    col, image = _column_and_projection(
        kernel_name, structure, plane, integration_method="simpson", eta_crit=0.0
    )
    assert _rel_err(image, col) < TOL_COLUMN


@pytest.mark.parametrize("plane", PLANES)
@pytest.mark.parametrize("kernel_name", SEPARABLE_KERNELS)
def test_separable_projection_is_exact(kernel_name, plane):
    """Product kernels: the z factor integrates to one, so projection is exact."""
    col, image = _column_and_projection(kernel_name, "separable", plane)
    assert _rel_err(image, col) < 1e-4  # measured 2e-5 (gaussian)


@pytest.mark.parametrize("plane", PLANES)
@pytest.mark.parametrize("kernel_name", FIXED_KERNELS)
def test_fixed_grid_projection_is_exact(kernel_name, plane):
    """Point masses: projecting before or after assignment is the same."""
    col, image = _column_and_projection(kernel_name, None, plane)
    assert _rel_err(image, col) < 1e-5  # measured 1e-7 (cic)


@pytest.mark.parametrize("eta_crit", [0.0, 30.0])  # quadrature only / sample table (all particles)
@pytest.mark.parametrize("structure", ["isotropic", "covariant"])
@pytest.mark.parametrize("kernel_name", ["cubic_spline", "wendland_c2"])
def test_projection_conserves_mass(kernel_name, structure, eta_crit):
    """Projected mass equals the total weight, quadrature and sample-table paths.

    Mass only: the sample table is too coarse for a pixel-wise check (5e-2
    off the reference even for the isotropic Gaussian, which is exact).
    """
    _, weights, _, _ = _particles(kernel_name)
    image = _cloud(kernel_name, structure).deposit(
        None, averaged=False, gridnums=(40, 56), plane_projection=[2, 0],
        kernel_name=kernel_name, structure=structure, eta_crit=eta_crit,
    )
    np.testing.assert_allclose(image.sum(), weights.sum(), rtol=1e-5)
