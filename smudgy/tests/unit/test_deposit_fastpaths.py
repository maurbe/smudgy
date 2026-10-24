"""Regression tests for the deposition fast paths.

Covers `PointCloud.deposit(fields=None)` (mass-only deposition), the
`extent` particle selection, the local-buffer sample branch of the
isotropic/covariant taichi kernels, and 1D covariant == 1D isotropic.
"""

import numpy as np
import pytest

from smudgy import PointCloud
from smudgy.backend.taichi import deposition as tdep

SEED = 7


def _cloud(dim, n=2000, boxsize=1.0, backend="taichi"):
    rng = np.random.default_rng(SEED)
    pos = rng.uniform(0.0, 1.0, size=(n, dim))
    w = rng.uniform(0.5, 1.5, size=n)
    pc = PointCloud(positions=pos, weights=w, boxsize=boxsize, backend=backend, verbose=False)
    return pc, pos, w


@pytest.mark.parametrize("backend", ["numpy", "taichi"])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("kernel_name", ["ngp", "cic", "tsc"])
def test_fields_none_is_weights_grid(backend, dim, kernel_name):
    pc, _, w = _cloud(dim, backend=backend)
    kw = dict(averaged=False, gridnums=16, adaptive=False, kernel_name=kernel_name)
    mass = pc.deposit(fields=None, **kw)
    _, weights = pc.deposit(fields=w, return_weights=True, **kw)
    assert mass.shape == (16,) * dim
    np.testing.assert_allclose(mass, weights, rtol=1e-5, atol=1e-6)
    assert mass.sum() == pytest.approx(w.sum(), rel=1e-5)


def test_fields_none_rejects_averaged():
    pc, _, _ = _cloud(2)
    with pytest.raises(ValueError):
        pc.deposit(fields=None, averaged=True, gridnums=8, adaptive=False, kernel_name="cic")


@pytest.mark.parametrize("structure", ["separable", "isotropic", "covariant"])
def test_fields_none_adaptive(structure):
    pc, _, w = _cloud(3, n=1000)
    kn = "tsc" if structure == "separable" else "cubic_spline"
    pc.global_setup(num_neighbors=16, structure=structure, kernel_name=kn)
    pc.find_neighbors()
    pc.compute_smoothing()
    kw = dict(averaged=False, gridnums=12, structure=structure, kernel_name=kn)
    mass = pc.deposit(fields=None, **kw)
    _, weights = pc.deposit(fields=w, return_weights=True, **kw)
    np.testing.assert_allclose(mass, weights, rtol=1e-4, atol=1e-6 * weights.max())


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("kernel_name", ["ngp", "cic", "tsc"])
def test_extent_selects_inside_particles(dim, kernel_name):
    """deposit(extent=...) == backend deposit of the particles inside it."""
    pc, pos, w = _cloud(dim)
    lo, hi = 0.2, 0.7
    extent = [[lo, hi]] * dim
    fg, wg = pc.deposit(fields=w, averaged=False, gridnums=10, extent=extent, adaptive=False,
                        kernel_name=kernel_name, return_weights=True)

    p32 = pc.decomposition.local_positions
    inside = np.all((p32 >= lo) & (p32 <= hi), axis=1)
    ref_f, ref_w = tdep.deposit(p32[inside] - np.float32(lo), w[inside, None], w[inside],
                                np.full(dim, hi - lo, np.float32), [10] * dim, adaptive=False,
                                kernel_name=kernel_name, periodic=False)
    np.testing.assert_allclose(wg, ref_w, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(fg, ref_f, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("structure", ["isotropic", "covariant"])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("periodic", [True, False])
def test_sample_buffer_matches_direct(structure, dim, periodic):
    """Local-buffer sample branch == direct per-sample atomics (buffer size 1
    forces every particle onto the direct path)."""
    rng = np.random.default_rng(SEED)
    n, g = 3000, {1: 64, 2: 32, 3: 16}[dim]
    pos = rng.random((n, dim), dtype=np.float32)
    fld = rng.random((n, 2), dtype=np.float32)
    w = rng.uniform(0.5, 1.5, n).astype(np.float32)
    box = np.ones(dim, np.float32)
    h = (rng.uniform(0.2, 3.9, n) / g / 2).astype(np.float32)  # eta < eta_crit=4
    common = (pos, fld, w, box, (g,) * dim, periodic, "cubic_spline")
    if structure == "isotropic":
        prep = tdep._prepare_isotropic(*common, h, "midpoint", 4.0)
    else:
        vecs = np.linalg.qr(rng.normal(size=(n, dim, dim)))[0].astype(np.float32)
        vals = (h[:, None] * rng.uniform(0.3, 1.0, (n, dim))).astype(np.float32)
        prep = tdep._prepare_covariant(*common, vecs, vals, "midpoint", 4.0)
    assert prep.sample_buffer > 1

    prep.kernel(*prep.args)
    buffered = prep.grid_weights.copy(), prep.grid_fields.copy()

    # args: (evaluate, sigma, support, dim, quad_points, periodic, eta_crit, B, ...)
    args = list(prep.args)
    assert args[7] == prep.sample_buffer
    args[7] = 1
    prep.grid_weights[:] = 0
    prep.grid_fields[:] = 0
    prep.kernel(*args)

    for got, ref in zip(buffered, (prep.grid_weights, prep.grid_fields)):
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5 * ref.max())
    assert buffered[0].sum() == pytest.approx(prep.grid_weights.sum(), rel=1e-5)


@pytest.mark.parametrize("kernel_name", ["cubic_spline", "wendland_c4", "gaussian", "tophat"])
@pytest.mark.parametrize("method", ["midpoint", "trapezoidal", "simpson"])
@pytest.mark.parametrize("periodic", [True, False])
def test_covariant_1d_equals_isotropic_1d(kernel_name, method, periodic):
    """In 1D a covariant kernel with eigenvector +-1 and eigenvalue h is the
    isotropic kernel with smoothing length h (regression: the 1D covariant
    quadrature used a signed q)."""
    rng = np.random.default_rng(SEED)
    n, g = 3000, 64
    pos = rng.random((n, 1), dtype=np.float32)
    fld = rng.random((n, 2), dtype=np.float32)
    w = rng.uniform(0.5, 1.5, n).astype(np.float32)
    box = np.ones(1, np.float32)
    h = (rng.uniform(0.2, 8.0, n) / g / 2).astype(np.float32)  # both branches
    sign = rng.choice([-1.0, 1.0], n).astype(np.float32)
    kw = dict(adaptive=True, kernel_name=kernel_name, periodic=periodic, integration_method=method)
    fi, wi = tdep.deposit(pos, fld, w, box, (g,), structure="isotropic", particle_hsml=h, **kw)
    fc, wc = tdep.deposit(pos, fld, w, box, (g,), structure="covariant",
                          particle_hmat_eigvecs=sign[:, None, None],
                          particle_hmat_eigvals=h[:, None], **kw)
    np.testing.assert_allclose(wc, wi, rtol=1e-5, atol=1e-6 * wi.max())
    np.testing.assert_allclose(fc, fi, rtol=1e-5, atol=1e-6 * np.abs(fi).max())
