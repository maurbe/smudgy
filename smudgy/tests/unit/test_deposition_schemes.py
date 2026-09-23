"""Weight-level correctness checks for the fixed-grid mass-assignment schemes.

`p` below is B-spline **order** -- point count, `= degree + 1`, the exponent
that would appear in `sinc^p(k H/2)` for this scheme's window -- never
"degree". The tutorial notebook's "Degree" column is a different, `p - 1`,
number; don't confuse the two.
"""

import numpy as np
import pytest

from smudgy.backend.numpy import deposition as dep

# (kernel_name, order p)
SCHEMES = [
    ("ngp", 1),
    ("cic", 2),
    ("tsc", 3),
    ("bspline4", 4),
    ("bspline5", 5),
    ("bspline6", 6),
]

# Support half-width in cells, by scheme name.
SUPPORT_HALF_WIDTH = {
    "ngp": 0.5,
    "cic": 1.0,
    "tsc": 1.5,
    "bspline4": 2.0,
    "bspline5": 2.5,
    "bspline6": 3.0,
}

GRIDNUM = 64
SEED = 42


def _deposit_single_particle(name, dim, pos, gridnum=GRIDNUM):
    """Deposit one unit-weight particle and return the flattened grid weights."""
    kernel = dep._KERNELS[(name, dim)]
    particle_positions = np.asarray([pos], dtype=np.float32)
    particle_weights = np.array([1.0], dtype=np.float32)
    particle_fields = particle_weights[:, None]
    boxsizes = np.ones(dim, dtype=np.float32)
    gridnums = [gridnum] * dim
    _, grid_weights = kernel(
        particle_positions,
        particle_fields,
        particle_weights,
        boxsizes,
        gridnums,
        True,  # periodic
    )
    return grid_weights


@pytest.mark.parametrize("name,p", SCHEMES)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_partition_of_unity(name, p, dim):
    """Deposited weights for a single particle sum to 1, for random sub-cell offsets."""
    rng = np.random.default_rng(SEED)
    for _ in range(20):
        pos = rng.uniform(0.0, 1.0, size=dim)
        gw = _deposit_single_particle(name, dim, pos)
        assert gw.sum() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("name,p", SCHEMES)
def test_stencil_point_count_1d(name, p):
    """A single particle touches at most p grid points along one axis.

    At exact boundary offsets (e.g. d = +/-0.5 for bspline5) the outermost
    weight can be exactly zero, so this asserts `<=`, not `==`.
    """
    rng = np.random.default_rng(SEED)
    for _ in range(50):
        pos = rng.uniform(0.0, 1.0, size=1)
        gw = _deposit_single_particle(name, 1, pos)
        nnz = np.count_nonzero(gw > 1e-8)
        assert nnz <= p


@pytest.mark.parametrize("name,p", SCHEMES)
def test_support_half_width_1d(name, p):
    """Nonzero weights stay within the scheme's known support half-width."""
    half_width = SUPPORT_HALF_WIDTH[name]
    rng = np.random.default_rng(SEED)
    H = 1.0 / GRIDNUM
    for _ in range(50):
        # Stay well clear of the periodic wrap so index arithmetic stays linear.
        pos = rng.uniform(0.1, 0.9, size=1)
        gw = _deposit_single_particle(name, 1, pos)
        nz_idx = np.nonzero(gw > 1e-8)[0]
        grid_pos = pos[0] / H - 0.5  # same convention deposition.py uses internally
        dist_cells = np.abs(nz_idx - grid_pos)
        assert np.all(dist_cells <= half_width + 1e-6)


@pytest.mark.parametrize("name,p", [(n, p) for n, p in SCHEMES if p >= 2])
def test_centroid_1d(name, p):
    """For p >= 2, the deposited centre of mass equals the particle position."""
    rng = np.random.default_rng(SEED)
    H = 1.0 / GRIDNUM
    for _ in range(50):
        pos = rng.uniform(0.1, 0.9, size=1)
        gw = _deposit_single_particle(name, 1, pos)
        idx = np.arange(GRIDNUM)
        centroid = (idx * gw).sum() / gw.sum()
        grid_pos = pos[0] / H - 0.5
        assert centroid == pytest.approx(grid_pos, abs=1e-4)


@pytest.mark.parametrize("name", ["bspline4", "bspline5", "bspline6"])
@pytest.mark.parametrize("dim", [2, 3])
def test_non_cubic_grid_mass_conserved(name, dim):
    """Regression test for the 3D flat-index bug: a non-cubic grid (Ny != Nz)
    silently corrupted deposition for the old "pqs" (now bspline6) scheme.
    Covers all three B-spline schemes, both backends aren't touched here (see
    test_backend_consistency_deposition.py for that), 2D and 3D.
    """
    gridnums = [12, 16, 20][:dim]
    rng = np.random.default_rng(SEED)
    N = 2000
    positions = rng.uniform(0.0, 1.0, size=(N, dim)).astype(np.float32)
    weights = np.ones(N, dtype=np.float32)
    fields = weights[:, None]
    boxsizes = np.ones(dim, dtype=np.float32)

    grid_fields, grid_weights = dep.deposit(
        positions, fields, weights, boxsizes, gridnums, True, name
    )

    assert grid_weights.shape == tuple(gridnums)
    assert np.isfinite(grid_weights).all()
    assert grid_weights.sum() == pytest.approx(weights.sum(), rel=1e-5)
