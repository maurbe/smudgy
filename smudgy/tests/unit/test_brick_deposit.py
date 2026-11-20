"""Single-rank checks of brick deposition (``deposit(brick=True)``).

A brick placed onto the global grid must reproduce the full-grid deposit:
every cell value is computed by the same float32 arithmetic, so on one thread
the two agree bit for bit; the tolerance only absorbs the nondeterministic
order of multi-threaded atomic adds.
"""

import numpy as np
import pytest
from mpi4py import MPI

from smudgy import PointCloud, execution
from smudgy.backend.quadrants import deposition as qdep

RTOL, ATOL = 1e-6, 1e-7


def _place(grids, offset, gridnums):
    """Brick grids -> full grids, through the MPI reduction on one rank."""
    return execution._sum_bricks(MPI.COMM_SELF, grids, offset, gridnums)


def _particles(rng, n, dim, where):
    """Particles in a blob at the box centre, or straddling the corner 0 == 1."""
    centre = np.full(dim, 0.5 if where == "centre" else 0.0)
    pos = centre + 0.06 * rng.standard_normal((n, dim))
    return np.mod(pos, 1.0).astype(np.float32)


def _smoothing(rng, n, dim, structure, scale):
    h = (scale * rng.uniform(0.3, 1.5, n)).astype(np.float32)
    if structure == "separable":
        return dict(particle_hsml=(h[:, None] * rng.uniform(0.5, 1.5, (n, dim))).astype(np.float32))
    if structure == "isotropic":
        return dict(particle_hsml=h)
    vecs = np.linalg.qr(rng.normal(size=(n, dim, dim)))[0].astype(np.float32)
    vals = (h[:, None] * rng.uniform(0.2, 1.0, (n, dim))).astype(np.float32)
    return dict(particle_hmat_eigvecs=vecs, particle_hmat_eigvals=vals)


METHODS = [
    ("cic", None), ("tsc", None), ("bspline5", None), ("bspline6", None),
    ("tsc", "separable"), ("gaussian", "separable"),
    ("cubic_spline", "isotropic"), ("wendland_c2", "isotropic"),
    ("cubic_spline", "covariant"), ("gaussian", "covariant"),
]


@pytest.mark.parametrize("kernel_name,structure", METHODS)
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("where", ["centre", "corner"])
# small h: mostly sample table; large h: mostly cell quadrature
@pytest.mark.parametrize("scale", [0.01, 0.05])
def test_brick_equals_full_grid(kernel_name, structure, dim, periodic, where, scale):
    rng = np.random.default_rng(7)
    n = 400
    gridnums = (40, 32, 24)[:dim]
    pos = _particles(rng, n, dim, where)
    fields = rng.random((n, 2), dtype=np.float32)
    weights = rng.uniform(0.5, 1.5, n).astype(np.float32)
    kw = dict(adaptive=structure is not None, kernel_name=kernel_name,
              structure=structure, periodic=periodic)
    if structure is not None:
        kw.update(_smoothing(rng, n, dim, structure, scale))
    args = (pos, fields, weights, np.ones(dim, np.float32), gridnums)

    full = qdep.deposit(*args, **kw)
    *bricks, offset = qdep.deposit(*args, **kw, brick=True)
    # the blob is narrow: the brick must really be smaller than the grid
    if scale == 0.01 and (where == "centre" or periodic):
        assert bricks[1].size < np.prod(gridnums)
    placed = _place(tuple(bricks), offset, gridnums)
    for p, f in zip(placed, full):
        np.testing.assert_allclose(p, f, rtol=RTOL, atol=ATOL * float(np.abs(f).max()))


def test_corner_brick_wraps_around():
    """A blob at the periodic corner gives a brick that wraps around the axes."""
    rng = np.random.default_rng(1)
    pos = _particles(rng, 300, 2, "corner")
    *bricks, offset = qdep.deposit(
        pos, np.zeros((300, 0), np.float32), np.ones(300, np.float32),
        np.ones(2, np.float32), (64, 64), adaptive=False, kernel_name="cic",
        periodic=True, brick=True,
    )
    shape = np.array(bricks[1].shape)
    assert np.all(shape < 32) and np.all(offset + shape > 64)


def test_empty_particle_set():
    *bricks, offset = qdep.deposit(
        np.zeros((0, 3), np.float32), np.zeros((0, 1), np.float32),
        np.zeros(0, np.float32), np.ones(3, np.float32), (8, 8, 8),
        adaptive=True, kernel_name="cubic_spline", structure="isotropic",
        periodic=False, particle_hsml=np.zeros(0, np.float32), brick=True,
    )
    assert bricks[1].size == 0
    placed = _place(tuple(bricks), offset, (8, 8, 8))
    assert placed[0].shape == (1, 8, 8, 8) and not placed[1].any()


@pytest.mark.parametrize(
    "offset,shape,n,expected",
    [
        ((2,), (3,), 8, [((0, 3), (2, 5))]),
        ((-2,), (5,), 8, [((0, 2), (6, 8)), ((2, 5), (0, 3))]),
        ((6,), (4,), 8, [((0, 2), (6, 8)), ((2, 4), (0, 2))]),
        ((0,), (8,), 8, [((0, 8), (0, 8))]),
        ((3,), (0,), 8, []),
    ],
)
def test_brick_pieces(offset, shape, n, expected):
    pieces = [
        ((b[0].start, b[0].stop), (g[0].start, g[0].stop))
        for b, g in execution._brick_pieces(offset, shape, (n,))
    ]
    assert pieces == expected


@pytest.mark.parametrize("structure", ["isotropic", "covariant"])
@pytest.mark.parametrize("extent", [None, [[0.1, 0.6], [0.2, 0.9], [0.0, 1.0]]])
def test_pointcloud_brick_keyword(structure, extent):
    rng = np.random.default_rng(2)
    pos = rng.uniform(0, 1, (600, 3)).astype(np.float32)
    pc = PointCloud(pos, boxsize=1.0, verbose=False).global_setup(
        num_neighbors=16, structure=structure, kernel_name="cubic_spline")
    pc.add_fields("f", rng.random(600).astype(np.float32))
    pc.compute_smoothing()
    kw = dict(fields="f", averaged=True, gridnums=16, extent=extent, return_weights=True)
    for b, f in zip(pc.deposit(**kw, brick=True), pc.deposit(**kw, brick=False)):
        np.testing.assert_allclose(b, f, rtol=1e-5, atol=1e-7)
    weights = pc.deposit(None, averaged=False, gridnums=16, plane_projection=[0, 2])
    np.testing.assert_allclose(
        weights,
        pc.deposit(None, averaged=False, gridnums=16, plane_projection=[0, 2], brick=False),
        rtol=1e-6, atol=1e-7,
    )
