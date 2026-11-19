"""Fixed-grid (B-spline stencil) deposition, NumPy reference backend.

One generic routine for every stencil and dimension: per axis, each particle
gets K cell indices and weights (the textbook B-spline assignment functions on
a cell-centred grid, as in the Quadrants backend); the K**dim tensor-product
contributions are accumulated with ``np.bincount`` in float64.
"""

from itertools import product

import numpy as np

from .._common import as_f32, check_fixed_grid_kernel


# -----------------------------------------------------------------------------
# 1D stencils: g = position in cell units -> (base (N,), weights (N, K)); the
# stencil covers cells base - SHIFT + o for o in [0, K).
# -----------------------------------------------------------------------------
def _ngp(g):
    return np.floor(g), np.ones((g.shape[0], 1), dtype=g.dtype)


def _cic(g):
    gx = g - 0.5
    ix = np.floor(gx)
    fx = gx - ix
    return ix, np.stack([1.0 - fx, fx], axis=1)


def _tsc(g):
    gx = g - 0.5
    ix = np.rint(gx)
    dx = gx - ix
    return ix, np.stack(
        [0.5 * (0.5 - dx) ** 2, 0.75 - dx * dx, 0.5 * (0.5 + dx) ** 2], axis=1
    )


def _bspline4(g):
    """Cubic B-spline."""
    gx = g - 0.5
    ix = np.floor(gx)
    dx = gx - ix
    return ix, np.stack(
        [
            (1.0 - dx) ** 3 / 6.0,
            (4.0 - 6.0 * dx**2 + 3.0 * dx**3) / 6.0,
            (1.0 + 3.0 * dx + 3.0 * dx**2 - 3.0 * dx**3) / 6.0,
            dx**3 / 6.0,
        ],
        axis=1,
    )


def _bspline5(g):
    """Quartic B-spline (5-point)."""
    gx = g - 0.5
    ix = np.floor(gx + 0.5)
    dx = gx - ix
    return ix, np.stack(
        [
            (1.0 - 2.0 * dx) ** 4 / 384.0,
            (
                55.0
                + 20.0 * (1.0 + dx)
                - 120.0 * (1.0 + dx) ** 2
                + 80.0 * (1.0 + dx) ** 3
                - 16.0 * (1.0 + dx) ** 4
            )
            / 96.0,
            115.0 / 192.0 - (5.0 / 8.0) * dx**2 + (1.0 / 4.0) * dx**4,
            (
                55.0
                + 20.0 * (1.0 - dx)
                - 120.0 * (1.0 - dx) ** 2
                + 80.0 * (1.0 - dx) ** 3
                - 16.0 * (1.0 - dx) ** 4
            )
            / 96.0,
            (1.0 + 2.0 * dx) ** 4 / 384.0,
        ],
        axis=1,
    )


def _bspline6(g):
    """Quintic B-spline (6-point)."""
    gx = g - 0.5
    ix = np.floor(gx)
    dx = gx - ix
    return ix, np.stack(
        [
            (1.0 - dx) ** 5 / 120.0,
            ((2.0 - dx) ** 5 - 6.0 * (1.0 - dx) ** 5) / 120.0,
            ((3.0 - dx) ** 5 - 6.0 * (2.0 - dx) ** 5 + 15.0 * (1.0 - dx) ** 5) / 120.0,
            ((2.0 + dx) ** 5 - 6.0 * (1.0 + dx) ** 5 + 15.0 * dx**5) / 120.0,
            ((1.0 + dx) ** 5 - 6.0 * dx**5) / 120.0,
            dx**5 / 120.0,
        ],
        axis=1,
    )


# name -> (stencil, SHIFT = offset of the first cell)
_STENCILS = {
    "ngp": (_ngp, 0),
    "cic": (_cic, 0),
    "tsc": (_tsc, 1),
    "bspline4": (_bspline4, 1),
    "bspline5": (_bspline5, 2),
    "bspline6": (_bspline6, 2),
}

# (cell, particle) pairs accumulated per np.bincount call (bounds memory).
_BATCH = 1 << 24


def deposit(
    particle_positions,
    particle_fields,
    particle_weights,
    boxsizes,
    gridnums,
    periodic,
    kernel_name,
    *,
    adaptive=False,
    structure=None,
    **kwargs,
):
    """Deposit particle data onto a grid with a fixed-grid stencil (NumPy backend).

    Only fixed-grid deposition (``adaptive=False``) with the stencils ``ngp``,
    ``cic``, ``tsc``, ``bspline4``, ``bspline5`` and ``bspline6`` is supported.
    Grid cells cover ``[0, boxsizes)``; stencil cells outside the grid are
    wrapped (periodic) or dropped.

    Returns
    -------
    grid_fields : ndarray, shape (F, *gridnums)
    grid_weights : ndarray, shape (*gridnums,)
    """
    if adaptive:
        raise ValueError(
            "backend='numpy' does not support adaptive deposition; "
            "use backend='quadrants'."
        )
    check_fixed_grid_kernel(kernel_name, structure)

    positions = as_f32(particle_positions)
    fields = as_f32(particle_fields)
    weights = as_f32(particle_weights)
    boxsizes = as_f32(boxsizes)
    gridnums = tuple(int(n) for n in gridnums)
    n, dim = positions.shape
    F = fields.shape[1]
    n_cells = int(np.prod(gridnums))
    stencil, shift = _STENCILS[kernel_name]

    # per axis: cell indices (n, K), in-grid mask (n, K) or None, weights (n, K)
    inv_dx = np.array(gridnums, dtype=np.float32) / boxsizes
    axes = []
    for d in range(dim):
        base, w = stencil(positions[:, d] * inv_dx[d])
        idx = base.astype(np.int64)[:, None] + (np.arange(w.shape[1]) - shift)
        if periodic:
            axes.append((idx % gridnums[d], None, w))
        else:
            axes.append((idx, (idx >= 0) & (idx < gridnums[d]), w))

    strides = [int(np.prod(gridnums[d + 1 :])) for d in range(dim)]  # row-major
    weighted_fields = fields * weights[:, None]
    grid_w = np.zeros(n_cells, dtype=np.float64)
    grid_f = np.zeros((F, n_cells), dtype=np.float64)
    batch = []  # (cells, particles, stencil weights) per stencil offset

    def flush():
        if not batch:
            return
        cells = np.concatenate([b[0] for b in batch])
        rows = np.concatenate([b[1] for b in batch])
        w = np.concatenate([b[2] for b in batch])
        grid_w[:] += np.bincount(cells, weights=w * weights[rows], minlength=n_cells)
        for f in range(F):
            grid_f[f] += np.bincount(
                cells, weights=w * weighted_fields[rows, f], minlength=n_cells
            )
        batch.clear()

    particles = np.arange(n)
    pending = 0
    for offsets in product(range(axes[0][2].shape[1]), repeat=dim):
        cells = np.zeros(n, dtype=np.int64)
        w = np.ones(n, dtype=np.float32)
        ok = None
        for d, o in enumerate(offsets):
            idx, ok_d, w_d = axes[d]
            cells += idx[:, o] * strides[d]
            w = w * w_d[:, o]
            if ok_d is not None:
                ok = ok_d[:, o] if ok is None else ok & ok_d[:, o]
        if ok is None:
            batch.append((cells, particles, w))
        else:
            batch.append((cells[ok], particles[ok], w[ok]))
        pending += batch[-1][0].shape[0]
        if pending >= _BATCH:
            flush()
            pending = 0
    flush()

    return (
        grid_f.reshape(F, *gridnums).astype(np.float32),
        grid_w.reshape(gridnums).astype(np.float32),
    )
