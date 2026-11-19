"""Particle-to-grid deposition (Quadrants backend).

Fixed-grid deposition uses B-spline stencils (NGP ... 6-point). Adaptive
deposition spreads each particle with its own smoothing: separable
(tensor-product kernels, integrated exactly per cell), isotropic or covariant
(radial kernels; sampled for kernels small against the cell size, per-cell
quadrature otherwise). Every kernel is compiled per dimension from one source.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import quadrants as qd

from .._common import ADAPTIVE_STRUCTURES, as_f32, check_fixed_grid_kernel
from . import check_index_limits
from .kernels import (
    _separable_integrate_1d,
    build_kernel_sample_grid,
    create_separable_kernel,
    create_spherical_kernel,
)


# =============================================================================
# Shared helpers
# =============================================================================
@qd.func
def _apply_pbc(idx: qd.i32, gridnum: qd.i32) -> qd.i32:
    r = idx % gridnum
    if r < 0:
        r += gridnum
    return r


@qd.func
def _cell_index_from_pos(
    pos: qd.f32, cellSize_inv: qd.f32, gridnum: qd.i32, periodic: qd.template()
):
    """Return ``(cell index, valid)`` of a position along one axis.

    Periodic grids wrap the index (always valid); otherwise positions outside
    the grid are invalid.
    """
    idx = qd.floor(pos * cellSize_inv, int)
    valid = True
    if qd.static(periodic):
        idx = _apply_pbc(idx, gridnum)
    else:
        if idx < 0 or idx >= gridnum:
            valid = False
    return idx, valid


@qd.func
def _cell_index_from_int(idx: qd.i32, gridnum: qd.i32, periodic: qd.template()):
    """`_cell_index_from_pos` for an already-integer (unwrapped) cell index."""
    valid = True
    if qd.static(periodic):
        idx = _apply_pbc(idx, gridnum)
    else:
        if idx < 0 or idx >= gridnum:
            valid = False
    return idx, valid


def _sample_buffer_size(eta_crit, dim):
    """Return the cells per axis of the sample branch's per-particle buffer.

    A particle in that branch has kernel diameter eta < eta_crit cells along
    its narrowest axis, so (for isotropic kernels on cubic cells) it touches
    at most ceil(eta_crit) + 1 cells per axis. Particles that don't fit
    (elongated covariant ellipsoids, anisotropic cells) fall back to direct
    per-sample deposition. Capped so the local buffer (B**dim floats) stays
    small; a compile-time constant so Quadrants can keep it in registers/stack.
    """
    b = int(np.ceil(eta_crit)) + 1
    return max(1, min(b, {1: 64, 2: 16, 3: 6}[dim]))



def _round_up_stencil(n, bucket=8):
    """Round the stencil size up to the nearest multiple of `bucket`.

    Keeps MAX_STENCIL (a qd.template() arg) taking only a handful of
    distinct values across a run, so Quadrants reuses one compiled kernel
    per bucket instead of JIT-recompiling every time smoothing lengths
    drift by a tiny amount between timesteps.
    """
    return ((n + bucket - 1) // bucket) * bucket


# =============================================================================
# Quadrature helpers
# =============================================================================
QUADRATURE_POINTS_1D = {
    "midpoint": ((0.5, 1.0),),
    "trapezoidal": (
        (0.0, 0.5),
        (1.0, 0.5),
    ),
    "simpson": (
        (0.0, 1.0 / 6.0),
        (0.5, 4.0 / 6.0),
        (1.0, 1.0 / 6.0),
    ),
}

QUADRATURE_POINTS_2D = {
    "midpoint": ((0.5, 0.5, 1.0),),
    "trapezoidal": (
        (0.0, 0.0, 0.25),
        (1.0, 0.0, 0.25),
        (0.0, 1.0, 0.25),
        (1.0, 1.0, 0.25),
    ),
    "simpson": (
        (0.0, 0.0, 1.0 / 36.0),
        (1.0, 0.0, 1.0 / 36.0),
        (0.0, 1.0, 1.0 / 36.0),
        (1.0, 1.0, 1.0 / 36.0),
        (0.5, 0.0, 4.0 / 36.0),
        (0.5, 1.0, 4.0 / 36.0),
        (0.0, 0.5, 4.0 / 36.0),
        (1.0, 0.5, 4.0 / 36.0),
        (0.5, 0.5, 16.0 / 36.0),
    ),
}

QUADRATURE_POINTS_3D = {
    "midpoint": ((0.5, 0.5, 0.5, 1.0),),
    "trapezoidal": (
        (0.0, 0.0, 0.0, 1.0 / 8.0),
        (1.0, 0.0, 0.0, 1.0 / 8.0),
        (0.0, 1.0, 0.0, 1.0 / 8.0),
        (1.0, 1.0, 0.0, 1.0 / 8.0),
        (0.0, 0.0, 1.0, 1.0 / 8.0),
        (1.0, 0.0, 1.0, 1.0 / 8.0),
        (0.0, 1.0, 1.0, 1.0 / 8.0),
        (1.0, 1.0, 1.0, 1.0 / 8.0),
    ),
    "simpson": (
        # 8 corners, weight 1/216
        (0.0, 0.0, 0.0, 1.0 / 216.0),
        (1.0, 0.0, 0.0, 1.0 / 216.0),
        (0.0, 1.0, 0.0, 1.0 / 216.0),
        (1.0, 1.0, 0.0, 1.0 / 216.0),
        (0.0, 0.0, 1.0, 1.0 / 216.0),
        (1.0, 0.0, 1.0, 1.0 / 216.0),
        (0.0, 1.0, 1.0, 1.0 / 216.0),
        (1.0, 1.0, 1.0, 1.0 / 216.0),
        # 12 edge midpoints, weight 4/216
        (0.5, 0.0, 0.0, 4.0 / 216.0),
        (0.0, 0.5, 0.0, 4.0 / 216.0),
        (0.0, 0.0, 0.5, 4.0 / 216.0),
        (0.5, 1.0, 0.0, 4.0 / 216.0),
        (1.0, 0.5, 0.0, 4.0 / 216.0),
        (1.0, 0.0, 0.5, 4.0 / 216.0),
        (0.5, 0.0, 1.0, 4.0 / 216.0),
        (0.0, 0.5, 1.0, 4.0 / 216.0),
        (0.0, 1.0, 0.5, 4.0 / 216.0),
        (0.5, 1.0, 1.0, 4.0 / 216.0),
        (1.0, 0.5, 1.0, 4.0 / 216.0),
        (1.0, 1.0, 0.5, 4.0 / 216.0),
        # 6 face centers, weight 16/216
        (0.5, 0.5, 0.0, 16.0 / 216.0),
        (0.5, 0.5, 1.0, 16.0 / 216.0),
        (0.5, 0.0, 0.5, 16.0 / 216.0),
        (0.5, 1.0, 0.5, 16.0 / 216.0),
        (0.0, 0.5, 0.5, 16.0 / 216.0),
        (1.0, 0.5, 0.5, 16.0 / 216.0),
        # center, weight 64/216
        (0.5, 0.5, 0.5, 64.0 / 216.0),
    ),
}


# =============================================================================
# ||                                                                         ||
# ||                          STATIC KERNELS                                 ||
# ||                                                                         ||
# =============================================================================


# -----------------------------------------------------------------------------
# 1D stencils. Each takes g = position in cell units (pos * N / L, NOT shifted
# by the half-cell offset) and returns (base, w): the stencil covers cells
# base - SHIFT + o for o in [0, K), with weights w[o]. The weight formulas are
# the textbook B-spline assignment functions (cell-centred grid).
# -----------------------------------------------------------------------------
@qd.func
def _ngp_stencil(g: qd.f32):
    return qd.floor(g, int), qd.Vector([1.0])


@qd.func
def _cic_stencil(g: qd.f32):
    gx = g - 0.5
    ix = qd.floor(gx, int)
    fx = gx - ix
    return ix, qd.Vector([1.0 - fx, fx])


@qd.func
def _tsc_stencil(g: qd.f32):
    gx = g - 0.5
    ix = qd.round(gx, int)
    dx = gx - ix
    return ix, qd.Vector(
        [0.5 * (0.5 - dx) ** 2, 0.75 - dx * dx, 0.5 * (0.5 + dx) ** 2]
    )


@qd.func
def _bspline4_stencil(g: qd.f32):
    """Cubic B-spline (4-point)."""
    gx = g - 0.5
    ix = qd.floor(gx, int)
    dx = gx - ix
    return ix, qd.Vector(
        [
            (1.0 - dx) ** 3 / 6.0,
            (4.0 - 6.0 * dx**2 + 3.0 * dx**3) / 6.0,
            (1.0 + 3.0 * dx + 3.0 * dx**2 - 3.0 * dx**3) / 6.0,
            dx**3 / 6.0,
        ]
    )


@qd.func
def _bspline5_stencil(g: qd.f32):
    """Quartic B-spline (order 5, 5-point -- the textbook "PQS")."""
    gx = g - 0.5
    ix = qd.floor(gx + 0.5, int)
    dx = gx - ix
    return ix, qd.Vector(
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
        ]
    )


@qd.func
def _bspline6_stencil(g: qd.f32):
    """Quintic B-spline (6-point)."""
    gx = g - 0.5
    ix = qd.floor(gx, int)
    dx = gx - ix
    return ix, qd.Vector(
        [
            (1.0 - dx) ** 5 / 120.0,
            ((2.0 - dx) ** 5 - 6.0 * (1.0 - dx) ** 5) / 120.0,
            ((3.0 - dx) ** 5 - 6.0 * (2.0 - dx) ** 5 + 15.0 * (1.0 - dx) ** 5)
            / 120.0,
            ((2.0 + dx) ** 5 - 6.0 * (1.0 + dx) ** 5 + 15.0 * dx**5) / 120.0,
            ((1.0 + dx) ** 5 - 6.0 * dx**5) / 120.0,
            dx**5 / 120.0,
        ]
    )


# name -> (stencil func, K = stencil width, SHIFT = offset of the first cell)
_FIXED_STENCILS = {
    "ngp": (_ngp_stencil, 1, 0),
    "cic": (_cic_stencil, 2, 0),
    "tsc": (_tsc_stencil, 3, 1),
    "bspline4": (_bspline4_stencil, 4, 1),
    "bspline5": (_bspline5_stencil, 5, 2),
    "bspline6": (_bspline6_stencil, 6, 2),
}


def _make_fixed_grid_kernel(dim, stencil_fn, K, SHIFT):
    """Build the fixed-grid deposition kernel for one (stencil, dim).

    Per particle, the K weights and K (wrapped) cell indices of every axis
    are computed once; the K**dim stencil loop is then fully unrolled
    (qd.static), so all weight/index lookups are static and the periodic
    wrap is done K*dim times instead of once per stencil point. Out-of-grid
    stencil points are skipped for non-periodic grids.
    """

    @qd.kernel
    def kernel(
        periodic: qd.template(),
        particle_positions: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, D)
        particle_fields: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, F)
        particle_weights: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (N,)
        boxsizes: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (D,)
        grid_fields: qd.types.ndarray(dtype=qd.f32, ndim=dim + 1),  # (*gridnums, F)
        grid_weights: qd.types.ndarray(dtype=qd.f32, ndim=dim),  # gridnums
    ):
        F = grid_fields.shape[dim]
        N = qd.Vector.zero(qd.i32, dim)
        inv_dx = qd.Vector.zero(qd.f32, dim)
        for d in qd.static(range(dim)):
            N[d] = grid_weights.shape[d]
            inv_dx[d] = N[d] / boxsizes[d]

        for p in range(particle_positions.shape[0]):
            idx = qd.Matrix.zero(qd.i32, dim, K)
            wgt = qd.Matrix.zero(qd.f32, dim, K)
            ok = qd.Matrix.zero(qd.i32, dim, K)
            for d in qd.static(range(dim)):
                base, w1d = stencil_fn(particle_positions[p, d] * inv_dx[d])
                for o in qd.static(range(K)):
                    i = base + o - SHIFT
                    if qd.static(periodic):
                        i = _apply_pbc(i, N[d])
                        ok[d, o] = 1
                    else:
                        ok[d, o] = 0 <= i < N[d]
                    idx[d, o] = i
                    wgt[d, o] = w1d[o]

            w_particle = particle_weights[p]

            if qd.static(dim == 1):
                for ox in qd.static(range(K)):
                    if ok[0, ox]:
                        x = idx[0, ox]
                        w = wgt[0, ox] * w_particle
                        qd.atomic_add(grid_weights[x], w)
                        for f in range(F):
                            qd.atomic_add(grid_fields[x, f], particle_fields[p, f] * w)
            elif qd.static(dim == 2):
                for ox in qd.static(range(K)):
                    for oy in qd.static(range(K)):
                        if ok[0, ox] and ok[1, oy]:
                            x, y = idx[0, ox], idx[1, oy]
                            w = wgt[0, ox] * wgt[1, oy] * w_particle
                            qd.atomic_add(grid_weights[x, y], w)
                            for f in range(F):
                                qd.atomic_add(
                                    grid_fields[x, y, f], particle_fields[p, f] * w
                                )
            else:
                for ox in qd.static(range(K)):
                    for oy in qd.static(range(K)):
                        for oz in qd.static(range(K)):
                            if ok[0, ox] and ok[1, oy] and ok[2, oz]:
                                x, y, z = idx[0, ox], idx[1, oy], idx[2, oz]
                                w = wgt[0, ox] * wgt[1, oy] * wgt[2, oz] * w_particle
                                qd.atomic_add(grid_weights[x, y, z], w)
                                for f in range(F):
                                    qd.atomic_add(
                                        grid_fields[x, y, z, f],
                                        particle_fields[p, f] * w,
                                    )

    return kernel


_FIXED_GRID_KERNELS = {
    (name, dim): _make_fixed_grid_kernel(dim, fn, K, SHIFT)
    for name, (fn, K, SHIFT) in _FIXED_STENCILS.items()
    for dim in (1, 2, 3)
}


# =============================================================================
# Adaptive deposition: shared helpers
# =============================================================================
# The adaptive kernels below are written once for 1, 2 and 3 dimensions; `dim`
# is a template argument, so every per-axis loop is unrolled at compile time.
# Sums and products over axes start from the first term and run in axis order,
# so they compile to the same floating-point expressions as hand-written
# per-dimension code (results are bit-identical to that code).
@qd.func
def _grid_add(
    grid_weights: qd.template(),
    grid_fields: qd.template(),
    quantities: qd.template(),
    n,
    cell,
    w,
    dim: qd.template(),
):
    """Add weight `w` of particle `n` (and its fields times `w`) to grid cell `cell`."""
    F = grid_fields.shape[dim]
    if qd.static(dim == 1):
        qd.atomic_add(grid_weights[cell[0]], w)
        for f in range(F):
            qd.atomic_add(grid_fields[cell[0], f], quantities[n, f] * w)
    elif qd.static(dim == 2):
        qd.atomic_add(grid_weights[cell[0], cell[1]], w)
        for f in range(F):
            qd.atomic_add(grid_fields[cell[0], cell[1], f], quantities[n, f] * w)
    else:
        qd.atomic_add(grid_weights[cell[0], cell[1], cell[2]], w)
        for f in range(F):
            qd.atomic_add(
                grid_fields[cell[0], cell[1], cell[2], f], quantities[n, f] * w
            )


@qd.func
def _box_count(ext, dim: qd.template()) -> qd.i32:
    """Return the number of cells of a box with extents `ext` (0 if any is <= 0)."""
    count = 1
    for d in qd.static(range(dim)):
        if ext[d] <= 0:
            count = 0
        count *= qd.max(ext[d], 0)
    return count


@qd.func
def _next_offset(s, ext, dim: qd.template()):
    """Advance per-axis offsets `s` to the next cell of a box with extents `ext`.

    The last axis runs fastest, as in nested loops over the axes.
    """
    s[dim - 1] += 1
    for d in qd.static(range(dim - 1, 0, -1)):
        if s[d] >= ext[d]:
            s[d] = 0
            s[d - 1] += 1
    return s


@qd.func
def _isotropic_cell_integral(
    evaluate_fn: qd.template(),
    quad_points: qd.template(),
    dim: qd.template(),
    a,
    x_cell,
    hsm_phys: qd.f32,
    kernel_prefactor: qd.f32,
    cell_size,
) -> qd.f32:
    """Quadrature of the kernel mass over cell `a` (cell coordinates)."""
    s = 0.0
    for qp in qd.static(quad_points):  # (offset per axis..., weight)
        d0 = (x_cell[0] - (a[0] + qp[0])) * cell_size[0]
        q = 0.0
        if qd.static(dim == 1):
            q = qd.abs(d0) / hsm_phys
        else:
            r2 = d0 * d0
            for d in qd.static(range(1, dim)):
                dd = (x_cell[d] - (a[d] + qp[d])) * cell_size[d]
                r2 = r2 + dd * dd
            q = qd.sqrt(r2) / hsm_phys
        s += qp[dim] * kernel_prefactor * evaluate_fn(q, dim)
    for d in qd.static(range(dim)):
        s = s * cell_size[d]
    return s


@qd.func
def _covariant_cell_integral(
    evaluate_fn: qd.template(),
    quad_points: qd.template(),
    dim: qd.template(),
    a,
    x_cell,
    E,
    ev,
    kernel_prefactor: qd.f32,
    cell_size,
) -> qd.f32:
    """Quadrature of the ellipsoidal kernel mass over cell `a`.

    `E` holds the principal axes as columns, `ev` the semi-axis lengths.
    """
    s = 0.0
    for qp in qd.static(quad_points):  # (offset per axis..., weight)
        dx = qd.Vector.zero(qd.f32, dim)
        for d in qd.static(range(dim)):
            dx[d] = (x_cell[d] - (a[d] + qp[d])) * cell_size[d]
        # separation projected onto each principal axis, in units of that
        # semi-axis: the ellipsoidal q
        q2 = 0.0
        q = 0.0
        for c in qd.static(range(dim)):
            proj = E[0, c] * dx[0]
            for b in qd.static(range(1, dim)):
                proj = proj + E[b, c] * dx[b]
            xi = proj / ev[c]
            if qd.static(dim == 1):
                q = qd.abs(xi)  # kernels expect q >= 0
            elif qd.static(c == 0):
                q2 = xi * xi
            else:
                q2 = q2 + xi * xi
        if qd.static(dim > 1):
            q = qd.sqrt(q2)
        s += qp[dim] * kernel_prefactor * evaluate_fn(q, dim)
    for d in qd.static(range(dim)):
        s = s * cell_size[d]
    return s


@qd.func
def _cell_integral(
    evaluate_fn: qd.template(),
    quad_points: qd.template(),
    dim: qd.template(),
    covariant: qd.template(),
    a,
    x_cell,
    hsm_phys: qd.f32,
    E,
    ev,
    kernel_prefactor: qd.f32,
    cell_size,
) -> qd.f32:
    """Kernel mass over cell `a`, isotropic (`hsm_phys`) or covariant (`E`, `ev`)."""
    result = 0.0
    if qd.static(covariant):
        result = _covariant_cell_integral(
            evaluate_fn, quad_points, dim, a, x_cell, E, ev, kernel_prefactor, cell_size
        )
    else:
        result = _isotropic_cell_integral(
            evaluate_fn, quad_points, dim, a, x_cell, hsm_phys, kernel_prefactor,
            cell_size,
        )
    return result


@qd.func
def _quadrature_total(
    evaluate_fn: qd.template(),
    quad_points: qd.template(),
    dim: qd.template(),
    covariant: qd.template(),
    lo,
    ext,
    x_cell,
    hsm_phys: qd.f32,
    E,
    ev,
    kernel_prefactor: qd.f32,
    cell_size,
) -> qd.f32:
    """Sum the cell integrals over the box `lo + [0, ext)`.

    Nested loops, so that per-axis terms are hoisted out of the inner loops.
    """
    total = 0.0
    if qd.static(dim == 1):
        for i in range(lo[0], lo[0] + ext[0]):
            total += _cell_integral(
                evaluate_fn, quad_points, dim, covariant, qd.Vector([i]), x_cell,
                hsm_phys, E, ev, kernel_prefactor, cell_size,
            )
    elif qd.static(dim == 2):
        for i in range(lo[0], lo[0] + ext[0]):
            for j in range(lo[1], lo[1] + ext[1]):
                total += _cell_integral(
                    evaluate_fn, quad_points, dim, covariant, qd.Vector([i, j]),
                    x_cell, hsm_phys, E, ev, kernel_prefactor, cell_size,
                )
    else:
        for i in range(lo[0], lo[0] + ext[0]):
            for j in range(lo[1], lo[1] + ext[1]):
                for k in range(lo[2], lo[2] + ext[2]):
                    total += _cell_integral(
                        evaluate_fn, quad_points, dim, covariant,
                        qd.Vector([i, j, k]), x_cell, hsm_phys, E, ev,
                        kernel_prefactor, cell_size,
                    )
    return total


@qd.func
def _quadrature_cell_add(
    evaluate_fn: qd.template(),
    quad_points: qd.template(),
    dim: qd.template(),
    covariant: qd.template(),
    periodic: qd.template(),
    a,
    x_cell,
    hsm_phys: qd.f32,
    E,
    ev,
    kernel_prefactor: qd.f32,
    cell_size,
    N,
    wj: qd.f32,
    correction: qd.f32,
    n,
    grid_weights: qd.template(),
    grid_fields: qd.template(),
    quantities: qd.template(),
):
    """Deposit the kernel mass of cell `a`, scaled by wj * correction."""
    integral = _cell_integral(
        evaluate_fn, quad_points, dim, covariant, a, x_cell, hsm_phys, E, ev,
        kernel_prefactor, cell_size,
    )
    cell = qd.Vector.zero(qd.i32, dim)
    for d in qd.static(range(dim)):
        if qd.static(periodic):
            cell[d] = _apply_pbc(a[d], N[d])
        else:
            cell[d] = a[d]
    _grid_add(
        grid_weights, grid_fields, quantities, n, cell, wj * integral * correction, dim
    )


@qd.func
def _quadrature_deposit(
    evaluate_fn: qd.template(),
    quad_points: qd.template(),
    dim: qd.template(),
    covariant: qd.template(),
    periodic: qd.template(),
    lo,
    ext,
    x_cell,
    hsm_phys: qd.f32,
    E,
    ev,
    kernel_prefactor: qd.f32,
    cell_size,
    N,
    wj: qd.f32,
    correction: qd.f32,
    n,
    grid_weights: qd.template(),
    grid_fields: qd.template(),
    quantities: qd.template(),
):
    """Deposit every cell of the box `lo + [0, ext)`, weight wj * integral * correction."""
    if qd.static(dim == 1):
        for i in range(lo[0], lo[0] + ext[0]):
            _quadrature_cell_add(
                evaluate_fn, quad_points, dim, covariant, periodic, qd.Vector([i]),
                x_cell, hsm_phys, E, ev, kernel_prefactor, cell_size, N, wj,
                correction, n, grid_weights, grid_fields, quantities,
            )
    elif qd.static(dim == 2):
        for i in range(lo[0], lo[0] + ext[0]):
            for j in range(lo[1], lo[1] + ext[1]):
                _quadrature_cell_add(
                    evaluate_fn, quad_points, dim, covariant, periodic,
                    qd.Vector([i, j]), x_cell, hsm_phys, E, ev, kernel_prefactor,
                    cell_size, N, wj, correction, n, grid_weights, grid_fields,
                    quantities,
                )
    else:
        for i in range(lo[0], lo[0] + ext[0]):
            for j in range(lo[1], lo[1] + ext[1]):
                for k in range(lo[2], lo[2] + ext[2]):
                    _quadrature_cell_add(
                        evaluate_fn, quad_points, dim, covariant, periodic,
                        qd.Vector([i, j, k]), x_cell, hsm_phys, E, ev,
                        kernel_prefactor, cell_size, N, wj, correction, n,
                        grid_weights, grid_fields, quantities,
                    )


@qd.func
def _deposit_sample_buffer(
    buf,
    i0,
    wj: qd.f32,
    n,
    N,
    periodic: qd.template(),
    B: qd.template(),
    dim: qd.template(),
    grid_weights: qd.template(),
    grid_fields: qd.template(),
    quantities: qd.template(),
):
    """Add a particle's per-cell sample sums `buf` (B**dim cells from `i0`)."""
    if qd.static(dim == 1):
        for a in range(B):
            v = buf[a]
            if v != 0.0:
                ix, ok_x = _cell_index_from_int(i0[0] + a, N[0], periodic)
                if ok_x:
                    _grid_add(
                        grid_weights, grid_fields, quantities, n, qd.Vector([ix]),
                        wj * v, dim,
                    )
    elif qd.static(dim == 2):
        for a in range(B):
            ix, ok_x = _cell_index_from_int(i0[0] + a, N[0], periodic)
            for b in range(B):
                v = buf[a * B + b]
                if v != 0.0:
                    iy, ok_y = _cell_index_from_int(i0[1] + b, N[1], periodic)
                    if ok_x and ok_y:
                        _grid_add(
                            grid_weights, grid_fields, quantities, n,
                            qd.Vector([ix, iy]), wj * v, dim,
                        )
    else:
        for a in range(B):
            ix, ok_x = _cell_index_from_int(i0[0] + a, N[0], periodic)
            for b in range(B):
                iy, ok_y = _cell_index_from_int(i0[1] + b, N[1], periodic)
                for c in range(B):
                    v = buf[(a * B + b) * B + c]
                    if v != 0.0:
                        iz, ok_z = _cell_index_from_int(i0[2] + c, N[2], periodic)
                        if ok_x and ok_y and ok_z:
                            _grid_add(
                                grid_weights, grid_fields, quantities, n,
                                qd.Vector([ix, iy, iz]), wj * v, dim,
                            )


# =============================================================================
# Separable kernel
# =============================================================================
@qd.kernel
def _separable(
    F_1d_fn: qd.template(),
    sigma_fn: qd.template(),
    kernel_support: qd.f32,
    dim: qd.template(),
    periodic: qd.template(),
    positions: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, D)
    quantities: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, F)
    smoothing_lengths: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, D)
    particle_weights: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (N,)
    cell_size_inv: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (D,)
    max_support: qd.f32,
    MAX_STENCIL: qd.template(),
    fields: qd.types.ndarray(dtype=qd.f32),  # (*gridnums, F)
    weights: qd.types.ndarray(dtype=qd.f32),  # gridnums
):
    """Tensor-product kernel, integrated exactly per cell and axis."""
    sig = sigma_fn(dim)
    N = qd.Vector.zero(qd.i32, dim)
    inv = qd.Vector.zero(qd.f32, dim)
    for d in qd.static(range(dim)):
        N[d] = weights.shape[d]
        inv[d] = cell_size_inv[d]

    for n in range(positions.shape[0]):
        wts = qd.Matrix.zero(qd.f32, dim, MAX_STENCIL)
        wj = particle_weights[n]

        h = qd.Vector.zero(qd.f32, dim)
        for d in qd.static(range(dim)):
            h[d] = smoothing_lengths[n, d]
        if qd.static(periodic):  # support at most half the box
            h_max = h[0]
            for d in qd.static(range(1, dim)):
                h_max = qd.max(h_max, h[d])
            if h_max > max_support:
                scale = max_support / h_max
                for d in qd.static(range(dim)):
                    h[d] *= scale

        lo = qd.Vector.zero(qd.i32, dim)
        ext = qd.Vector.zero(qd.i32, dim)
        for d in qd.static(range(dim)):
            x_cell = positions[n, d] * inv[d]
            h_cell = h[d] * inv[d]
            inv_h = 1.0 / h_cell
            support_cell = kernel_support * h_cell
            i_min = qd.cast(qd.floor(x_cell - support_cell), qd.i32)
            i_max = qd.cast(qd.ceil(x_cell + support_cell), qd.i32) - 1
            if qd.static(not periodic):
                i_min = qd.max(i_min, 0)
                i_max = qd.min(i_max, N[d] - 1)
            lo[d] = i_min
            ext[d] = i_max - i_min + 1
            for si in range(ext[d]):
                i = i_min + si
                q_left = (i - x_cell) * inv_h
                q_right = ((i + 1) - x_cell) * inv_h
                integral = _separable_integrate_1d(
                    F_1d_fn, kernel_support, dim, q_left, q_right
                )
                # normalisation on axis 0, particle weight on axis 1
                if qd.static(dim == 1):
                    wts[d, si] = wj * sig * integral
                elif qd.static(d == 0):
                    wts[d, si] = sig * integral
                elif qd.static(d == 1):
                    wts[d, si] = wj * integral
                else:
                    wts[d, si] = integral

        s = qd.Vector.zero(qd.i32, dim)
        for _ in range(_box_count(ext, dim)):
            cell = qd.Vector.zero(qd.i32, dim)
            w = wts[0, s[0]]
            for d in qd.static(range(dim)):
                if qd.static(d > 0):
                    w = w * wts[d, s[d]]
                c = lo[d] + s[d]
                if qd.static(periodic):
                    c = _apply_pbc(c, N[d])
                cell[d] = c
            _grid_add(weights, fields, quantities, n, cell, w, dim)
            s = _next_offset(s, ext, dim)


# =============================================================================
# Isotropic kernel
# =============================================================================
@qd.kernel
def _isotropic(
    evaluate_fn: qd.template(),
    sigma_fn: qd.template(),
    kernel_support: qd.f32,
    dim: qd.template(),
    quad_points: qd.template(),
    periodic: qd.template(),
    eta_crit: qd.f32,
    B: qd.template(),
    positions: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, D)
    quantities: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, F)
    smoothing_lengths: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (N,)
    particle_weights: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (N,)
    sample_coords: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (S, D)
    sample_integrals: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (S,)
    num_samples: qd.i32,
    cell_size: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (D,)
    cell_size_inv: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (D,)
    max_support: qd.f32,
    fields: qd.types.ndarray(dtype=qd.f32),  # (*gridnums, F)
    weights: qd.types.ndarray(dtype=qd.f32),  # gridnums
):
    """Radial kernel W(|x - x_n| / h_n).

    Particles with kernel diameter eta < eta_crit cells are deposited with the
    precomputed sample table (through a local B**dim cell buffer if they fit
    into it), larger ones by per-cell quadrature renormalised to the
    particle's weight.
    """
    sig = sigma_fn(dim)
    N = qd.Vector.zero(qd.i32, dim)
    cs = qd.Vector.zero(qd.f32, dim)
    inv = qd.Vector.zero(qd.f32, dim)
    for d in qd.static(range(dim)):
        N[d] = weights.shape[d]
        cs[d] = cell_size[d]
        inv[d] = cell_size_inv[d]
    no_axes = qd.Matrix.zero(qd.f32, dim, dim)  # unused by isotropic cells
    no_ev = qd.Vector.zero(qd.f32, dim)

    for n in range(positions.shape[0]):
        hsm_phys = smoothing_lengths[n]
        if qd.static(periodic):
            if hsm_phys > max_support:
                hsm_phys = max_support

        h_pow = hsm_phys
        for d in qd.static(range(1, dim)):
            h_pow = h_pow * hsm_phys
        kernel_prefactor = sig / h_pow
        support_phys = kernel_support * hsm_phys

        x_phys = qd.Vector.zero(qd.f32, dim)
        x_cell = qd.Vector.zero(qd.f32, dim)
        support_cell = qd.Vector.zero(qd.f32, dim)
        i0 = qd.Vector.zero(qd.i32, dim)
        fits = True
        eta = 0.0
        for d in qd.static(range(dim)):
            x_phys[d] = positions[n, d]
            x_cell[d] = x_phys[d] * inv[d]
            support_cell[d] = support_phys * inv[d]
            i0[d] = qd.floor(x_cell[d] - support_cell[d], int)
            fits = fits and (qd.floor(x_cell[d] + support_cell[d], int) - i0[d] < B)
            if qd.static(d == 0):
                eta = support_cell[d]
            else:
                eta = qd.min(eta, support_cell[d])
        eta = 2.0 * eta  # kernel diameter in cells, along the narrowest axis
        wj = particle_weights[n]

        if eta < eta_crit and fits:
            # samples summed per cell in a local buffer: one atomic per cell
            buf = qd.Vector.zero(qd.f32, qd.static(B**dim))
            for s in range(num_samples):
                idx = 0
                inside = True
                for d in qd.static(range(dim)):
                    x_sample = x_phys[d] + sample_coords[s, d] * hsm_phys
                    li = qd.floor(x_sample * inv[d], int) - i0[d]
                    inside = inside and (0 <= li < B)
                    idx = idx * B + li
                if inside:
                    buf[idx] += sample_integrals[s]
            _deposit_sample_buffer(
                buf, i0, wj, n, N, periodic, B, dim, weights, fields, quantities
            )

        elif eta < eta_crit:
            # too large for the buffer: one atomic per sample
            for s in range(num_samples):
                cell = qd.Vector.zero(qd.i32, dim)
                valid = True
                for d in qd.static(range(dim)):
                    x_sample = x_phys[d] + sample_coords[s, d] * hsm_phys
                    c, ok = _cell_index_from_pos(x_sample, inv[d], N[d], periodic)
                    cell[d] = c
                    valid = valid and ok
                if valid:
                    integral = sample_integrals[s]
                    if integral != 0.0:
                        _grid_add(
                            weights, fields, quantities, n, cell, wj * integral, dim
                        )

        else:
            # per-cell quadrature; cell ceil(x + support) lies outside the
            # support, so the stencil stops one cell before it
            lo = qd.Vector.zero(qd.i32, dim)
            ext = qd.Vector.zero(qd.i32, dim)
            for d in qd.static(range(dim)):
                i_min = i0[d]
                i_max = qd.ceil(x_cell[d] + support_cell[d], int) - 1
                if qd.static(not periodic):
                    i_min = qd.max(i_min, 0)
                    i_max = qd.min(i_max, N[d] - 1)
                lo[d] = i_min
                ext[d] = i_max - i_min + 1
            # pass 1: total over the cells, to renormalise to the particle weight
            total_integral = _quadrature_total(
                evaluate_fn, quad_points, dim, False, lo, ext, x_cell, hsm_phys,
                no_axes, no_ev, kernel_prefactor, cs,
            )
            if total_integral > 0.0:
                _quadrature_deposit(
                    evaluate_fn, quad_points, dim, False, periodic, lo, ext, x_cell,
                    hsm_phys, no_axes, no_ev, kernel_prefactor, cs, N, wj,
                    1.0 / total_integral, n, weights, fields, quantities,
                )


# =============================================================================
# Covariant kernel
# =============================================================================
@qd.kernel
def _covariant(
    evaluate_fn: qd.template(),
    sigma_fn: qd.template(),
    kernel_support: qd.f32,
    dim: qd.template(),
    quad_points: qd.template(),
    periodic: qd.template(),
    eta_crit: qd.f32,
    B: qd.template(),
    positions: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, D)
    quantities: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, F)
    hmat_eigvecs: qd.types.ndarray(dtype=qd.f32, ndim=3),  # (N, D, D), columns
    hmat_eigvals: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, D)
    particle_weights: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (N,)
    sample_coords: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (S, D)
    sample_integrals: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (S,)
    num_samples: qd.i32,
    cell_size: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (D,)
    cell_size_inv: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (D,)
    max_support: qd.f32,
    fields: qd.types.ndarray(dtype=qd.f32),  # (*gridnums, F)
    weights: qd.types.ndarray(dtype=qd.f32),  # gridnums
):
    """Ellipsoidal kernel W(|diag(ev)^-1 E^T (x - x_n)|).

    Same branches as :func:`_isotropic`.

    The quadrature total is computed before the branch decision: a strongly
    elongated ellipsoid can span several cells along every axis (eta >=
    eta_crit) while being thinner than a cell across its minor axis, so that
    no quadrature node falls inside it. Such particles (and degenerate ones,
    with a zero semi-axis) are deposited with the sample table instead of
    being dropped.
    """
    sig = sigma_fn(dim)
    N = qd.Vector.zero(qd.i32, dim)
    cs = qd.Vector.zero(qd.f32, dim)
    inv = qd.Vector.zero(qd.f32, dim)
    for d in qd.static(range(dim)):
        N[d] = weights.shape[d]
        cs[d] = cell_size[d]
        inv[d] = cell_size_inv[d]

    for n in range(positions.shape[0]):
        E = qd.Matrix.zero(qd.f32, dim, dim)  # column c: principal axis c
        ev = qd.Vector.zero(qd.f32, dim)
        for r in qd.static(range(dim)):
            ev[r] = hmat_eigvals[n, r]
            for c in qd.static(range(dim)):
                E[r, c] = hmat_eigvecs[n, r, c]

        if qd.static(periodic):  # support at most half the box
            if qd.static(dim == 1):
                if ev[0] > max_support:
                    ev[0] = max_support
            else:
                max_eval = ev[0]
                for c in qd.static(range(1, dim)):
                    max_eval = qd.max(max_eval, ev[c])
                if max_eval > max_support:
                    scale = max_support / max_eval
                    for c in qd.static(range(dim)):
                        ev[c] *= scale

        degenerate = False
        for c in qd.static(range(dim)):
            degenerate = degenerate or ev[c] <= 0.0

        x_phys = qd.Vector.zero(qd.f32, dim)
        x_cell = qd.Vector.zero(qd.f32, dim)
        support_cell = qd.Vector.zero(qd.f32, dim)
        i0 = qd.Vector.zero(qd.i32, dim)
        lo = qd.Vector.zero(qd.i32, dim)
        ext = qd.Vector.zero(qd.i32, dim)
        fits = True
        eta = 0.0
        for d in qd.static(range(dim)):
            # half-extent of the support ellipsoid along axis d, in cells of
            # axis d: sqrt(sum_c (E[d, c] ev[c])^2) / cell_size[d]
            if qd.static(dim == 1):
                support_cell[d] = kernel_support * qd.abs(ev[0] * inv[0])
            else:
                s2 = (E[d, 0] * (ev[0] * inv[d])) ** 2
                for c in qd.static(range(1, dim)):
                    s2 = s2 + (E[d, c] * (ev[c] * inv[d])) ** 2
                support_cell[d] = kernel_support * qd.sqrt(s2)
            if qd.static(d == 0):
                eta = support_cell[d]
            else:
                eta = qd.min(eta, support_cell[d])
            x_phys[d] = positions[n, d]
            x_cell[d] = x_phys[d] * inv[d]
            i0[d] = qd.floor(x_cell[d] - support_cell[d], int)
            fits = fits and (qd.floor(x_cell[d] + support_cell[d], int) - i0[d] < B)
            # cell ceil(x + support) lies outside the support
            i_min = i0[d]
            i_max = qd.ceil(x_cell[d] + support_cell[d], int) - 1
            if qd.static(not periodic):
                i_min = qd.max(i_min, 0)
                i_max = qd.min(i_max, N[d] - 1)
            lo[d] = i_min
            ext[d] = i_max - i_min + 1
        eta = 2.0 * eta  # kernel diameter in cells, along the narrowest axis
        wj = particle_weights[n]
        use_samples = eta < eta_crit or degenerate

        kernel_prefactor = 0.0
        total_integral = 0.0
        if not use_samples:
            det_h = ev[0]
            for c in qd.static(range(1, dim)):
                det_h = det_h * ev[c]
            kernel_prefactor = sig / det_h
            total_integral = _quadrature_total(
                evaluate_fn, quad_points, dim, True, lo, ext, x_cell, 0.0, E, ev,
                kernel_prefactor, cs,
            )
            if total_integral <= 0.0:
                use_samples = True

        if use_samples and fits:
            # samples summed per cell in a local buffer: one atomic per cell
            buf = qd.Vector.zero(qd.f32, qd.static(B**dim))
            for s in range(num_samples):
                idx = 0
                inside = True
                for d in qd.static(range(dim)):
                    # unit-ball sample mapped into the ellipsoid: E diag(ev) c
                    x_sample = x_phys[d]
                    for c in qd.static(range(dim)):
                        x_sample = x_sample + E[d, c] * (ev[c] * sample_coords[s, c])
                    li = qd.floor(x_sample * inv[d], int) - i0[d]
                    inside = inside and (0 <= li < B)
                    idx = idx * B + li
                if inside:
                    buf[idx] += sample_integrals[s]
            _deposit_sample_buffer(
                buf, i0, wj, n, N, periodic, B, dim, weights, fields, quantities
            )

        elif use_samples:
            # too large for the buffer: one atomic per sample
            for s in range(num_samples):
                cell = qd.Vector.zero(qd.i32, dim)
                valid = True
                for d in qd.static(range(dim)):
                    x_sample = x_phys[d]
                    for c in qd.static(range(dim)):
                        x_sample = x_sample + E[d, c] * (ev[c] * sample_coords[s, c])
                    cd, ok = _cell_index_from_pos(x_sample, inv[d], N[d], periodic)
                    cell[d] = cd
                    valid = valid and ok
                if valid:
                    integral = sample_integrals[s]
                    if integral != 0.0:
                        _grid_add(
                            weights, fields, quantities, n, cell, wj * integral, dim
                        )

        else:
            # per-cell quadrature, renormalised with the pass-1 total
            _quadrature_deposit(
                evaluate_fn, quad_points, dim, True, periodic, lo, ext, x_cell, 0.0,
                E, ev, kernel_prefactor, cs, N, wj, 1.0 / total_integral, n,
                weights, fields, quantities,
            )


# =============================================================================
# Python wrappers
# =============================================================================
_ADAPTIVE_KERNELS = {
    "separable": _separable,
    "isotropic": _isotropic,
    "covariant": _covariant,
}


@dataclass(slots=True)
class KernelPreparation:
    """A prepared adaptive deposition: call ``kernel(*args)``, read the grids."""

    kernel: Any
    args: tuple
    grid_fields: np.ndarray  # (*gridnums, F), filled in place
    grid_weights: np.ndarray  # gridnums, filled in place
    sample_buffer: int | None = None  # isotropic / covariant only


def _prepare_separable(
    particle_positions,
    particle_fields,
    particle_weights,
    boxsizes,
    gridnums,
    periodic,
    kernel_name,
    particle_hsml,  # (N, D) -- per-axis smoothing length
):
    DIM = particle_positions.shape[1]
    particle_hsml = as_f32(particle_hsml)

    gridnums = tuple(int(n) for n in gridnums)
    boxsizes = tuple(float(b) for b in boxsizes)

    kspec = create_separable_kernel(kernel_name)
    support = kspec.support

    cell_size_invs = tuple(gridnums[d] / boxsizes[d] for d in range(DIM))
    max_support = min(0.5 * boxsizes[d] / support for d in range(DIM))

    # widest per-axis stencil; an empty particle set (e.g. an MPI rank
    # without particles inside `extent`) still needs a valid template value
    h_max = particle_hsml.max(axis=0) if particle_hsml.shape[0] else np.zeros(DIM)
    max_stencil = _round_up_stencil(
        max(
            int(np.ceil(2 * support * h_max[d] * cell_size_invs[d])) + 2
            for d in range(DIM)
        )
    )

    F = particle_fields.shape[1]
    grid_fields = np.zeros((*gridnums, F), dtype=np.float32)
    grid_weights = np.zeros(gridnums, dtype=np.float32)

    args = (
        kspec.F_1d,
        kspec.sigma,
        support,
        DIM,
        bool(periodic),
        particle_positions,
        particle_fields,
        particle_hsml,
        particle_weights,
        np.array(cell_size_invs, dtype=np.float32),
        max_support,
        max_stencil,
        grid_fields,
        grid_weights,
    )
    return KernelPreparation(_separable, args, grid_fields, grid_weights)


_QUADRATURE_TABLES = {
    1: QUADRATURE_POINTS_1D,
    2: QUADRATURE_POINTS_2D,
    3: QUADRATURE_POINTS_3D,
}


def _prepare_radial(
    structure,  # "isotropic" or "covariant"
    smoothing_args,  # (hsml,) or (eigvecs (N, D, D), eigvals (N, D))
    particle_positions,
    particle_fields,
    particle_weights,
    boxsizes,
    gridnums,
    periodic,
    kernel_name,
    integration_method,
    eta_crit,
):
    """Prepare an isotropic or covariant deposition.

    Both use the same kernel family, cell quadrature and sample table; only
    the smoothing arrays differ.
    """
    DIM = particle_positions.shape[1]
    gridnums = tuple(int(n) for n in gridnums)
    boxsizes = tuple(float(b) for b in boxsizes)

    kspec = create_spherical_kernel(kernel_name)
    support = kspec.support

    quad_table = _QUADRATURE_TABLES[DIM]
    if integration_method not in quad_table:
        raise ValueError(
            f"Unknown integration method: {integration_method!r}. "
            f"Available: {sorted(quad_table.keys())}"
        )
    quad_points = quad_table[integration_method]

    grid = build_kernel_sample_grid(kernel_name, DIM, eta_crit)
    sample_buffer = _sample_buffer_size(eta_crit, DIM)

    cell_sizes = tuple(boxsizes[d] / gridnums[d] for d in range(DIM))
    cell_size_invs = tuple(1.0 / c for c in cell_sizes)
    max_support = min(0.5 * boxsizes[d] / support for d in range(DIM))

    F = particle_fields.shape[1]
    grid_fields = np.zeros((*gridnums, F), dtype=np.float32)
    grid_weights = np.zeros(gridnums, dtype=np.float32)

    args = (
        kspec.evaluate,
        kspec.sigma,
        support,
        DIM,
        quad_points,
        bool(periodic),
        float(eta_crit),
        sample_buffer,
        particle_positions,
        particle_fields,
        *(as_f32(a) for a in smoothing_args),
        particle_weights,
        grid["coords"],
        grid["integrals"],
        grid["count"],
        np.array(cell_sizes, dtype=np.float32),
        np.array(cell_size_invs, dtype=np.float32),
        max_support,
        grid_fields,
        grid_weights,
    )
    return KernelPreparation(
        _ADAPTIVE_KERNELS[structure], args, grid_fields, grid_weights,
        sample_buffer=int(sample_buffer),
    )


# Which extra kwargs each adaptive method needs, purely for the
# self-documenting validation error below -- not used for dispatch.
_REQUIRED_KWARGS = {
    "separable": ("particle_hsml",),
    "isotropic": ("particle_hsml",),
    "covariant": ("particle_hmat_eigvecs", "particle_hmat_eigvals"),
}


def deposit(
    particle_positions,
    particle_fields,
    particle_weights,
    boxsizes,
    gridnums,
    *,
    adaptive,
    kernel_name,
    structure=None,
    periodic,
    # --- required by adaptive separable (per-axis) or isotropic (scalar) ---
    particle_hsml=None,  # (N, D) for separable, (N,) for isotropic
    # --- required by adaptive covariant ---
    particle_hmat_eigvecs=None,  # (N, D, D)
    particle_hmat_eigvals=None,  # (N, D)
    # --- required by adaptive isotropic / covariant deposition ---
    integration_method="midpoint",
    eta_crit=4.0,
):
    """Deposit particle data onto a grid covering ``[0, boxsizes)``.

    Parameters
    ----------
    particle_positions, particle_fields, particle_weights : np.ndarray
        Positions (N, D), fields (N, F) and weights (N,).
    boxsizes, gridnums : array_like
        Grid extent and number of cells per axis, shape (D,).
    adaptive : bool
        Fixed-grid stencil (False) or per-particle smoothing (True).
    kernel_name : str
        A fixed-grid stencil (``"ngp"``, ``"cic"``, ``"tsc"``,
        ``"bspline4"``, ``"bspline5"``, ``"bspline6"``) when
        ``adaptive=False``; otherwise a kernel of the structure's family
        (separable: ``"tophat"``, ``"tsc"``, ``"gaussian"``; isotropic and
        covariant: any radial kernel).
    structure : {"separable", "isotropic", "covariant"}, optional
        Adaptive structure; must be None when ``adaptive=False``.
    periodic : bool
        Wrap the grid periodically; otherwise contributions outside it are
        dropped.
    particle_hsml : np.ndarray, optional
        Smoothing lengths, (N, D) for separable, (N,) for isotropic.
    particle_hmat_eigvecs, particle_hmat_eigvals : np.ndarray, optional
        Covariant only: principal axes as columns (N, D, D) and semi-axis
        lengths (N, D).
    integration_method : {"midpoint", "trapezoidal", "simpson"}
        Cell quadrature rule (isotropic and covariant).
    eta_crit : float
        Kernel diameter in cells below which the sample table is used instead
        of cell quadrature (isotropic and covariant).

    Returns
    -------
    grid_fields : ndarray, shape (F, *gridnums)
    grid_weights : ndarray, shape (*gridnums,)

    """
    particle_positions = as_f32(particle_positions)
    particle_fields = as_f32(particle_fields)
    particle_weights = as_f32(particle_weights)
    boxsizes = as_f32(boxsizes)
    dim = particle_positions.shape[-1]
    check_index_limits(
        particle_fields=particle_fields,
        grid_fields=(*gridnums, particle_fields.shape[1]),
    )

    if not adaptive:
        check_fixed_grid_kernel(kernel_name, structure)
        key = (kernel_name, dim)
        try:
            kernel = _FIXED_GRID_KERNELS[key]
        except KeyError:
            raise ValueError(
                f"Unsupported combination: kernel_name={kernel_name!r}, dim={dim}."
            ) from None

        F = particle_fields.shape[1]
        grid_fields = np.zeros(tuple(gridnums) + (F,), dtype=np.float32)
        grid_weights = np.zeros(tuple(gridnums), dtype=np.float32)

        kernel(
            bool(periodic),
            particle_positions,
            particle_fields,
            particle_weights,
            boxsizes,
            grid_fields,
            grid_weights,
        )
        return np.moveaxis(grid_fields, -1, 0), grid_weights

    if structure not in ADAPTIVE_STRUCTURES:
        raise ValueError(
            f"structure={structure!r} is not available for adaptive deposition. "
            f"Available: {list(ADAPTIVE_STRUCTURES)}"
        )

    provided_kwargs = {
        "particle_hsml": particle_hsml,
        "particle_hmat_eigvecs": particle_hmat_eigvecs,
        "particle_hmat_eigvals": particle_hmat_eigvals,
    }
    missing = [
        name for name in _REQUIRED_KWARGS[structure] if provided_kwargs[name] is None
    ]
    if missing:
        raise ValueError(
            f"structure={structure!r} requires: {', '.join(missing)} (got None)."
        )

    common = dict(
        particle_positions=particle_positions,
        particle_fields=particle_fields,
        particle_weights=particle_weights,
        boxsizes=boxsizes,
        gridnums=gridnums,
        periodic=periodic,
        kernel_name=kernel_name,
    )

    if structure == "separable":
        prep = _prepare_separable(**common, particle_hsml=particle_hsml)
    else:
        smoothing_args = (
            (particle_hsml,)
            if structure == "isotropic"
            else (particle_hmat_eigvecs, particle_hmat_eigvals)
        )
        prep = _prepare_radial(
            structure,
            smoothing_args,
            **common,
            integration_method=integration_method,
            eta_crit=eta_crit,
        )

    prep.kernel(*prep.args)
    return np.moveaxis(prep.grid_fields, -1, 0), prep.grid_weights
