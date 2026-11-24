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
def _brick_index(idx: qd.i32, gridnum: qd.i32, offset: qd.i32, periodic: qd.template()):
    """Return ``(index into the output grid, valid)`` of a global cell index.

    `idx` is the unwrapped global cell index along one axis; the output grid
    (the full grid, or a brick) starts at global cell `offset`. Periodic
    indices are taken modulo the global grid size relative to `offset` (a
    brick may wrap around the axis); non-periodic cells outside the global
    grid are invalid.
    """
    valid = True
    if qd.static(periodic):
        idx = _apply_pbc(idx - offset, gridnum)
    else:
        if idx < 0 or idx >= gridnum:
            valid = False
        idx -= offset
    return idx, valid


@qd.func
def _load_grid_spec(grid_spec: qd.template(), dim: qd.template()):
    """Global grid size and brick offset per axis from `grid_spec` (2, D)."""
    N = qd.Vector.zero(qd.i32, dim)
    off = qd.Vector.zero(qd.i32, dim)
    for d in qd.static(range(dim)):
        N[d] = grid_spec[0, d]
        off[d] = grid_spec[1, d]
    return N, off


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
    return ix, qd.Vector([0.5 * (0.5 - dx) ** 2, 0.75 - dx * dx, 0.5 * (0.5 + dx) ** 2])


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
            ((3.0 - dx) ** 5 - 6.0 * (2.0 - dx) ** 5 + 15.0 * (1.0 - dx) ** 5) / 120.0,
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
        brick: qd.template(),
        particle_positions: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, D)
        particle_fields: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, F)
        particle_weights: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (N,)
        boxsizes: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (D,)
        grid_spec: qd.types.ndarray(dtype=qd.i32, ndim=2),  # (2, D)
        dropped: qd.types.ndarray(dtype=qd.i32, ndim=1),  # (1,)
        grid_fields: qd.types.ndarray(dtype=qd.f32, ndim=dim + 1),  # (*brick, F)
        grid_weights: qd.types.ndarray(dtype=qd.f32, ndim=dim),  # brick
    ):
        N, off = _load_grid_spec(grid_spec, dim)
        inv_dx = qd.Vector.zero(qd.f32, dim)
        for d in qd.static(range(dim)):
            inv_dx[d] = N[d] / boxsizes[d]

        for p in range(particle_positions.shape[0]):
            idx = qd.Matrix.zero(qd.i32, dim, K)
            wgt = qd.Matrix.zero(qd.f32, dim, K)
            ok = qd.Matrix.zero(qd.i32, dim, K)
            for d in qd.static(range(dim)):
                base, w1d = stencil_fn(particle_positions[p, d] * inv_dx[d])
                for o in qd.static(range(K)):
                    i, valid = _brick_index(base + o - SHIFT, N[d], off[d], periodic)
                    ok[d, o] = valid
                    idx[d, o] = i
                    wgt[d, o] = w1d[o]

            w_particle = particle_weights[p]

            if qd.static(dim == 1):
                for ox in qd.static(range(K)):
                    if ok[0, ox]:
                        _grid_add(
                            grid_weights,
                            grid_fields,
                            particle_fields,
                            dropped,
                            brick,
                            p,
                            qd.Vector([idx[0, ox]]),
                            wgt[0, ox] * w_particle,
                            dim,
                        )
            elif qd.static(dim == 2):
                for ox in qd.static(range(K)):
                    for oy in qd.static(range(K)):
                        if ok[0, ox] and ok[1, oy]:
                            _grid_add(
                                grid_weights,
                                grid_fields,
                                particle_fields,
                                dropped,
                                brick,
                                p,
                                qd.Vector([idx[0, ox], idx[1, oy]]),
                                wgt[0, ox] * wgt[1, oy] * w_particle,
                                dim,
                            )
            else:
                for ox in qd.static(range(K)):
                    for oy in qd.static(range(K)):
                        for oz in qd.static(range(K)):
                            if ok[0, ox] and ok[1, oy] and ok[2, oz]:
                                _grid_add(
                                    grid_weights,
                                    grid_fields,
                                    particle_fields,
                                    dropped,
                                    brick,
                                    p,
                                    qd.Vector([idx[0, ox], idx[1, oy], idx[2, oz]]),
                                    wgt[0, ox] * wgt[1, oy] * wgt[2, oz] * w_particle,
                                    dim,
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
    dropped: qd.template(),
    brick: qd.template(),
    n,
    cell,
    w,
    dim: qd.template(),
):
    """Add weight `w` of particle `n` (and its fields times `w`) to grid cell `cell`.

    With `brick` (compile-time), a cell outside the output grid is not
    written but counted in `dropped`: the brick bounds computed on the host
    must contain every cell a particle touches, so a nonzero count is an
    internal error. A full grid skips the check (callers pass valid cells).
    """
    inside = True
    if qd.static(brick):
        for d in qd.static(range(dim)):
            inside = inside and 0 <= cell[d] < grid_weights.shape[d]
    if not inside:
        qd.atomic_add(dropped[0], 1)
    else:
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
            evaluate_fn,
            quad_points,
            dim,
            a,
            x_cell,
            hsm_phys,
            kernel_prefactor,
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
                evaluate_fn,
                quad_points,
                dim,
                covariant,
                qd.Vector([i]),
                x_cell,
                hsm_phys,
                E,
                ev,
                kernel_prefactor,
                cell_size,
            )
    elif qd.static(dim == 2):
        for i in range(lo[0], lo[0] + ext[0]):
            for j in range(lo[1], lo[1] + ext[1]):
                total += _cell_integral(
                    evaluate_fn,
                    quad_points,
                    dim,
                    covariant,
                    qd.Vector([i, j]),
                    x_cell,
                    hsm_phys,
                    E,
                    ev,
                    kernel_prefactor,
                    cell_size,
                )
    else:
        for i in range(lo[0], lo[0] + ext[0]):
            for j in range(lo[1], lo[1] + ext[1]):
                for k in range(lo[2], lo[2] + ext[2]):
                    total += _cell_integral(
                        evaluate_fn,
                        quad_points,
                        dim,
                        covariant,
                        qd.Vector([i, j, k]),
                        x_cell,
                        hsm_phys,
                        E,
                        ev,
                        kernel_prefactor,
                        cell_size,
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
    off,
    wj: qd.f32,
    correction: qd.f32,
    n,
    grid_weights: qd.template(),
    grid_fields: qd.template(),
    quantities: qd.template(),
    dropped: qd.template(),
    brick: qd.template(),
):
    """Deposit the kernel mass of cell `a`, scaled by wj * correction."""
    integral = _cell_integral(
        evaluate_fn,
        quad_points,
        dim,
        covariant,
        a,
        x_cell,
        hsm_phys,
        E,
        ev,
        kernel_prefactor,
        cell_size,
    )
    cell = qd.Vector.zero(qd.i32, dim)
    for d in qd.static(range(dim)):
        # a lies inside the (clipped) grid: always valid
        c, _ok = _brick_index(a[d], N[d], off[d], periodic)
        cell[d] = c
    _grid_add(
        grid_weights,
        grid_fields,
        quantities,
        dropped,
        brick,
        n,
        cell,
        wj * integral * correction,
        dim,
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
    off,
    wj: qd.f32,
    correction: qd.f32,
    n,
    grid_weights: qd.template(),
    grid_fields: qd.template(),
    quantities: qd.template(),
    dropped: qd.template(),
    brick: qd.template(),
):
    """Deposit every cell of the box `lo + [0, ext)`, weight wj * integral * correction."""
    if qd.static(dim == 1):
        for i in range(lo[0], lo[0] + ext[0]):
            _quadrature_cell_add(
                evaluate_fn,
                quad_points,
                dim,
                covariant,
                periodic,
                qd.Vector([i]),
                x_cell,
                hsm_phys,
                E,
                ev,
                kernel_prefactor,
                cell_size,
                N,
                off,
                wj,
                correction,
                n,
                grid_weights,
                grid_fields,
                quantities,
                dropped,
                brick,
            )
    elif qd.static(dim == 2):
        for i in range(lo[0], lo[0] + ext[0]):
            for j in range(lo[1], lo[1] + ext[1]):
                _quadrature_cell_add(
                    evaluate_fn,
                    quad_points,
                    dim,
                    covariant,
                    periodic,
                    qd.Vector([i, j]),
                    x_cell,
                    hsm_phys,
                    E,
                    ev,
                    kernel_prefactor,
                    cell_size,
                    N,
                    off,
                    wj,
                    correction,
                    n,
                    grid_weights,
                    grid_fields,
                    quantities,
                    dropped,
                    brick,
                )
    else:
        for i in range(lo[0], lo[0] + ext[0]):
            for j in range(lo[1], lo[1] + ext[1]):
                for k in range(lo[2], lo[2] + ext[2]):
                    _quadrature_cell_add(
                        evaluate_fn,
                        quad_points,
                        dim,
                        covariant,
                        periodic,
                        qd.Vector([i, j, k]),
                        x_cell,
                        hsm_phys,
                        E,
                        ev,
                        kernel_prefactor,
                        cell_size,
                        N,
                        off,
                        wj,
                        correction,
                        n,
                        grid_weights,
                        grid_fields,
                        quantities,
                        dropped,
                        brick,
                    )


@qd.func
def _deposit_sample_buffer(
    buf,
    i0,
    wj: qd.f32,
    n,
    N,
    off,
    periodic: qd.template(),
    B: qd.template(),
    dim: qd.template(),
    grid_weights: qd.template(),
    grid_fields: qd.template(),
    quantities: qd.template(),
    dropped: qd.template(),
    brick: qd.template(),
):
    """Add a particle's per-cell sample sums `buf` (B**dim cells from `i0`)."""
    if qd.static(dim == 1):
        for a in range(B):
            v = buf[a]
            if v != 0.0:
                ix, ok_x = _brick_index(i0[0] + a, N[0], off[0], periodic)
                if ok_x:
                    _grid_add(
                        grid_weights,
                        grid_fields,
                        quantities,
                        dropped,
                        brick,
                        n,
                        qd.Vector([ix]),
                        wj * v,
                        dim,
                    )
    elif qd.static(dim == 2):
        for a in range(B):
            ix, ok_x = _brick_index(i0[0] + a, N[0], off[0], periodic)
            for b in range(B):
                v = buf[a * B + b]
                if v != 0.0:
                    iy, ok_y = _brick_index(i0[1] + b, N[1], off[1], periodic)
                    if ok_x and ok_y:
                        _grid_add(
                            grid_weights,
                            grid_fields,
                            quantities,
                            dropped,
                            brick,
                            n,
                            qd.Vector([ix, iy]),
                            wj * v,
                            dim,
                        )
    else:
        for a in range(B):
            ix, ok_x = _brick_index(i0[0] + a, N[0], off[0], periodic)
            for b in range(B):
                iy, ok_y = _brick_index(i0[1] + b, N[1], off[1], periodic)
                for c in range(B):
                    v = buf[(a * B + b) * B + c]
                    if v != 0.0:
                        iz, ok_z = _brick_index(i0[2] + c, N[2], off[2], periodic)
                        if ok_x and ok_y and ok_z:
                            _grid_add(
                                grid_weights,
                                grid_fields,
                                quantities,
                                dropped,
                                brick,
                                n,
                                qd.Vector([ix, iy, iz]),
                                wj * v,
                                dim,
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
    brick: qd.template(),
    positions: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, D)
    quantities: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, F)
    smoothing_lengths: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, D)
    particle_weights: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (N,)
    cell_size_inv: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (D,)
    max_support: qd.f32,
    MAX_STENCIL: qd.template(),
    grid_spec: qd.types.ndarray(dtype=qd.i32, ndim=2),  # (2, D)
    dropped: qd.types.ndarray(dtype=qd.i32, ndim=1),  # (1,)
    fields: qd.types.ndarray(dtype=qd.f32),  # (*brick, F)
    weights: qd.types.ndarray(dtype=qd.f32),  # brick
):
    """Tensor-product kernel, integrated exactly per cell and axis."""
    sig = sigma_fn(dim)
    N, off = _load_grid_spec(grid_spec, dim)
    inv = qd.Vector.zero(qd.f32, dim)
    for d in qd.static(range(dim)):
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
                # lo + s lies inside the (clipped) grid: always valid
                c, _ok = _brick_index(lo[d] + s[d], N[d], off[d], periodic)
                cell[d] = c
            _grid_add(weights, fields, quantities, dropped, brick, n, cell, w, dim)
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
    brick: qd.template(),
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
    grid_spec: qd.types.ndarray(dtype=qd.i32, ndim=2),  # (2, D)
    dropped: qd.types.ndarray(dtype=qd.i32, ndim=1),  # (1,)
    fields: qd.types.ndarray(dtype=qd.f32),  # (*brick, F)
    weights: qd.types.ndarray(dtype=qd.f32),  # brick
):
    """Radial kernel W(|x - x_n| / h_n).

    Particles with kernel diameter eta < eta_crit cells are deposited with the
    precomputed sample table (through a local B**dim cell buffer if they fit
    into it), larger ones by per-cell quadrature renormalised to the
    particle's weight.
    """
    sig = sigma_fn(dim)
    N, off = _load_grid_spec(grid_spec, dim)
    cs = qd.Vector.zero(qd.f32, dim)
    inv = qd.Vector.zero(qd.f32, dim)
    for d in qd.static(range(dim)):
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
                buf,
                i0,
                wj,
                n,
                N,
                off,
                periodic,
                B,
                dim,
                weights,
                fields,
                quantities,
                dropped,
                brick,
            )

        elif eta < eta_crit:
            # too large for the buffer: one atomic per sample
            for s in range(num_samples):
                cell = qd.Vector.zero(qd.i32, dim)
                valid = True
                for d in qd.static(range(dim)):
                    x_sample = x_phys[d] + sample_coords[s, d] * hsm_phys
                    c, ok = _brick_index(
                        qd.floor(x_sample * inv[d], int),
                        N[d],
                        off[d],
                        periodic,
                    )
                    cell[d] = c
                    valid = valid and ok
                if valid:
                    integral = sample_integrals[s]
                    if integral != 0.0:
                        _grid_add(
                            weights,
                            fields,
                            quantities,
                            dropped,
                            brick,
                            n,
                            cell,
                            wj * integral,
                            dim,
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
                evaluate_fn,
                quad_points,
                dim,
                False,
                lo,
                ext,
                x_cell,
                hsm_phys,
                no_axes,
                no_ev,
                kernel_prefactor,
                cs,
            )
            if total_integral > 0.0:
                _quadrature_deposit(
                    evaluate_fn,
                    quad_points,
                    dim,
                    False,
                    periodic,
                    lo,
                    ext,
                    x_cell,
                    hsm_phys,
                    no_axes,
                    no_ev,
                    kernel_prefactor,
                    cs,
                    N,
                    off,
                    wj,
                    1.0 / total_integral,
                    n,
                    weights,
                    fields,
                    quantities,
                    dropped,
                    brick,
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
    brick: qd.template(),
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
    grid_spec: qd.types.ndarray(dtype=qd.i32, ndim=2),  # (2, D)
    dropped: qd.types.ndarray(dtype=qd.i32, ndim=1),  # (1,)
    fields: qd.types.ndarray(dtype=qd.f32),  # (*brick, F)
    weights: qd.types.ndarray(dtype=qd.f32),  # brick
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
    N, off = _load_grid_spec(grid_spec, dim)
    cs = qd.Vector.zero(qd.f32, dim)
    inv = qd.Vector.zero(qd.f32, dim)
    for d in qd.static(range(dim)):
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
                evaluate_fn,
                quad_points,
                dim,
                True,
                lo,
                ext,
                x_cell,
                0.0,
                E,
                ev,
                kernel_prefactor,
                cs,
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
                buf,
                i0,
                wj,
                n,
                N,
                off,
                periodic,
                B,
                dim,
                weights,
                fields,
                quantities,
                dropped,
                brick,
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
                    cd, ok = _brick_index(
                        qd.floor(x_sample * inv[d], int),
                        N[d],
                        off[d],
                        periodic,
                    )
                    cell[d] = cd
                    valid = valid and ok
                if valid:
                    integral = sample_integrals[s]
                    if integral != 0.0:
                        _grid_add(
                            weights,
                            fields,
                            quantities,
                            dropped,
                            brick,
                            n,
                            cell,
                            wj * integral,
                            dim,
                        )

        else:
            # per-cell quadrature, renormalised with the pass-1 total
            _quadrature_deposit(
                evaluate_fn,
                quad_points,
                dim,
                True,
                periodic,
                lo,
                ext,
                x_cell,
                0.0,
                E,
                ev,
                kernel_prefactor,
                cs,
                N,
                off,
                wj,
                1.0 / total_integral,
                n,
                weights,
                fields,
                quantities,
                dropped,
                brick,
            )


# =============================================================================
# Python wrappers
# =============================================================================
_ADAPTIVE_KERNELS = {
    "separable": _separable,
    "isotropic": _isotropic,
    "covariant": _covariant,
}

# Extra cells on each side of a brick: the host bounds below round
# differently from the kernels, so a particle's first or last cell can differ
# by one between the two.
_BRICK_MARGIN = 2


def _largest_gap_cover(cells, n):
    """Return ``(start, length)`` of the shortest periodic run holding `cells`.

    `cells` are integers in ``[0, n)``; the run is the complement of the
    largest run of absent cells.
    """
    present = np.flatnonzero(np.bincount(cells, minlength=n))
    gaps = np.diff(np.append(present, present[0] + n)) - 1
    k = int(np.argmax(gaps))
    return int(present[(k + 1) % present.size]), n - int(gaps[k])


def _axis_cover(g, below, above, n, periodic):
    """Return ``(offset, length)`` of the cells touched along one axis.

    `g` holds the particles' positions in cells; a particle touches cells
    ``floor(g - below) ... floor(g + above)`` (`below` and `above` scalars or
    per-particle arrays). Non-periodic covers are clipped to ``[0, n)``. On a
    periodic axis the cover may run past the end and wrap around; when the
    direct range is longer than half the axis (particles near both ends), the
    largest gap between occupied cells is cut out instead.
    """
    first = int(np.floor(np.min(g - below))) - _BRICK_MARGIN
    last = int(np.floor(np.max(g + above))) + _BRICK_MARGIN
    if not periodic:
        lo = min(max(first, 0), n)
        return lo, max(min(last + 1, n) - lo, 0)
    length = last - first + 1
    if 2 * length > n:
        start, span = _largest_gap_cover(np.floor(g).astype(np.int64) % n, n)
        reach_below = int(np.ceil(np.max(below))) + _BRICK_MARGIN + 1
        reach_above = int(np.ceil(np.max(above))) + _BRICK_MARGIN + 1
        if span + reach_below + reach_above < length:
            first, length = start - reach_below, span + reach_below + reach_above
    if length >= n:
        return 0, n
    return first % n, length


def _grid_spec(gridnums, periodic, positions=None, inv=None, reach=None):
    """Return the ``(2, D)`` grid spec, the brick's offset and its shape.

    Without `reach` the output grid is the full grid. Otherwise
    ``reach(d)`` gives ``(below, above)``: how many cells (scalars or
    per-particle arrays) the particles touch below and above their position
    along axis d, and the output grid is the brick of touched cells (see
    :func:`_axis_cover`). Grid spec rows: global grid size and brick offset
    per axis.
    """
    gridnums = np.asarray(gridnums, dtype=np.int64)
    lo = np.zeros(gridnums.size, dtype=np.int64)
    shape = gridnums.copy()
    if reach is not None:
        if positions.shape[0] == 0:
            shape[:] = 0
        else:
            for d in range(gridnums.size):
                g = positions[:, d] * np.float32(inv[d])
                below, above = reach(d)
                lo[d], shape[d] = _axis_cover(
                    g, below, above, int(gridnums[d]), periodic
                )
    spec = np.stack([gridnums, lo]).astype(np.int32)
    return spec, lo, tuple(int(n) for n in shape)


@dataclass(slots=True)
class KernelPreparation:
    """A prepared deposition: call ``kernel(*args)``, read the grids."""

    kernel: Any
    args: tuple
    grid_fields: np.ndarray  # (*brick, F), filled in place
    grid_weights: np.ndarray  # brick, filled in place
    offset: np.ndarray  # global index of the brick's first cell, (D,)
    dropped: np.ndarray  # (1,) count of writes outside the brick
    sample_buffer: int | None = None  # isotropic / covariant only


def _allocate(gridnums, periodic, F, positions, inv, reach):
    """Grid spec, brick offset, zeroed brick grids and the dropped-write counter.

    `reach` is None for the full grid (see :func:`_grid_spec`).
    """
    spec, lo, shape = _grid_spec(gridnums, periodic, positions, inv, reach)
    grid_fields = np.zeros((*shape, F), dtype=np.float32)
    grid_weights = np.zeros(shape, dtype=np.float32)
    return spec, lo, grid_fields, grid_weights, np.zeros(1, dtype=np.int32)


def _prepare_fixed(
    particle_positions,
    particle_fields,
    particle_weights,
    boxsizes,
    gridnums,
    periodic,
    kernel_name,
    brick,
):
    dim = particle_positions.shape[1]
    try:
        kernel = _FIXED_GRID_KERNELS[(kernel_name, dim)]
    except KeyError:
        raise ValueError(
            f"Unsupported combination: kernel_name={kernel_name!r}, dim={dim}."
        ) from None
    _, K, SHIFT = _FIXED_STENCILS[kernel_name]
    inv_dx = np.asarray(gridnums, dtype=np.float32) / boxsizes

    def reach(d):  # cells floor(g) - SHIFT + [-1, K - 1]
        return SHIFT + 1, K - 1 - SHIFT

    spec, lo, grid_fields, grid_weights, dropped = _allocate(
        gridnums,
        periodic,
        particle_fields.shape[1],
        particle_positions,
        inv_dx,
        reach if brick else None,
    )
    args = (
        bool(periodic),
        bool(brick),
        particle_positions,
        particle_fields,
        particle_weights,
        boxsizes,
        spec,
        dropped,
        grid_fields,
        grid_weights,
    )
    return KernelPreparation(kernel, args, grid_fields, grid_weights, lo, dropped)


def _prepare_separable(
    particle_positions,
    particle_fields,
    particle_weights,
    boxsizes,
    gridnums,
    periodic,
    kernel_name,
    brick,
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

    scale = 1.0
    if brick and periodic and particle_hsml.shape[0]:
        # the kernel's clamp: largest axis at most max_support
        scale = np.minimum(
            1.0, max_support / np.maximum(particle_hsml.max(axis=1), 1e-30)
        )

    def reach(d):
        half = support * (particle_hsml[:, d] * scale) * np.float32(cell_size_invs[d])
        return half, half

    spec, lo, grid_fields, grid_weights, dropped = _allocate(
        gridnums,
        periodic,
        particle_fields.shape[1],
        particle_positions,
        cell_size_invs,
        reach if brick else None,
    )

    args = (
        kspec.F_1d,
        kspec.sigma,
        support,
        DIM,
        bool(periodic),
        bool(brick),
        particle_positions,
        particle_fields,
        particle_hsml,
        particle_weights,
        np.array(cell_size_invs, dtype=np.float32),
        max_support,
        max_stencil,
        spec,
        dropped,
        grid_fields,
        grid_weights,
    )
    return KernelPreparation(_separable, args, grid_fields, grid_weights, lo, dropped)


_QUADRATURE_TABLES = {
    1: QUADRATURE_POINTS_1D,
    2: QUADRATURE_POINTS_2D,
    3: QUADRATURE_POINTS_3D,
}


def _radial_reach(structure, smoothing_args, support, max_support, periodic, inv):
    """``reach(d)`` for :func:`_grid_spec`: the support's half-extent in cells.

    Sample points lie inside the support, so it bounds every branch.
    """
    if structure == "isotropic":
        h = smoothing_args[0]
        if periodic:
            h = np.minimum(h, np.float32(max_support))

        def reach(d):
            half = support * h * np.float32(inv[d])
            return half, half

        return reach

    eigvecs, eigvals = smoothing_args
    eigvals = np.abs(eigvals)
    if periodic and eigvals.shape[0]:  # the kernel's clamp on the largest axis
        largest = eigvals.max(axis=1, keepdims=True)
        eigvals = eigvals * np.minimum(1.0, max_support / np.maximum(largest, 1e-30))

    def reach(d):  # axis d: sqrt(sum_c (E[d, c] ev[c])^2)
        s2 = np.einsum("nc,nc->n", eigvecs[:, d, :] ** 2, eigvals**2)
        half = support * np.sqrt(s2) * np.float32(inv[d])
        return half, half

    return reach


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
    brick,
    integration_method,
    eta_crit,
    projected=False,
):
    """Prepare an isotropic or covariant deposition.

    Both use the same kernel family, cell quadrature and sample table; only
    the smoothing arrays differ. `projected` deposits the 3D kernel integrated
    along the line of sight (2D grids only).
    """
    DIM = particle_positions.shape[1]
    gridnums = tuple(int(n) for n in gridnums)
    boxsizes = tuple(float(b) for b in boxsizes)
    if projected and DIM != 2:
        raise ValueError(f"projected=True needs 2D positions, got {DIM}D")

    kspec = create_spherical_kernel(kernel_name, projected)
    support = kspec.support

    quad_table = _QUADRATURE_TABLES[DIM]
    if integration_method not in quad_table:
        raise ValueError(
            f"Unknown integration method: {integration_method!r}. "
            f"Available: {sorted(quad_table.keys())}"
        )
    quad_points = quad_table[integration_method]

    grid = build_kernel_sample_grid(kernel_name, DIM, eta_crit, projected=projected)
    sample_buffer = _sample_buffer_size(eta_crit, DIM)

    cell_sizes = tuple(boxsizes[d] / gridnums[d] for d in range(DIM))
    cell_size_invs = tuple(1.0 / c for c in cell_sizes)
    max_support = min(0.5 * boxsizes[d] / support for d in range(DIM))

    smoothing_args = tuple(as_f32(a) for a in smoothing_args)
    reach = None
    if brick:
        reach = _radial_reach(
            structure, smoothing_args, support, max_support, periodic, cell_size_invs
        )
    spec, lo, grid_fields, grid_weights, dropped = _allocate(
        gridnums,
        periodic,
        particle_fields.shape[1],
        particle_positions,
        cell_size_invs,
        reach,
    )

    args = (
        kspec.evaluate,
        kspec.sigma,
        support,
        DIM,
        quad_points,
        bool(periodic),
        bool(brick),
        float(eta_crit),
        sample_buffer,
        particle_positions,
        particle_fields,
        *smoothing_args,
        particle_weights,
        grid["coords"],
        grid["integrals"],
        grid["count"],
        np.array(cell_sizes, dtype=np.float32),
        np.array(cell_size_invs, dtype=np.float32),
        max_support,
        spec,
        dropped,
        grid_fields,
        grid_weights,
    )
    return KernelPreparation(
        _ADAPTIVE_KERNELS[structure],
        args,
        grid_fields,
        grid_weights,
        lo,
        dropped,
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
    projected=False,
    brick=False,
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
    projected : bool, default False
        Isotropic and covariant, 2D only: deposit the 3D kernel integrated
        along the line of sight (plane projection of 3D particles, whose
        smoothing is the shadow of the 3D one) instead of the 2D kernel.
    brick : bool, default False
        Deposit onto the smallest box of cells (a brick) holding every cell
        the particles touch, instead of the full grid, and also return the
        brick's offset. Each cell value equals the full-grid value bit for
        bit. On a periodic axis the brick may run past the end of the axis
        and continue at its start (cell ``offset + i`` is global cell
        ``(offset + i) % gridnum``).

    Returns
    -------
    grid_fields : ndarray, shape (F, *gridnums) or (F, *brick_shape)
    grid_weights : ndarray, shape (*gridnums,) or (*brick_shape,)
    offset : ndarray, shape (D,)
        Only with ``brick=True``: global index of the brick's first cell.

    """
    particle_positions = as_f32(particle_positions)
    particle_fields = as_f32(particle_fields)
    particle_weights = as_f32(particle_weights)
    boxsizes = as_f32(boxsizes)
    check_index_limits(
        particle_fields=particle_fields,
        grid_fields=(*gridnums, particle_fields.shape[1]),
    )

    common = dict(
        particle_positions=particle_positions,
        particle_fields=particle_fields,
        particle_weights=particle_weights,
        boxsizes=boxsizes,
        gridnums=gridnums,
        periodic=periodic,
        kernel_name=kernel_name,
        brick=brick,
    )

    if not adaptive:
        check_fixed_grid_kernel(kernel_name, structure)
        prep = _prepare_fixed(**common)
    else:
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
            name
            for name in _REQUIRED_KWARGS[structure]
            if provided_kwargs[name] is None
        ]
        if missing:
            raise ValueError(
                f"structure={structure!r} requires: {', '.join(missing)} (got None)."
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
                projected=projected,
            )

    if prep.grid_weights.size:
        prep.kernel(*prep.args)
        if prep.dropped[0]:
            raise RuntimeError(
                f"internal error: {int(prep.dropped[0])} deposits fell outside the "
                "computed brick bounds"
            )
    grids = (np.moveaxis(prep.grid_fields, -1, 0), prep.grid_weights)
    return (*grids, prep.offset) if brick else grids
