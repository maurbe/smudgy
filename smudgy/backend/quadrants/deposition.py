from dataclasses import dataclass
from typing import Any

import numpy as np
import quadrants as qd

from .._common import ADAPTIVE_STRUCTURES, as_f32, check_fixed_grid_kernel
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
    """Returns (index, valid). `valid` is always True when periodic
    (wrapped via modulo); for non-periodic it's False when pos falls
    outside the grid -- mirrors the C++ optional-returning
    cell_index_from_pos, minus the std::optional machinery.
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
    """Cells per axis of the per-particle local buffer used by the sample
    branch of isotropic/covariant deposition.

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


@qd.func
def _isotropic_cell_integral_1d(
    evaluate_fn: qd.template(),
    quad_points: qd.template(),
    dim: qd.template(),
    a: qd.i32,
    x_cell: qd.f32,
    hsm_phys: qd.f32,
    kernel_prefactor: qd.f32,
    cellSize_x: qd.f32,
) -> qd.f32:
    """Quadrature-integrated kernel mass over cell a."""
    s = 0.0

    for ox, coeff in qd.static(quad_points):
        dx = (x_cell - (a + ox)) * cellSize_x
        q = qd.abs(dx) / hsm_phys
        s += coeff * kernel_prefactor * evaluate_fn(q, dim)

    return s * cellSize_x


@qd.func
def _isotropic_cell_integral_2d(
    evaluate_fn: qd.template(),
    quad_points: qd.template(),
    dim: qd.template(),
    a: qd.i32,
    b: qd.i32,
    x_cell: qd.f32,
    y_cell: qd.f32,
    hsm_phys: qd.f32,
    kernel_prefactor: qd.f32,
    cellSize_x: qd.f32,
    cellSize_y: qd.f32,
) -> qd.f32:
    """Quadrature-integrated kernel mass over cell (a, b), matching the
    C++ `eval` lambda + integrate_cell_2d(...) * cellSize_x * cellSize_y.
    """
    s = 0.0

    for ox, oy, coeff in qd.static(quad_points):

        dx = (x_cell - (a + ox)) * cellSize_x
        dy = (y_cell - (b + oy)) * cellSize_y

        r = qd.sqrt(dx * dx + dy * dy)
        q = r / hsm_phys

        s += coeff * kernel_prefactor * evaluate_fn(q, dim)

    return s * cellSize_x * cellSize_y


@qd.func
def _isotropic_cell_integral_3d(
    evaluate_fn: qd.template(),
    quad_points: qd.template(),
    dim: qd.template(),
    a: qd.i32,
    b: qd.i32,
    c: qd.i32,
    x_cell: qd.f32,
    y_cell: qd.f32,
    z_cell: qd.f32,
    hsm_phys: qd.f32,
    kernel_prefactor: qd.f32,
    cellSize_x: qd.f32,
    cellSize_y: qd.f32,
    cellSize_z: qd.f32,
) -> qd.f32:
    """Quadrature-integrated kernel mass over cell (a, b, c)."""
    s = 0.0

    for ox, oy, oz, coeff in qd.static(quad_points):

        dx = (x_cell - (a + ox)) * cellSize_x
        dy = (y_cell - (b + oy)) * cellSize_y
        dz = (z_cell - (c + oz)) * cellSize_z

        r = qd.sqrt(dx * dx + dy * dy + dz * dz)
        q = r / hsm_phys

        s += coeff * kernel_prefactor * evaluate_fn(q, dim)

    return s * cellSize_x * cellSize_y * cellSize_z


@qd.func
def _covariant_cell_integral_1d(
    evaluate_fn: qd.template(),
    quad_points: qd.template(),
    dim: qd.template(),
    a: qd.i32,
    x_cell: qd.f32,
    e0x: qd.f32,
    eval0: qd.f32,
    kernel_prefactor: qd.f32,
    cellSize_x: qd.f32,
) -> qd.f32:

    s = 0.0

    for ox, coeff in qd.static(quad_points):
        dx = (x_cell - (a + ox)) * cellSize_x

        # project the separation vector onto each principal axis and
        # scale by that axis's own semi-axis length -- this is what
        # turns the isotropic q = |d| / h into the ellipsoidal q
        xi0 = (e0x * dx) / eval0
        q = qd.abs(xi0)  # kernels expect q >= 0 (2D/3D get it from the sqrt)
        s += coeff * kernel_prefactor * evaluate_fn(q, dim)

    return s * cellSize_x


@qd.func
def _covariant_cell_integral_2d(
    evaluate_fn: qd.template(),
    quad_points: qd.template(),
    dim: qd.template(),
    a: qd.i32,
    b: qd.i32,
    x_cell: qd.f32,
    y_cell: qd.f32,
    e0x: qd.f32,
    e0y: qd.f32,
    e1x: qd.f32,
    e1y: qd.f32,
    eval0: qd.f32,
    eval1: qd.f32,
    kernel_prefactor: qd.f32,
    cellSize_x: qd.f32,
    cellSize_y: qd.f32,
) -> qd.f32:

    s = 0.0

    for ox, oy, coeff in qd.static(quad_points):
        dx = (x_cell - (a + ox)) * cellSize_x
        dy = (y_cell - (b + oy)) * cellSize_y

        # project the separation vector onto each principal axis and
        # scale by that axis's own semi-axis length -- this is what
        # turns the isotropic q = |d| / h into the ellipsoidal q
        xi0 = (e0x * dx + e0y * dy) / eval0
        xi1 = (e1x * dx + e1y * dy) / eval1

        q = qd.sqrt(xi0 * xi0 + xi1 * xi1)

        s += coeff * kernel_prefactor * evaluate_fn(q, dim)

    return s * cellSize_x * cellSize_y


@qd.func
def _covariant_cell_integral_3d(
    evaluate_fn: qd.template(),
    quad_points: qd.template(),
    dim: qd.template(),
    a: qd.i32,
    b: qd.i32,
    c: qd.i32,
    x_cell: qd.f32,
    y_cell: qd.f32,
    z_cell: qd.f32,
    e0x: qd.f32,
    e0y: qd.f32,
    e0z: qd.f32,
    e1x: qd.f32,
    e1y: qd.f32,
    e1z: qd.f32,
    e2x: qd.f32,
    e2y: qd.f32,
    e2z: qd.f32,
    eval0: qd.f32,
    eval1: qd.f32,
    eval2: qd.f32,
    kernel_prefactor: qd.f32,
    cellSize_x: qd.f32,
    cellSize_y: qd.f32,
    cellSize_z: qd.f32,
) -> qd.f32:

    s = 0.0

    for ox, oy, oz, coeff in qd.static(quad_points):
        dx = (x_cell - (a + ox)) * cellSize_x
        dy = (y_cell - (b + oy)) * cellSize_y
        dz = (z_cell - (c + oz)) * cellSize_z

        # project the separation vector onto each principal axis and
        # scale by that axis's own semi-axis length -- this is what
        # turns the isotropic q = |d| / h into the ellipsoidal q
        xi0 = (e0x * dx + e0y * dy + e0z * dz) / eval0
        xi1 = (e1x * dx + e1y * dy + e1z * dz) / eval1
        xi2 = (e2x * dx + e2y * dy + e2z * dz) / eval2

        q = qd.sqrt(xi0 * xi0 + xi1 * xi1 + xi2 * xi2)

        s += coeff * kernel_prefactor * evaluate_fn(q, dim)

    return s * cellSize_x * cellSize_y * cellSize_z


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
    """Cubic B-spline (was "pcs")."""
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
    """Quintic B-spline (6-point, was "pqs")."""
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
# ||                                                                         ||
# ||                       ADAPTIVE KERNELS                                  ||
# ||                                                                         ||
# =============================================================================


# =============================================================================
# Separable Kernels
# =============================================================================
@qd.kernel
def _separable_1d(
    F_1d_fn: qd.template(),
    sigma_fn: qd.template(),
    kernel_support: qd.f32,
    dim: qd.template(),  # only new line in the signature
    periodic: qd.template(),
    positions: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, 1)
    quantities: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, F)
    smoothing_lengths: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, 1)
    particle_weights: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (N,)
    cellSize_x_inv: qd.f32,
    max_support: qd.f32,
    MAX_STENCIL: qd.template(),
    Nx: qd.i32,
    fields: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (Nx, F)
    weights: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (Nx,)
):
    F = fields.shape[1]
    sig = sigma_fn(dim)

    for n in range(positions.shape[0]):

        wx = qd.Vector.zero(qd.f32, MAX_STENCIL)
        wj = particle_weights[n]

        x_phys = positions[n, 0]

        hsm_x_phys = smoothing_lengths[n, 0]

        if qd.static(periodic):
            hsm_max = hsm_x_phys
            if hsm_max > max_support:
                scale = max_support / hsm_max
                hsm_x_phys *= scale

        x_cell = x_phys * cellSize_x_inv

        hsm_x_cell = hsm_x_phys * cellSize_x_inv

        inv_hx = 1.0 / hsm_x_cell

        support_x_cell = kernel_support * hsm_x_cell

        i_min = qd.cast(qd.floor(x_cell - support_x_cell), qd.i32)
        i_max = qd.cast(qd.ceil(x_cell + support_x_cell), qd.i32) - 1

        if qd.static(not periodic):
            i_min = qd.max(i_min, 0)
            i_max = qd.min(i_max, Nx - 1)

        nx = i_max - i_min + 1
        for si in range(nx):
            i = i_min + si
            q_left = (i - x_cell) * inv_hx
            q_right = ((i + 1) - x_cell) * inv_hx
            wx[si] = (
                wj
                * sig
                * _separable_integrate_1d(F_1d_fn, kernel_support, dim, q_left, q_right)
            )  # sig here!

        for si in qd.ndrange(nx):

            i = i_min + si

            ii = i
            if qd.static(periodic):
                ii = _apply_pbc(i, Nx)

            w = wx[si]
            qd.atomic_add(weights[ii], w)
            for f in range(F):
                qd.atomic_add(
                    fields[ii, f],
                    quantities[n, f] * w,
                )


@qd.kernel
def _separable_2d(
    F_1d_fn: qd.template(),
    sigma_fn: qd.template(),
    kernel_support: qd.f32,
    dim: qd.template(),  # only new line in the signature
    periodic: qd.template(),
    positions: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, 2)
    quantities: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, F)
    smoothing_lengths: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, 2)
    particle_weights: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (N,)
    cellSize_x_inv: qd.f32,
    cellSize_y_inv: qd.f32,
    max_support: qd.f32,
    MAX_STENCIL: qd.template(),
    Nx: qd.i32,
    Ny: qd.i32,
    fields: qd.types.ndarray(dtype=qd.f32, ndim=3),  # (Nx, Ny, F)
    weights: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (Nx, Ny)
):
    F = fields.shape[2]
    sig = sigma_fn(dim)

    for n in range(positions.shape[0]):

        wx = qd.Vector.zero(qd.f32, MAX_STENCIL)
        wy = qd.Vector.zero(qd.f32, MAX_STENCIL)
        wj = particle_weights[n]

        x_phys = positions[n, 0]
        y_phys = positions[n, 1]

        hsm_x_phys = smoothing_lengths[n, 0]
        hsm_y_phys = smoothing_lengths[n, 1]

        if qd.static(periodic):
            hsm_max = qd.max(hsm_x_phys, hsm_y_phys)
            if hsm_max > max_support:
                scale = max_support / hsm_max
                hsm_x_phys *= scale
                hsm_y_phys *= scale

        x_cell = x_phys * cellSize_x_inv
        y_cell = y_phys * cellSize_y_inv

        hsm_x_cell = hsm_x_phys * cellSize_x_inv
        hsm_y_cell = hsm_y_phys * cellSize_y_inv

        inv_hx = 1.0 / hsm_x_cell
        inv_hy = 1.0 / hsm_y_cell

        support_x_cell = kernel_support * hsm_x_cell
        support_y_cell = kernel_support * hsm_y_cell

        i_min = qd.cast(qd.floor(x_cell - support_x_cell), qd.i32)
        i_max = qd.cast(qd.ceil(x_cell + support_x_cell), qd.i32) - 1
        j_min = qd.cast(qd.floor(y_cell - support_y_cell), qd.i32)
        j_max = qd.cast(qd.ceil(y_cell + support_y_cell), qd.i32) - 1

        if qd.static(not periodic):
            i_min = qd.max(i_min, 0)
            i_max = qd.min(i_max, Nx - 1)
            j_min = qd.max(j_min, 0)
            j_max = qd.min(j_max, Ny - 1)

        nx = i_max - i_min + 1
        for si in range(nx):
            i = i_min + si
            q_left = (i - x_cell) * inv_hx
            q_right = ((i + 1) - x_cell) * inv_hx
            wx[si] = sig * _separable_integrate_1d(
                F_1d_fn, kernel_support, dim, q_left, q_right
            )  # sig here!

        ny = j_max - j_min + 1
        for sj in range(ny):
            j = j_min + sj
            q_left = (j - y_cell) * inv_hy
            q_right = ((j + 1) - y_cell) * inv_hy
            wy[sj] = wj * _separable_integrate_1d(
                F_1d_fn, kernel_support, dim, q_left, q_right
            )  # wj here!

        for si, sj in qd.ndrange(nx, ny):

            i = i_min + si
            j = j_min + sj

            ii = i
            jj = j
            if qd.static(periodic):
                ii = _apply_pbc(i, Nx)
                jj = _apply_pbc(j, Ny)

            w = wx[si] * wy[sj]
            qd.atomic_add(weights[ii, jj], w)
            for f in range(F):
                qd.atomic_add(
                    fields[ii, jj, f],
                    quantities[n, f] * w,
                )


@qd.kernel
def _separable_3d(
    F_1d_fn: qd.template(),
    sigma_fn: qd.template(),
    kernel_support: qd.f32,
    dim: qd.template(),  # only new line in the signature
    periodic: qd.template(),
    positions: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, 3)
    quantities: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, F)
    smoothing_lengths: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, 3)
    particle_weights: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (N,)
    cellSize_x_inv: qd.f32,
    cellSize_y_inv: qd.f32,
    cellSize_z_inv: qd.f32,
    max_support: qd.f32,
    MAX_STENCIL: qd.template(),
    Nx: qd.i32,
    Ny: qd.i32,
    Nz: qd.i32,
    fields: qd.types.ndarray(dtype=qd.f32, ndim=4),  # (Nx, Ny, Nz, F)
    weights: qd.types.ndarray(dtype=qd.f32, ndim=3),  # (Nx, Ny, Nz)
):
    F = fields.shape[3]
    sig = sigma_fn(dim)

    for n in range(positions.shape[0]):

        wx = qd.Vector.zero(qd.f32, MAX_STENCIL)
        wy = qd.Vector.zero(qd.f32, MAX_STENCIL)
        wz = qd.Vector.zero(qd.f32, MAX_STENCIL)
        wj = particle_weights[n]

        x_phys = positions[n, 0]
        y_phys = positions[n, 1]
        z_phys = positions[n, 2]

        hsm_x_phys = smoothing_lengths[n, 0]
        hsm_y_phys = smoothing_lengths[n, 1]
        hsm_z_phys = smoothing_lengths[n, 2]

        if qd.static(periodic):
            hsm_max = qd.max(hsm_x_phys, qd.max(hsm_y_phys, hsm_z_phys))
            if hsm_max > max_support:
                scale = max_support / hsm_max
                hsm_x_phys *= scale
                hsm_y_phys *= scale
                hsm_z_phys *= scale

        x_cell = x_phys * cellSize_x_inv
        y_cell = y_phys * cellSize_y_inv
        z_cell = z_phys * cellSize_z_inv

        hsm_x_cell = hsm_x_phys * cellSize_x_inv
        hsm_y_cell = hsm_y_phys * cellSize_y_inv
        hsm_z_cell = hsm_z_phys * cellSize_z_inv

        inv_hx = 1.0 / hsm_x_cell
        inv_hy = 1.0 / hsm_y_cell
        inv_hz = 1.0 / hsm_z_cell

        support_x_cell = kernel_support * hsm_x_cell
        support_y_cell = kernel_support * hsm_y_cell
        support_z_cell = kernel_support * hsm_z_cell

        i_min = qd.cast(qd.floor(x_cell - support_x_cell), qd.i32)
        i_max = qd.cast(qd.ceil(x_cell + support_x_cell), qd.i32) - 1
        j_min = qd.cast(qd.floor(y_cell - support_y_cell), qd.i32)
        j_max = qd.cast(qd.ceil(y_cell + support_y_cell), qd.i32) - 1
        k_min = qd.cast(qd.floor(z_cell - support_z_cell), qd.i32)
        k_max = qd.cast(qd.ceil(z_cell + support_z_cell), qd.i32) - 1

        if qd.static(not periodic):
            i_min = qd.max(i_min, 0)
            i_max = qd.min(i_max, Nx - 1)
            j_min = qd.max(j_min, 0)
            j_max = qd.min(j_max, Ny - 1)
            k_min = qd.max(k_min, 0)
            k_max = qd.min(k_max, Nz - 1)

        nx = i_max - i_min + 1
        for si in range(nx):
            i = i_min + si
            q_left = (i - x_cell) * inv_hx
            q_right = ((i + 1) - x_cell) * inv_hx
            wx[si] = sig * _separable_integrate_1d(
                F_1d_fn, kernel_support, dim, q_left, q_right
            )

        ny = j_max - j_min + 1
        for sj in range(ny):
            j = j_min + sj
            q_left = (j - y_cell) * inv_hy
            q_right = ((j + 1) - y_cell) * inv_hy
            wy[sj] = wj * _separable_integrate_1d(
                F_1d_fn, kernel_support, dim, q_left, q_right
            )

        nz = k_max - k_min + 1
        for sk in range(nz):
            k = k_min + sk
            q_left = (k - z_cell) * inv_hz
            q_right = ((k + 1) - z_cell) * inv_hz
            wz[sk] = _separable_integrate_1d(
                F_1d_fn, kernel_support, dim, q_left, q_right
            )

        for si, sj, sk in qd.ndrange(nx, ny, nz):

            i = i_min + si
            j = j_min + sj
            k = k_min + sk

            ii = i
            jj = j
            kk = k
            if qd.static(periodic):
                ii = _apply_pbc(i, Nx)
                jj = _apply_pbc(j, Ny)
                kk = _apply_pbc(k, Nz)

            w = wx[si] * wy[sj] * wz[sk]

            qd.atomic_add(weights[ii, jj, kk], w)
            for f in range(F):
                qd.atomic_add(
                    fields[ii, jj, kk, f],
                    quantities[n, f] * w,
                )


# =============================================================================
# Isotropic Kernels
# =============================================================================
@qd.kernel
def _isotropic_1d(
    evaluate_fn: qd.template(),
    sigma_fn: qd.template(),
    kernel_support: qd.f32,
    dim: qd.template(),  # only new line in the signature
    quad_points: qd.template(),
    periodic: qd.template(),
    eta_crit: qd.f32,
    B: qd.template(),
    positions: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, 1)
    quantities: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, F)
    smoothing_lengths: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (N,)
    particle_weights: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (N,)
    sample_coords: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (S, 1)
    sample_integrals: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (S,)
    num_samples: qd.i32,
    cellSize_x: qd.f32,
    cellSize_x_inv: qd.f32,
    max_support: qd.f32,
    Nx: qd.i32,
    fields: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (Nx, F)
    weights: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (Nx)
):
    F = fields.shape[1]
    sig = sigma_fn(dim)

    for n in range(positions.shape[0]):
        x_phys = positions[n, 0]

        hsm_phys = smoothing_lengths[n]
        if qd.static(periodic):
            if hsm_phys > max_support:
                hsm_phys = max_support

        kernel_prefactor = sig / hsm_phys

        x_cell = x_phys * cellSize_x_inv

        support_phys = kernel_support * hsm_phys
        support_x_cell = support_phys * cellSize_x_inv

        eta = 2.0 * support_x_cell  # diameter convention

        wj = particle_weights[n]

        i0 = qd.floor(x_cell - support_x_cell, int)
        fits = qd.floor(x_cell + support_x_cell, int) - i0 < B

        if eta < eta_crit and fits:
            # --- anti-aliasing branch: deposit via precomputed kernel samples,
            # summed per cell in a local buffer first (one atomic per touched
            # cell instead of one per sample) ---
            buf = qd.Vector.zero(qd.f32, B)
            for s in range(num_samples):
                x_sample = x_phys + sample_coords[s, 0] * hsm_phys
                li = qd.floor(x_sample * cellSize_x_inv, int) - i0
                if 0 <= li < B:
                    buf[li] += sample_integrals[s]

            for a in range(B):
                v = buf[a]
                if v != 0.0:
                    ix, ix_valid = _cell_index_from_int(i0 + a, Nx, periodic)
                    if ix_valid:
                        w = wj * v
                        qd.atomic_add(weights[ix], w)
                        for f in range(F):
                            qd.atomic_add(fields[ix, f], quantities[n, f] * w)

        elif eta < eta_crit:
            # --- same, direct per-sample deposition (particle too large for
            # the local buffer) ---
            for s in range(num_samples):
                x_sample = x_phys + sample_coords[s, 0] * hsm_phys

                ix, ix_valid = _cell_index_from_pos(
                    x_sample, cellSize_x_inv, Nx, periodic
                )

                if ix_valid:
                    integral = sample_integrals[s]
                    if integral != 0.0:
                        w = wj * integral
                        qd.atomic_add(weights[ix], w)
                        for f in range(F):
                            qd.atomic_add(fields[ix, f], quantities[n, f] * w)

        else:
            # --- cell-quadrature branch, mass-conserving via two-pass renormalization ---
            # cell ceil(x + s) lies entirely outside the (compact) support,
            # so the stencil stops one cell before it
            i_min = i0
            i_max = qd.ceil(x_cell + support_x_cell, int) - 1

            if qd.static(not periodic):
                i_min = qd.max(i_min, 0)
                i_max = qd.min(i_max, Nx - 1)

            # pass 1: total mass over the affected cells (no per-cell storage)
            total_integral = 0.0
            for a in range(i_min, i_max + 1):
                integral = _isotropic_cell_integral_1d(
                    evaluate_fn,
                    quad_points,
                    dim,
                    a,
                    x_cell,
                    hsm_phys,
                    kernel_prefactor,
                    cellSize_x,
                )
                total_integral += integral

            if total_integral > 0.0:  # BUGFIX: C++ divides unconditionally
                correction = 1.0 / total_integral

                # pass 2: recompute (cheap) + deposit normalized weight
                for a in range(i_min, i_max + 1):
                    aa = a
                    if qd.static(periodic):
                        aa = _apply_pbc(a, Nx)

                    integral = _isotropic_cell_integral_1d(
                        evaluate_fn,
                        quad_points,
                        dim,
                        a,
                        x_cell,
                        hsm_phys,
                        kernel_prefactor,
                        cellSize_x,
                    )
                    w_corr = wj * integral * correction

                    qd.atomic_add(weights[aa], w_corr)
                    for f in range(F):
                        qd.atomic_add(fields[aa, f], quantities[n, f] * w_corr)


@qd.kernel
def _isotropic_2d(
    evaluate_fn: qd.template(),
    sigma_fn: qd.template(),
    kernel_support: qd.f32,
    dim: qd.template(),  # only new line in the signature
    quad_points: qd.template(),
    periodic: qd.template(),
    eta_crit: qd.f32,
    B: qd.template(),
    positions: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, 2)
    quantities: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, F)
    smoothing_lengths: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (N,)
    particle_weights: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (N,)
    sample_coords: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (S, 2)
    sample_integrals: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (S,)
    num_samples: qd.i32,
    cellSize_x: qd.f32,
    cellSize_y: qd.f32,
    cellSize_x_inv: qd.f32,
    cellSize_y_inv: qd.f32,
    max_support: qd.f32,
    Nx: qd.i32,
    Ny: qd.i32,
    fields: qd.types.ndarray(dtype=qd.f32, ndim=3),  # (Nx, Ny, F)
    weights: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (Nx, Ny)
):
    F = fields.shape[2]
    sig = sigma_fn(dim)

    for n in range(positions.shape[0]):
        x_phys = positions[n, 0]
        y_phys = positions[n, 1]

        hsm_phys = smoothing_lengths[n]
        if qd.static(periodic):
            if hsm_phys > max_support:
                hsm_phys = max_support

        kernel_prefactor = sig / (hsm_phys * hsm_phys)

        x_cell = x_phys * cellSize_x_inv
        y_cell = y_phys * cellSize_y_inv

        support_phys = kernel_support * hsm_phys
        support_x_cell = support_phys * cellSize_x_inv
        support_y_cell = support_phys * cellSize_y_inv

        eta = 2.0 * qd.min(support_x_cell, support_y_cell)  # diameter convention

        wj = particle_weights[n]

        i0 = qd.floor(x_cell - support_x_cell, int)
        j0 = qd.floor(y_cell - support_y_cell, int)
        fits = (qd.floor(x_cell + support_x_cell, int) - i0 < B) and (
            qd.floor(y_cell + support_y_cell, int) - j0 < B
        )

        if eta < eta_crit and fits:
            # --- anti-aliasing branch: deposit via precomputed kernel samples,
            # summed per cell in a local buffer first (one atomic per touched
            # cell instead of one per sample) ---
            buf = qd.Vector.zero(qd.f32, B * B)
            for s in range(num_samples):
                x_sample = x_phys + sample_coords[s, 0] * hsm_phys
                y_sample = y_phys + sample_coords[s, 1] * hsm_phys
                li = qd.floor(x_sample * cellSize_x_inv, int) - i0
                lj = qd.floor(y_sample * cellSize_y_inv, int) - j0
                if 0 <= li < B and 0 <= lj < B:
                    buf[li * B + lj] += sample_integrals[s]

            for a in range(B):
                ix, ix_valid = _cell_index_from_int(i0 + a, Nx, periodic)
                for b in range(B):
                    v = buf[a * B + b]
                    if v != 0.0:
                        iy, iy_valid = _cell_index_from_int(j0 + b, Ny, periodic)
                        if ix_valid and iy_valid:
                            w = wj * v
                            qd.atomic_add(weights[ix, iy], w)
                            for f in range(F):
                                qd.atomic_add(fields[ix, iy, f], quantities[n, f] * w)

        elif eta < eta_crit:
            # --- same, direct per-sample deposition (particle too large for
            # the local buffer) ---
            for s in range(num_samples):
                x_sample = x_phys + sample_coords[s, 0] * hsm_phys
                y_sample = y_phys + sample_coords[s, 1] * hsm_phys

                ix, ix_valid = _cell_index_from_pos(
                    x_sample, cellSize_x_inv, Nx, periodic
                )
                iy, iy_valid = _cell_index_from_pos(
                    y_sample, cellSize_y_inv, Ny, periodic
                )

                if ix_valid and iy_valid:
                    integral = sample_integrals[s]
                    if integral != 0.0:
                        w = wj * integral
                        qd.atomic_add(weights[ix, iy], w)
                        for f in range(F):
                            qd.atomic_add(fields[ix, iy, f], quantities[n, f] * w)

        else:
            # --- cell-quadrature branch, mass-conserving via two-pass renormalization ---
            # cell ceil(x + s) lies entirely outside the (compact) support,
            # so the stencil stops one cell before it
            i_min = i0
            i_max = qd.ceil(x_cell + support_x_cell, int) - 1
            j_min = j0
            j_max = qd.ceil(y_cell + support_y_cell, int) - 1

            if qd.static(not periodic):
                i_min = qd.max(i_min, 0)
                i_max = qd.min(i_max, Nx - 1)
                j_min = qd.max(j_min, 0)
                j_max = qd.min(j_max, Ny - 1)

            # pass 1: total mass over the affected cells (no per-cell storage)
            total_integral = 0.0
            for a in range(i_min, i_max + 1):
                for b in range(j_min, j_max + 1):
                    integral = _isotropic_cell_integral_2d(
                        evaluate_fn,
                        quad_points,
                        dim,
                        a,
                        b,
                        x_cell,
                        y_cell,
                        hsm_phys,
                        kernel_prefactor,
                        cellSize_x,
                        cellSize_y,
                    )
                    total_integral += integral

            if total_integral > 0.0:  # BUGFIX: C++ divides unconditionally
                correction = 1.0 / total_integral

                # pass 2: recompute (cheap) + deposit normalized weight
                for a in range(i_min, i_max + 1):
                    aa = a
                    if qd.static(periodic):
                        aa = _apply_pbc(a, Nx)

                    for b in range(j_min, j_max + 1):
                        bb = b
                        if qd.static(periodic):
                            bb = _apply_pbc(b, Ny)

                        integral = _isotropic_cell_integral_2d(
                            evaluate_fn,
                            quad_points,
                            dim,
                            a,
                            b,
                            x_cell,
                            y_cell,
                            hsm_phys,
                            kernel_prefactor,
                            cellSize_x,
                            cellSize_y,
                        )
                        w_corr = wj * integral * correction

                        qd.atomic_add(weights[aa, bb], w_corr)
                        for f in range(F):
                            qd.atomic_add(fields[aa, bb, f], quantities[n, f] * w_corr)


@qd.kernel
def _isotropic_3d(
    evaluate_fn: qd.template(),
    sigma_fn: qd.template(),
    kernel_support: qd.f32,
    dim: qd.template(),  # only new line in the signature
    quad_points: qd.template(),
    periodic: qd.template(),
    eta_crit: qd.f32,
    B: qd.template(),
    positions: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, 3)
    quantities: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, F)
    smoothing_lengths: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (N,)
    particle_weights: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (N,)
    sample_coords: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (S, 3)
    sample_integrals: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (S,)
    num_samples: qd.i32,
    cellSize_x: qd.f32,
    cellSize_y: qd.f32,
    cellSize_z: qd.f32,
    cellSize_x_inv: qd.f32,
    cellSize_y_inv: qd.f32,
    cellSize_z_inv: qd.f32,
    max_support: qd.f32,
    Nx: qd.i32,
    Ny: qd.i32,
    Nz: qd.i32,
    fields: qd.types.ndarray(dtype=qd.f32, ndim=4),  # (Nx, Ny, Nz, F)
    weights: qd.types.ndarray(dtype=qd.f32, ndim=3),  # (Nx, Ny, Nz)
):
    F = fields.shape[3]
    sig = sigma_fn(dim)

    for n in range(positions.shape[0]):
        x_phys = positions[n, 0]
        y_phys = positions[n, 1]
        z_phys = positions[n, 2]

        hsm_phys = smoothing_lengths[n]
        if qd.static(periodic):
            if hsm_phys > max_support:
                hsm_phys = max_support

        kernel_prefactor = sig / (hsm_phys * hsm_phys * hsm_phys)

        x_cell = x_phys * cellSize_x_inv
        y_cell = y_phys * cellSize_y_inv
        z_cell = z_phys * cellSize_z_inv

        support_phys = kernel_support * hsm_phys
        support_x_cell = support_phys * cellSize_x_inv
        support_y_cell = support_phys * cellSize_y_inv
        support_z_cell = support_phys * cellSize_z_inv

        eta = 2.0 * qd.min(
            support_x_cell, qd.min(support_y_cell, support_z_cell)
        )  # diameter convention
        wj = particle_weights[n]

        i0 = qd.floor(x_cell - support_x_cell, int)
        j0 = qd.floor(y_cell - support_y_cell, int)
        k0 = qd.floor(z_cell - support_z_cell, int)
        fits = (
            (qd.floor(x_cell + support_x_cell, int) - i0 < B)
            and (qd.floor(y_cell + support_y_cell, int) - j0 < B)
            and (qd.floor(z_cell + support_z_cell, int) - k0 < B)
        )

        if eta < eta_crit and fits:
            # --- anti-aliasing branch: deposit via precomputed kernel samples,
            # summed per cell in a local buffer first (one atomic per touched
            # cell instead of one per sample) ---
            buf = qd.Vector.zero(qd.f32, B * B * B)
            for s in range(num_samples):
                x_sample = x_phys + sample_coords[s, 0] * hsm_phys
                y_sample = y_phys + sample_coords[s, 1] * hsm_phys
                z_sample = z_phys + sample_coords[s, 2] * hsm_phys
                li = qd.floor(x_sample * cellSize_x_inv, int) - i0
                lj = qd.floor(y_sample * cellSize_y_inv, int) - j0
                lk = qd.floor(z_sample * cellSize_z_inv, int) - k0
                if 0 <= li < B and 0 <= lj < B and 0 <= lk < B:
                    buf[(li * B + lj) * B + lk] += sample_integrals[s]

            for a in range(B):
                ix, ix_valid = _cell_index_from_int(i0 + a, Nx, periodic)
                for b in range(B):
                    iy, iy_valid = _cell_index_from_int(j0 + b, Ny, periodic)
                    for c in range(B):
                        v = buf[(a * B + b) * B + c]
                        if v != 0.0:
                            iz, iz_valid = _cell_index_from_int(k0 + c, Nz, periodic)
                            if ix_valid and iy_valid and iz_valid:
                                w = wj * v
                                qd.atomic_add(weights[ix, iy, iz], w)
                                for f in range(F):
                                    qd.atomic_add(
                                        fields[ix, iy, iz, f], quantities[n, f] * w
                                    )

        elif eta < eta_crit:
            # --- same, direct per-sample deposition (particle too large for
            # the local buffer) ---
            for s in range(num_samples):
                x_sample = x_phys + sample_coords[s, 0] * hsm_phys
                y_sample = y_phys + sample_coords[s, 1] * hsm_phys
                z_sample = z_phys + sample_coords[s, 2] * hsm_phys

                ix, ix_valid = _cell_index_from_pos(
                    x_sample, cellSize_x_inv, Nx, periodic
                )
                iy, iy_valid = _cell_index_from_pos(
                    y_sample, cellSize_y_inv, Ny, periodic
                )
                iz, iz_valid = _cell_index_from_pos(
                    z_sample, cellSize_z_inv, Nz, periodic
                )

                if ix_valid and iy_valid and iz_valid:
                    integral = sample_integrals[s]
                    if integral != 0.0:
                        w = wj * integral
                        qd.atomic_add(weights[ix, iy, iz], w)
                        for f in range(F):
                            qd.atomic_add(fields[ix, iy, iz, f], quantities[n, f] * w)

        else:
            # --- cell-quadrature branch, mass-conserving via two-pass renormalization ---
            # cell ceil(x + s) lies entirely outside the (compact) support,
            # so the stencil stops one cell before it
            i_min = i0
            i_max = qd.ceil(x_cell + support_x_cell, int) - 1
            j_min = j0
            j_max = qd.ceil(y_cell + support_y_cell, int) - 1
            k_min = k0
            k_max = qd.ceil(z_cell + support_z_cell, int) - 1

            if qd.static(not periodic):
                i_min = qd.max(i_min, 0)
                i_max = qd.min(i_max, Nx - 1)
                j_min = qd.max(j_min, 0)
                j_max = qd.min(j_max, Ny - 1)
                k_min = qd.max(k_min, 0)
                k_max = qd.min(k_max, Nz - 1)

            # pass 1: total mass over the affected cells (no per-cell storage)
            total_integral = 0.0
            for a in range(i_min, i_max + 1):
                for b in range(j_min, j_max + 1):
                    for c in range(k_min, k_max + 1):
                        integral = _isotropic_cell_integral_3d(
                            evaluate_fn,
                            quad_points,
                            dim,
                            a,
                            b,
                            c,
                            x_cell,
                            y_cell,
                            z_cell,
                            hsm_phys,
                            kernel_prefactor,
                            cellSize_x,
                            cellSize_y,
                            cellSize_z,
                        )
                        total_integral += integral

            if total_integral > 0.0:  # BUGFIX: C++ divides unconditionally
                correction = 1.0 / total_integral

                # pass 2: recompute (cheap) + deposit normalized weight
                for a in range(i_min, i_max + 1):
                    aa = a
                    if qd.static(periodic):
                        aa = _apply_pbc(a, Nx)

                    for b in range(j_min, j_max + 1):
                        bb = b
                        if qd.static(periodic):
                            bb = _apply_pbc(b, Ny)

                        for c in range(k_min, k_max + 1):
                            cc = c
                            if qd.static(periodic):
                                cc = _apply_pbc(c, Nz)

                            integral = _isotropic_cell_integral_3d(
                                evaluate_fn,
                                quad_points,
                                dim,
                                a,
                                b,
                                c,
                                x_cell,
                                y_cell,
                                z_cell,
                                hsm_phys,
                                kernel_prefactor,
                                cellSize_x,
                                cellSize_y,
                                cellSize_z,
                            )
                            w_corr = wj * integral * correction

                            qd.atomic_add(weights[aa, bb, cc], w_corr)
                            for f in range(F):
                                qd.atomic_add(
                                    fields[aa, bb, cc, f], quantities[n, f] * w_corr
                                )


# =============================================================================
# Covariant Kernels
# =============================================================================
@qd.kernel
def _covariant_1d(
    evaluate_fn: qd.template(),
    sigma_fn: qd.template(),
    kernel_support: qd.f32,
    dim: qd.template(),  # only new line in the signature
    quad_points: qd.template(),
    periodic: qd.template(),
    eta_crit: qd.f32,
    B: qd.template(),
    positions: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N,1)
    quantities: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N,F)
    hmat_eigvecs: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N,1)
    hmat_eigvals: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N,1)
    particle_weights: qd.types.ndarray(dtype=qd.f32, ndim=1),
    sample_coords: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (S,1)
    sample_integrals: qd.types.ndarray(dtype=qd.f32, ndim=1),
    num_samples: qd.i32,
    cellSize_x: qd.f32,
    cellSize_x_inv: qd.f32,
    max_support: qd.f32,
    Nx: qd.i32,
    fields: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (Nx,F)
    weights: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (Nx,)
):
    F = fields.shape[1]
    sig = sigma_fn(dim)

    for n in range(positions.shape[0]):

        x_phys = positions[n, 0]

        # always 1 in 1D, but kept for API symmetry
        e0 = hmat_eigvecs[n, 0]

        eval0 = hmat_eigvals[n, 0]

        if qd.static(periodic):
            if eval0 > max_support:
                eval0 = max_support

        eval0_cell = eval0 * cellSize_x_inv

        x_cell = x_phys * cellSize_x_inv
        support_x_cell = kernel_support * qd.abs(eval0_cell)

        eta = 2.0 * support_x_cell  # diameter convention
        wj = particle_weights[n]

        i0 = qd.floor(x_cell - support_x_cell, int)
        fits = qd.floor(x_cell + support_x_cell, int) - i0 < B

        if eta < eta_crit and fits:
            # sample deposition summed per cell in a local buffer first (one
            # atomic per touched cell instead of one per sample)
            buf = qd.Vector.zero(qd.f32, B)
            for s in range(num_samples):
                x_sample = x_phys + e0 * (eval0 * sample_coords[s, 0])
                li = qd.floor(x_sample * cellSize_x_inv, int) - i0
                if 0 <= li < B:
                    buf[li] += sample_integrals[s]

            for a in range(B):
                v = buf[a]
                if v != 0.0:
                    ix, ix_valid = _cell_index_from_int(i0 + a, Nx, periodic)
                    if ix_valid:
                        w = wj * v
                        qd.atomic_add(weights[ix], w)
                        for f in range(F):
                            qd.atomic_add(fields[ix, f], quantities[n, f] * w)

        elif eta < eta_crit:

            for s in range(num_samples):

                cx = sample_coords[s, 0]

                x_sample = x_phys + e0 * (eval0 * cx)

                ix, valid = _cell_index_from_pos(
                    x_sample,
                    cellSize_x_inv,
                    Nx,
                    periodic,
                )

                if valid:
                    integral = sample_integrals[s]
                    if integral != 0.0:
                        w = wj * integral

                        qd.atomic_add(weights[ix], w)

                        for f in range(F):
                            qd.atomic_add(
                                fields[ix, f],
                                quantities[n, f] * w,
                            )

        else:
            detH = eval0
            kernel_prefactor = sig / detH

            # cell ceil(x + s) lies entirely outside the (compact) support
            i_min = i0
            i_max = qd.ceil(x_cell + support_x_cell, int) - 1

            if qd.static(not periodic):
                i_min = qd.max(i_min, 0)
                i_max = qd.min(i_max, Nx - 1)

            total_integral = 0.0

            for a in range(i_min, i_max + 1):

                integral = _covariant_cell_integral_1d(
                    evaluate_fn,
                    quad_points,
                    dim,
                    a,
                    x_cell,
                    e0,
                    eval0,
                    kernel_prefactor,
                    cellSize_x,
                )

                total_integral += integral

            if total_integral > 0.0:

                correction = 1.0 / total_integral

                for a in range(i_min, i_max + 1):

                    aa = a
                    if qd.static(periodic):
                        aa = _apply_pbc(a, Nx)

                    integral = _covariant_cell_integral_1d(
                        evaluate_fn,
                        quad_points,
                        dim,
                        a,
                        x_cell,
                        e0,
                        eval0,
                        kernel_prefactor,
                        cellSize_x,
                    )

                    w_corr = wj * integral * correction

                    qd.atomic_add(weights[aa], w_corr)

                    for f in range(F):
                        qd.atomic_add(
                            fields[aa, f],
                            quantities[n, f] * w_corr,
                        )


@qd.kernel
def _covariant_2d(
    evaluate_fn: qd.template(),
    sigma_fn: qd.template(),
    kernel_support: qd.f32,
    dim: qd.template(),  # only new line in the signature
    quad_points: qd.template(),
    periodic: qd.template(),
    eta_crit: qd.f32,
    B: qd.template(),
    positions: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N,2)
    quantities: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N,F)
    hmat_eigvecs: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N,4) column-major
    hmat_eigvals: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N,2)
    particle_weights: qd.types.ndarray(dtype=qd.f32, ndim=1),
    sample_coords: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (S,2)
    sample_integrals: qd.types.ndarray(dtype=qd.f32, ndim=1),
    num_samples: qd.i32,
    cellSize_x: qd.f32,
    cellSize_y: qd.f32,
    cellSize_x_inv: qd.f32,
    cellSize_y_inv: qd.f32,
    max_support: qd.f32,
    Nx: qd.i32,
    Ny: qd.i32,
    fields: qd.types.ndarray(dtype=qd.f32, ndim=3),  # (Nx,Ny,F)
    weights: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (Nx,Ny)
):
    F = fields.shape[2]
    sig = sigma_fn(dim)

    for n in range(positions.shape[0]):

        x_phys = positions[n, 0]
        y_phys = positions[n, 1]

        # column-major
        e0x = hmat_eigvecs[n, 0]
        e0y = hmat_eigvecs[n, 1]

        e1x = hmat_eigvecs[n, 2]
        e1y = hmat_eigvecs[n, 3]

        eval0 = hmat_eigvals[n, 0]
        eval1 = hmat_eigvals[n, 1]

        if qd.static(periodic):
            max_eval = qd.max(eval0, eval1)
            if max_eval > max_support:
                scale = max_support / max_eval
                eval0 *= scale
                eval1 *= scale

        eval0_cell = eval0 * cellSize_x_inv
        eval1_cell = eval1 * cellSize_y_inv

        support_x_cell = kernel_support * qd.sqrt(
            (e0x * eval0_cell) ** 2 + (e1x * eval1_cell) ** 2
        )
        support_y_cell = kernel_support * qd.sqrt(
            (e0y * eval0_cell) ** 2 + (e1y * eval1_cell) ** 2
        )

        eta = 2.0 * qd.min(support_x_cell, support_y_cell)  # diameter convention

        # A degenerate ellipsoid (any eigenvalue exactly 0 -- possible with
        # few neighbors, e.g. near-coplanar configurations) makes
        # detH = eval0*eval1 = 0 in the quadrature branch below, which would
        # divide by zero. eta itself doesn't reliably flag this (it's driven
        # by the largest eigenvalue per axis, not the product), so check
        # explicitly and force the (division-free) sample branch instead.
        degenerate = eval0 <= 0.0 or eval1 <= 0.0

        wj = particle_weights[n]

        x_cell = x_phys * cellSize_x_inv
        y_cell = y_phys * cellSize_y_inv
        i0 = qd.floor(x_cell - support_x_cell, int)
        j0 = qd.floor(y_cell - support_y_cell, int)
        fits = (qd.floor(x_cell + support_x_cell, int) - i0 < B) and (
            qd.floor(y_cell + support_y_cell, int) - j0 < B
        )
        use_samples = eta < eta_crit or degenerate

        # cell ceil(x + s) lies entirely outside the (compact) support
        i_min = i0
        i_max = qd.ceil(x_cell + support_x_cell, int) - 1
        j_min = j0
        j_max = qd.ceil(y_cell + support_y_cell, int) - 1
        if qd.static(not periodic):
            i_min = qd.max(i_min, 0)
            i_max = qd.min(i_max, Nx - 1)
            j_min = qd.max(j_min, 0)
            j_max = qd.min(j_max, Ny - 1)

        # Quadrature pass 1 (total kernel weight over the affected cells) is
        # evaluated before the branch decision: a strongly elongated
        # ellipse can span several cells along the grid axes (eta >=
        # eta_crit) while being thinner than a cell across its minor axis,
        # so that no quadrature node falls inside its support. Its total is
        # then 0 and, instead of silently dropping the particle's mass, it
        # is deposited via the (normalized) sample table.
        kernel_prefactor = 0.0
        total_integral = 0.0
        if not use_samples:
            detH = eval0 * eval1
            kernel_prefactor = sig / detH
            for a in range(i_min, i_max + 1):
                for b in range(j_min, j_max + 1):
                    total_integral += _covariant_cell_integral_2d(
                        evaluate_fn,
                        quad_points,
                        dim,
                        a,
                        b,
                        x_cell,
                        y_cell,
                        e0x,
                        e0y,
                        e1x,
                        e1y,
                        eval0,
                        eval1,
                        kernel_prefactor,
                        cellSize_x,
                        cellSize_y,
                    )
            if total_integral <= 0.0:
                use_samples = True

        if use_samples and fits:
            # sample deposition summed per cell in a local buffer first (one
            # atomic per touched cell instead of one per sample)
            buf = qd.Vector.zero(qd.f32, B * B)
            for s in range(num_samples):
                cx = sample_coords[s, 0]
                cy = sample_coords[s, 1]

                x_sample = x_phys + e0x * (eval0 * cx) + e1x * (eval1 * cy)
                y_sample = y_phys + e0y * (eval0 * cx) + e1y * (eval1 * cy)

                li = qd.floor(x_sample * cellSize_x_inv, int) - i0
                lj = qd.floor(y_sample * cellSize_y_inv, int) - j0
                if 0 <= li < B and 0 <= lj < B:
                    buf[li * B + lj] += sample_integrals[s]

            for a in range(B):
                ix, ix_valid = _cell_index_from_int(i0 + a, Nx, periodic)
                for b in range(B):
                    v = buf[a * B + b]
                    if v != 0.0:
                        iy, iy_valid = _cell_index_from_int(j0 + b, Ny, periodic)
                        if ix_valid and iy_valid:
                            w = wj * v
                            qd.atomic_add(weights[ix, iy], w)
                            for f in range(F):
                                qd.atomic_add(fields[ix, iy, f], quantities[n, f] * w)

        elif use_samples:

            for s in range(num_samples):

                cx = sample_coords[s, 0]
                cy = sample_coords[s, 1]

                x_sample = x_phys + e0x * (eval0 * cx) + e1x * (eval1 * cy)
                y_sample = y_phys + e0y * (eval0 * cx) + e1y * (eval1 * cy)

                ix, ix_valid = _cell_index_from_pos(
                    x_sample, cellSize_x_inv, Nx, periodic
                )
                iy, iy_valid = _cell_index_from_pos(
                    y_sample, cellSize_y_inv, Ny, periodic
                )

                if ix_valid and iy_valid:
                    integral = sample_integrals[s]
                    if integral != 0.0:
                        w = wj * integral
                        qd.atomic_add(weights[ix, iy], w)

                        for f in range(F):
                            qd.atomic_add(
                                fields[ix, iy, f],
                                quantities[n, f] * w,
                            )

        else:
            # cell-quadrature branch, mass-conserving via two-pass
            # renormalization (pass 1 computed above)
            if total_integral > 0.0:

                correction = 1.0 / total_integral

                for a in range(i_min, i_max + 1):

                    aa = a
                    if qd.static(periodic):
                        aa = _apply_pbc(a, Nx)

                    for b in range(j_min, j_max + 1):

                        bb = b
                        if qd.static(periodic):
                            bb = _apply_pbc(b, Ny)

                        integral = _covariant_cell_integral_2d(
                            evaluate_fn,
                            quad_points,
                            dim,
                            a,
                            b,
                            x_cell,
                            y_cell,
                            e0x,
                            e0y,
                            e1x,
                            e1y,
                            eval0,
                            eval1,
                            kernel_prefactor,
                            cellSize_x,
                            cellSize_y,
                        )

                        w_corr = wj * integral * correction

                        qd.atomic_add(weights[aa, bb], w_corr)

                        for f in range(F):
                            qd.atomic_add(
                                fields[aa, bb, f],
                                quantities[n, f] * w_corr,
                            )


@qd.kernel
def _covariant_3d(
    evaluate_fn: qd.template(),
    sigma_fn: qd.template(),
    kernel_support: qd.f32,
    dim: qd.template(),  # only new line in the signature
    quad_points: qd.template(),
    periodic: qd.template(),
    eta_crit: qd.f32,
    B: qd.template(),
    positions: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, 3)
    quantities: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, F)
    hmat_eigvecs: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, 9) column-major
    hmat_eigvals: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (N, 3)
    particle_weights: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (N,)
    sample_coords: qd.types.ndarray(dtype=qd.f32, ndim=2),  # (S, 3)
    sample_integrals: qd.types.ndarray(dtype=qd.f32, ndim=1),  # (S,)
    num_samples: qd.i32,
    cellSize_x: qd.f32,
    cellSize_y: qd.f32,
    cellSize_z: qd.f32,
    cellSize_x_inv: qd.f32,
    cellSize_y_inv: qd.f32,
    cellSize_z_inv: qd.f32,
    max_support: qd.f32,
    Nx: qd.i32,
    Ny: qd.i32,
    Nz: qd.i32,
    fields: qd.types.ndarray(dtype=qd.f32, ndim=4),  # (Nx, Ny, Nz, F)
    weights: qd.types.ndarray(dtype=qd.f32, ndim=3),  # (Nx, Ny, Nz)
):
    F = fields.shape[3]
    sig = sigma_fn(dim)

    for n in range(positions.shape[0]):
        x_phys = positions[n, 0]
        y_phys = positions[n, 1]
        z_phys = positions[n, 2]

        # column-major: column j is eigenvector j
        e0x = hmat_eigvecs[n, 0]
        e0y = hmat_eigvecs[n, 1]
        e0z = hmat_eigvecs[n, 2]

        e1x = hmat_eigvecs[n, 3]
        e1y = hmat_eigvecs[n, 4]
        e1z = hmat_eigvecs[n, 5]

        e2x = hmat_eigvecs[n, 6]
        e2y = hmat_eigvecs[n, 7]
        e2z = hmat_eigvecs[n, 8]

        eval0 = hmat_eigvals[n, 0]
        eval1 = hmat_eigvals[n, 1]
        eval2 = hmat_eigvals[n, 2]

        if qd.static(periodic):
            max_eval = qd.max(eval0, qd.max(eval1, eval2))
            if max_eval > max_support:
                scale = max_support / max_eval
                eval0 *= scale
                eval1 *= scale
                eval2 *= scale

        # needed for eta (branch decision) *and* reused in the else
        # branch's support_*_cell formulas -- computed once either way
        eval0_cell = eval0 * cellSize_x_inv
        eval1_cell = eval1 * cellSize_y_inv
        eval2_cell = eval2 * cellSize_z_inv

        support_x_cell = kernel_support * qd.sqrt(
            (e0x * eval0_cell) ** 2 + (e1x * eval1_cell) ** 2 + (e2x * eval2_cell) ** 2
        )
        support_y_cell = kernel_support * qd.sqrt(
            (e0y * eval0_cell) ** 2 + (e1y * eval1_cell) ** 2 + (e2y * eval2_cell) ** 2
        )
        support_z_cell = kernel_support * qd.sqrt(
            (e0z * eval0_cell) ** 2 + (e1z * eval1_cell) ** 2 + (e2z * eval2_cell) ** 2
        )

        eta = 2.0 * qd.min(
            support_x_cell, qd.min(support_y_cell, support_z_cell)
        )  # diameter convention

        # A degenerate ellipsoid (any eigenvalue exactly 0 -- possible with
        # few neighbors, e.g. near-coplanar configurations) makes
        # detH = eval0*eval1*eval2 = 0 in the quadrature branch below, which
        # would divide by zero. eta itself doesn't reliably flag this (it's
        # driven by the largest eigenvalue per axis, not the product), so
        # check explicitly and force the (division-free) sample branch.
        degenerate = eval0 <= 0.0 or eval1 <= 0.0 or eval2 <= 0.0
        wj = particle_weights[n]

        x_cell = x_phys * cellSize_x_inv
        y_cell = y_phys * cellSize_y_inv
        z_cell = z_phys * cellSize_z_inv
        i0 = qd.floor(x_cell - support_x_cell, int)
        j0 = qd.floor(y_cell - support_y_cell, int)
        k0 = qd.floor(z_cell - support_z_cell, int)
        fits = (
            (qd.floor(x_cell + support_x_cell, int) - i0 < B)
            and (qd.floor(y_cell + support_y_cell, int) - j0 < B)
            and (qd.floor(z_cell + support_z_cell, int) - k0 < B)
        )
        use_samples = eta < eta_crit or degenerate

        # cell ceil(x + s) lies entirely outside the (compact) support, so
        # the stencil stops one cell before it
        i_min = i0
        i_max = qd.ceil(x_cell + support_x_cell, int) - 1
        j_min = j0
        j_max = qd.ceil(y_cell + support_y_cell, int) - 1
        k_min = k0
        k_max = qd.ceil(z_cell + support_z_cell, int) - 1
        if qd.static(not periodic):
            i_min = qd.max(i_min, 0)
            i_max = qd.min(i_max, Nx - 1)
            j_min = qd.max(j_min, 0)
            j_max = qd.min(j_max, Ny - 1)
            k_min = qd.max(k_min, 0)
            k_max = qd.min(k_max, Nz - 1)

        # Quadrature pass 1 (total kernel weight over the affected cells) is
        # evaluated before the branch decision: a strongly elongated
        # ellipsoid can span several cells along every grid axis (eta >=
        # eta_crit) while being thinner than a cell across its minor axis,
        # so that no quadrature node falls inside its support. Its total is
        # then 0 and, instead of silently dropping the particle's mass, it
        # is deposited via the (normalized) sample table.
        kernel_prefactor = 0.0
        total_integral = 0.0
        if not use_samples:
            detH = eval0 * eval1 * eval2
            kernel_prefactor = sig / detH
            for a in range(i_min, i_max + 1):
                for b in range(j_min, j_max + 1):
                    for c in range(k_min, k_max + 1):
                        total_integral += _covariant_cell_integral_3d(
                            evaluate_fn,
                            quad_points,
                            dim,
                            a,
                            b,
                            c,
                            x_cell,
                            y_cell,
                            z_cell,
                            e0x,
                            e0y,
                            e0z,
                            e1x,
                            e1y,
                            e1z,
                            e2x,
                            e2y,
                            e2z,
                            eval0,
                            eval1,
                            eval2,
                            kernel_prefactor,
                            cellSize_x,
                            cellSize_y,
                            cellSize_z,
                        )
            if total_integral <= 0.0:
                use_samples = True

        if use_samples and fits:
            # --- anti-aliasing branch: deposit via precomputed kernel samples,
            # summed per cell in a local buffer first (one atomic per touched
            # cell instead of one per sample) ---
            buf = qd.Vector.zero(qd.f32, B * B * B)
            for s in range(num_samples):
                cx = sample_coords[s, 0]
                cy = sample_coords[s, 1]
                cz = sample_coords[s, 2]

                # map the isotropic unit-sphere sample into the
                # particle's physical ellipsoid: R diag(evals) c
                x_sample = (
                    x_phys
                    + e0x * (eval0 * cx)
                    + e1x * (eval1 * cy)
                    + e2x * (eval2 * cz)
                )
                y_sample = (
                    y_phys
                    + e0y * (eval0 * cx)
                    + e1y * (eval1 * cy)
                    + e2y * (eval2 * cz)
                )
                z_sample = (
                    z_phys
                    + e0z * (eval0 * cx)
                    + e1z * (eval1 * cy)
                    + e2z * (eval2 * cz)
                )

                li = qd.floor(x_sample * cellSize_x_inv, int) - i0
                lj = qd.floor(y_sample * cellSize_y_inv, int) - j0
                lk = qd.floor(z_sample * cellSize_z_inv, int) - k0
                if 0 <= li < B and 0 <= lj < B and 0 <= lk < B:
                    buf[(li * B + lj) * B + lk] += sample_integrals[s]

            for a in range(B):
                ix, ix_valid = _cell_index_from_int(i0 + a, Nx, periodic)
                for b in range(B):
                    iy, iy_valid = _cell_index_from_int(j0 + b, Ny, periodic)
                    for c in range(B):
                        v = buf[(a * B + b) * B + c]
                        if v != 0.0:
                            iz, iz_valid = _cell_index_from_int(k0 + c, Nz, periodic)
                            if ix_valid and iy_valid and iz_valid:
                                w = wj * v
                                qd.atomic_add(weights[ix, iy, iz], w)
                                for f in range(F):
                                    qd.atomic_add(
                                        fields[ix, iy, iz, f], quantities[n, f] * w
                                    )

        elif use_samples:
            # --- same, direct per-sample deposition (particle too large for
            # the local buffer) ---
            for s in range(num_samples):
                cx = sample_coords[s, 0]
                cy = sample_coords[s, 1]
                cz = sample_coords[s, 2]

                # map the isotropic unit-sphere sample into the
                # particle's physical ellipsoid: R diag(evals) c
                x_sample = (
                    x_phys
                    + e0x * (eval0 * cx)
                    + e1x * (eval1 * cy)
                    + e2x * (eval2 * cz)
                )
                y_sample = (
                    y_phys
                    + e0y * (eval0 * cx)
                    + e1y * (eval1 * cy)
                    + e2y * (eval2 * cz)
                )
                z_sample = (
                    z_phys
                    + e0z * (eval0 * cx)
                    + e1z * (eval1 * cy)
                    + e2z * (eval2 * cz)
                )

                ix, ix_valid = _cell_index_from_pos(
                    x_sample, cellSize_x_inv, Nx, periodic
                )
                iy, iy_valid = _cell_index_from_pos(
                    y_sample, cellSize_y_inv, Ny, periodic
                )
                iz, iz_valid = _cell_index_from_pos(
                    z_sample, cellSize_z_inv, Nz, periodic
                )

                if ix_valid and iy_valid and iz_valid:
                    integral = sample_integrals[s]
                    if integral != 0.0:
                        w = wj * integral
                        qd.atomic_add(weights[ix, iy, iz], w)
                        for f in range(F):
                            qd.atomic_add(fields[ix, iy, iz, f], quantities[n, f] * w)

        else:
            # --- cell-quadrature branch, mass-conserving via two-pass
            # renormalization (pass 1 computed above) ---
            if total_integral > 0.0:  # BUGFIX: C++ divides unconditionally
                correction = 1.0 / total_integral

                # pass 2: recompute (cheap) + deposit normalized weight
                for a in range(i_min, i_max + 1):
                    aa = a
                    if qd.static(periodic):
                        aa = _apply_pbc(a, Nx)

                    for b in range(j_min, j_max + 1):
                        bb = b
                        if qd.static(periodic):
                            bb = _apply_pbc(b, Ny)

                        for c in range(k_min, k_max + 1):
                            cc = c
                            if qd.static(periodic):
                                cc = _apply_pbc(c, Nz)

                            integral = _covariant_cell_integral_3d(
                                evaluate_fn,
                                quad_points,
                                dim,
                                a,
                                b,
                                c,
                                x_cell,
                                y_cell,
                                z_cell,
                                e0x,
                                e0y,
                                e0z,
                                e1x,
                                e1y,
                                e1z,
                                e2x,
                                e2y,
                                e2z,
                                eval0,
                                eval1,
                                eval2,
                                kernel_prefactor,
                                cellSize_x,
                                cellSize_y,
                                cellSize_z,
                            )
                            w_corr = wj * integral * correction

                            qd.atomic_add(weights[aa, bb, cc], w_corr)
                            for f in range(F):
                                qd.atomic_add(
                                    fields[aa, bb, cc, f], quantities[n, f] * w_corr
                                )


# =============================================================================
# ||                                                                         ||
# ||                               WRAPPERS                                  ||
# ||                                                                         ||
# =============================================================================
_KERNELS = {
    False: _FIXED_GRID_KERNELS,  # static (fixed-stencil) methods
    True: {  # adaptive (smoothing-length-dependent) methods
        ("separable", 1): _separable_1d,
        ("separable", 2): _separable_2d,
        ("separable", 3): _separable_3d,
        ("isotropic", 1): _isotropic_1d,
        ("isotropic", 2): _isotropic_2d,
        ("isotropic", 3): _isotropic_3d,
        ("covariant", 1): _covariant_1d,
        ("covariant", 2): _covariant_2d,
        ("covariant", 3): _covariant_3d,
    },
}


@dataclass(slots=True)
class KernelPreparation:
    """Everything needed to run one adaptive deposition, plus enough
    derived/introspectable state (cell sizes, stencil/sample counts, ...)
    to show a front-end what was actually computed. `kernel` + `args` are
    what actually gets called; the rest of the fields exist for
    inspection/debugging, not for driving the call themselves.
    """

    # -------------------------------------------------------------------
    # What to call
    # -------------------------------------------------------------------
    kernel: Any  # the @qd.kernel to invoke
    args: tuple  # exact positional args for `kernel`

    # -------------------------------------------------------------------
    # Output arrays (also embedded in `args`; kept here too since Quadrants
    # fills ndarrays in place -- these are the objects to hand back)
    # -------------------------------------------------------------------
    grid_fields: np.ndarray
    grid_weights: np.ndarray

    # -------------------------------------------------------------------
    # Derived quantities, for introspection / front-end display only
    # -------------------------------------------------------------------
    structure: str = ""
    dim: int = 0
    kernel_name: str = ""
    support: float = 0.0
    periodic: bool = False
    cell_sizes: tuple = ()
    max_support: float = 0.0
    max_stencil: int | None = None  # separable only
    num_samples: int | None = None  # isotropic / covariant only
    eta_crit: float | None = None  # isotropic / covariant only
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

    max_stencil = _round_up_stencil(
        max(
            int(np.ceil(2 * support * particle_hsml[:, d].max() * cell_size_invs[d]))
            + 2
            for d in range(DIM)
        )
    )

    F = particle_fields.shape[1]
    grid_fields = np.zeros((*gridnums, F), dtype=np.float32)
    grid_weights = np.zeros(gridnums, dtype=np.float32)

    kernel = _KERNELS[True][("separable", DIM)]
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
        *cell_size_invs,
        max_support,
        max_stencil,
        *gridnums,
        grid_fields,
        grid_weights,
    )

    return KernelPreparation(
        kernel=kernel,
        args=args,
        grid_fields=grid_fields,
        grid_weights=grid_weights,
        structure="separable",
        kernel_name=kernel_name,
        support=support,
        periodic=bool(periodic),
        cell_sizes=tuple(1.0 / c for c in cell_size_invs),
        max_support=float(max_support),
        max_stencil=int(max_stencil),
    )


def _prepare_isotropic(
    particle_positions,
    particle_fields,
    particle_weights,
    boxsizes,
    gridnums,
    periodic,
    kernel_name,
    particle_hsml,  # (N,) -- single smoothing length
    integration_method,
    eta_crit,
):
    DIM = particle_positions.shape[1]
    particle_hsml = as_f32(particle_hsml)

    gridnums = tuple(int(n) for n in gridnums)
    boxsizes = tuple(float(b) for b in boxsizes)

    kspec = create_spherical_kernel(kernel_name)
    support = kspec.support

    quad_table = {
        1: QUADRATURE_POINTS_1D,
        2: QUADRATURE_POINTS_2D,
        3: QUADRATURE_POINTS_3D,
    }[DIM]
    if integration_method not in quad_table:
        raise ValueError(
            f"Unknown integration method: {integration_method!r}. "
            f"Available: {sorted(quad_table.keys())}"
        )
    quad_points = quad_table[integration_method]

    grid = build_kernel_sample_grid(kernel_name, DIM, eta_crit)
    sample_buffer = _sample_buffer_size(eta_crit, DIM)
    sample_coords, sample_integrals, num_samples = (
        grid["coords"],
        grid["integrals"],
        grid["count"],
    )

    cell_sizes = tuple(boxsizes[d] / gridnums[d] for d in range(DIM))
    cell_size_invs = tuple(1.0 / c for c in cell_sizes)
    max_support = min(0.5 * boxsizes[d] / support for d in range(DIM))

    F = particle_fields.shape[1]
    grid_fields = np.zeros((*gridnums, F), dtype=np.float32)
    grid_weights = np.zeros(gridnums, dtype=np.float32)

    kernel = _KERNELS[True][("isotropic", DIM)]
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
        particle_hsml,
        particle_weights,
        sample_coords,
        sample_integrals,
        num_samples,
        *cell_sizes,
        *cell_size_invs,
        max_support,
        *gridnums,
        grid_fields,
        grid_weights,
    )

    return KernelPreparation(
        kernel=kernel,
        args=args,
        grid_fields=grid_fields,
        grid_weights=grid_weights,
        structure="isotropic",
        kernel_name=kernel_name,
        support=support,
        periodic=bool(periodic),
        cell_sizes=cell_sizes,
        max_support=float(max_support),
        num_samples=int(num_samples),
        eta_crit=float(eta_crit),
        sample_buffer=int(sample_buffer),
    )


def _prepare_covariant(
    particle_positions,
    particle_fields,
    particle_weights,
    boxsizes,
    gridnums,
    periodic,
    kernel_name,
    particle_hmat_eigvecs,  # (N, D, D)
    particle_hmat_eigvals,  # (N, D)
    integration_method,
    eta_crit,
):
    DIM = particle_positions.shape[1]

    # qd.kernels read the flattened matrix as [e0x,e0y,e0z, e1x,e1y,e1z, ...]
    # i.e. column j (eigenvector j) contiguous -- transpose(0, 2, 1) turns
    # (N, row, col) into (N, col, row) so reshape(-1, D*D) lays out each
    # column contiguously.
    particle_hmat_eigvecs = as_f32(
        particle_hmat_eigvecs.transpose(0, 2, 1).reshape(-1, DIM * DIM)
    )
    particle_hmat_eigvals = as_f32(particle_hmat_eigvals)

    gridnums = tuple(int(n) for n in gridnums)
    boxsizes = tuple(float(b) for b in boxsizes)

    kspec = create_spherical_kernel(kernel_name)
    support = kspec.support

    quad_table = {
        1: QUADRATURE_POINTS_1D,
        2: QUADRATURE_POINTS_2D,
        3: QUADRATURE_POINTS_3D,
    }[DIM]
    if integration_method not in quad_table:
        raise ValueError(
            f"Unknown integration method: {integration_method!r}. "
            f"Available: {sorted(quad_table.keys())}"
        )
    quad_points = quad_table[integration_method]

    grid = build_kernel_sample_grid(kernel_name, DIM, eta_crit)
    sample_buffer = _sample_buffer_size(eta_crit, DIM)
    sample_coords, sample_integrals, num_samples = (
        grid["coords"],
        grid["integrals"],
        grid["count"],
    )

    cell_sizes = tuple(boxsizes[d] / gridnums[d] for d in range(DIM))
    cell_size_invs = tuple(1.0 / c for c in cell_sizes)
    max_support = min(0.5 * boxsizes[d] / support for d in range(DIM))

    F = particle_fields.shape[1]
    grid_fields = np.zeros((*gridnums, F), dtype=np.float32)
    grid_weights = np.zeros(gridnums, dtype=np.float32)

    kernel = _KERNELS[True][("covariant", DIM)]
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
        particle_hmat_eigvecs,
        particle_hmat_eigvals,
        particle_weights,
        sample_coords,
        sample_integrals,
        num_samples,
        *cell_sizes,
        *cell_size_invs,
        max_support,
        *gridnums,
        grid_fields,
        grid_weights,
    )

    return KernelPreparation(
        kernel=kernel,
        args=args,
        grid_fields=grid_fields,
        grid_weights=grid_weights,
        structure="covariant",
        kernel_name=kernel_name,
        support=support,
        periodic=bool(periodic),
        cell_sizes=cell_sizes,
        max_support=float(max_support),
        num_samples=int(num_samples),
        eta_crit=float(eta_crit),
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
    """Deposit particle data onto a grid.

    Parameters
    ----------
    particle_positions, particle_fields, particle_weights, boxsizes, gridnums, periodic
        Always required.
    adaptive : bool
        Select fixed-grid or smoothing-length-dependent deposition.
    kernel_name : str
        A fixed-grid stencil when ``adaptive=False``; otherwise a kernel from
        the selected adaptive structure's family.
    structure : {"separable", "isotropic", "covariant"}, optional
        Required when ``adaptive=True``:

        - fixed-grid (``adaptive=False``): `kernel_name` is one of
          ``"ngp"``, ``"cic"``, ``"tsc"``, ``"bspline4"``, ``"bspline5"``,
          ``"bspline6"``.
        - separable: needs `particle_hsml` (N, D) and one of the separable
          kernels ``"tophat"``, ``"tsc"``, ``"gaussian"``. Integrated
          exactly; `integration_method` is ignored.
        - isotropic: needs `particle_hsml` (N,), a radial kernel such as
          ``"cubic_spline"``, `integration_method`, `eta_crit`.
        - covariant: needs `particle_hmat_eigvecs` (N, D, D),
          `particle_hmat_eigvals` (N, D), plus the same kernel/quadrature
          options as ``"isotropic"``.

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

    if not adaptive:
        check_fixed_grid_kernel(kernel_name, structure)
        key = (kernel_name, dim)
        try:
            kernel = _KERNELS[False][key]
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
    elif structure == "isotropic":
        prep = _prepare_isotropic(
            **common,
            particle_hsml=particle_hsml,
            integration_method=integration_method,
            eta_crit=eta_crit,
        )
    else:  # "covariant"
        prep = _prepare_covariant(
            **common,
            particle_hmat_eigvecs=particle_hmat_eigvecs,
            particle_hmat_eigvals=particle_hmat_eigvals,
            integration_method=integration_method,
            eta_crit=eta_crit,
        )

    prep.kernel(*prep.args)
    return np.moveaxis(prep.grid_fields, -1, 0), prep.grid_weights
