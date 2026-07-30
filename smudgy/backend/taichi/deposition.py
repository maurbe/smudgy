from dataclasses import dataclass
from typing import Any

import numpy as np
import taichi as ti

from .kernels import (
    _separable_integrate_1d,
    build_kernel_sample_grid,
    create_separable_kernel,
    create_spherical_kernel,
)


# =============================================================================
# Shared helpers
# =============================================================================
@ti.func
def _apply_pbc(idx: ti.i32, gridnum: ti.i32) -> ti.i32:
    r = idx % gridnum
    if r < 0:
        r += gridnum
    return r


@ti.func
def _cell_index_from_pos(
    pos: ti.f32, cellSize_inv: ti.f32, gridnum: ti.i32, periodic: ti.template()
):
    """Returns (index, valid). `valid` is always True when periodic
    (wrapped via modulo); for non-periodic it's False when pos falls
    outside the grid -- mirrors the C++ optional-returning
    cell_index_from_pos, minus the std::optional machinery.
    """
    idx = ti.floor(pos * cellSize_inv, int)
    valid = True
    if ti.static(periodic):
        idx = _apply_pbc(idx, gridnum)
    else:
        if idx < 0 or idx >= gridnum:
            valid = False
    return idx, valid


def _as_float32(array):
    """Return a C-contiguous float32 array. No copy if `array` is
    already float32 and C-contiguous; copies otherwise.
    """
    return np.ascontiguousarray(array, dtype=np.float32)


def _round_up_stencil(n, bucket=8):
    """Round the stencil size up to the nearest multiple of `bucket`.
    Keeps MAX_STENCIL (a ti.template() arg) taking only a handful of
    distinct values across a run, so Taichi reuses one compiled kernel
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


@ti.func
def _isotropic_cell_integral_1d(
    evaluate_fn: ti.template(),
    quad_points: ti.template(),
    dim: ti.template(),
    a: ti.i32,
    x_cell: ti.f32,
    hsm_phys: ti.f32,
    kernel_prefactor: ti.f32,
    cellSize_x: ti.f32,
) -> ti.f32:
    """Quadrature-integrated kernel mass over cell a."""
    s = 0.0

    for ox, coeff in ti.static(quad_points):
        dx = (x_cell - (a + ox)) * cellSize_x
        q = ti.abs(dx) / hsm_phys
        s += coeff * kernel_prefactor * evaluate_fn(q, dim)

    return s * cellSize_x


@ti.func
def _isotropic_cell_integral_2d(
    evaluate_fn: ti.template(),
    quad_points: ti.template(),
    dim: ti.template(),
    a: ti.i32,
    b: ti.i32,
    x_cell: ti.f32,
    y_cell: ti.f32,
    hsm_phys: ti.f32,
    kernel_prefactor: ti.f32,
    cellSize_x: ti.f32,
    cellSize_y: ti.f32,
) -> ti.f32:
    """Quadrature-integrated kernel mass over cell (a, b), matching the
    C++ `eval` lambda + integrate_cell_2d(...) * cellSize_x * cellSize_y.
    """
    s = 0.0

    for ox, oy, coeff in ti.static(quad_points):

        dx = (x_cell - (a + ox)) * cellSize_x
        dy = (y_cell - (b + oy)) * cellSize_y

        r = ti.sqrt(dx * dx + dy * dy)
        q = r / hsm_phys

        s += coeff * kernel_prefactor * evaluate_fn(q, dim)

    return s * cellSize_x * cellSize_y


@ti.func
def _isotropic_cell_integral_3d(
    evaluate_fn: ti.template(),
    quad_points: ti.template(),
    dim: ti.template(),
    a: ti.i32,
    b: ti.i32,
    c: ti.i32,
    x_cell: ti.f32,
    y_cell: ti.f32,
    z_cell: ti.f32,
    hsm_phys: ti.f32,
    kernel_prefactor: ti.f32,
    cellSize_x: ti.f32,
    cellSize_y: ti.f32,
    cellSize_z: ti.f32,
) -> ti.f32:
    """Quadrature-integrated kernel mass over cell (a, b, c)."""
    s = 0.0

    for ox, oy, oz, coeff in ti.static(quad_points):

        dx = (x_cell - (a + ox)) * cellSize_x
        dy = (y_cell - (b + oy)) * cellSize_y
        dz = (z_cell - (c + oz)) * cellSize_z

        r = ti.sqrt(dx * dx + dy * dy + dz * dz)
        q = r / hsm_phys

        s += coeff * kernel_prefactor * evaluate_fn(q, dim)

    return s * cellSize_x * cellSize_y * cellSize_z


@ti.func
def _covariant_cell_integral_1d(
    evaluate_fn: ti.template(),
    quad_points: ti.template(),
    dim: ti.template(),
    a: ti.i32,
    x_cell: ti.f32,
    e0x: ti.f32,
    eval0: ti.f32,
    kernel_prefactor: ti.f32,
    cellSize_x: ti.f32,
) -> ti.f32:

    s = 0.0

    for ox, coeff in ti.static(quad_points):
        dx = (x_cell - (a + ox)) * cellSize_x

        # project the separation vector onto each principal axis and
        # scale by that axis's own semi-axis length -- this is what
        # turns the isotropic q = |d| / h into the ellipsoidal q
        xi0 = (e0x * dx) / eval0
        q = xi0
        s += coeff * kernel_prefactor * evaluate_fn(q, dim)

    return s * cellSize_x


@ti.func
def _covariant_cell_integral_2d(
    evaluate_fn: ti.template(),
    quad_points: ti.template(),
    dim: ti.template(),
    a: ti.i32,
    b: ti.i32,
    x_cell: ti.f32,
    y_cell: ti.f32,
    e0x: ti.f32,
    e0y: ti.f32,
    e1x: ti.f32,
    e1y: ti.f32,
    eval0: ti.f32,
    eval1: ti.f32,
    kernel_prefactor: ti.f32,
    cellSize_x: ti.f32,
    cellSize_y: ti.f32,
) -> ti.f32:

    s = 0.0

    for ox, oy, coeff in ti.static(quad_points):
        dx = (x_cell - (a + ox)) * cellSize_x
        dy = (y_cell - (b + oy)) * cellSize_y

        # project the separation vector onto each principal axis and
        # scale by that axis's own semi-axis length -- this is what
        # turns the isotropic q = |d| / h into the ellipsoidal q
        xi0 = (e0x * dx + e0y * dy) / eval0
        xi1 = (e1x * dx + e1y * dy) / eval1

        q = ti.sqrt(xi0 * xi0 + xi1 * xi1)

        s += coeff * kernel_prefactor * evaluate_fn(q, dim)

    return s * cellSize_x * cellSize_y


@ti.func
def _covariant_cell_integral_3d(
    evaluate_fn: ti.template(),
    quad_points: ti.template(),
    dim: ti.template(),
    a: ti.i32,
    b: ti.i32,
    c: ti.i32,
    x_cell: ti.f32,
    y_cell: ti.f32,
    z_cell: ti.f32,
    e0x: ti.f32,
    e0y: ti.f32,
    e0z: ti.f32,
    e1x: ti.f32,
    e1y: ti.f32,
    e1z: ti.f32,
    e2x: ti.f32,
    e2y: ti.f32,
    e2z: ti.f32,
    eval0: ti.f32,
    eval1: ti.f32,
    eval2: ti.f32,
    kernel_prefactor: ti.f32,
    cellSize_x: ti.f32,
    cellSize_y: ti.f32,
    cellSize_z: ti.f32,
) -> ti.f32:

    s = 0.0

    for ox, oy, oz, coeff in ti.static(quad_points):
        dx = (x_cell - (a + ox)) * cellSize_x
        dy = (y_cell - (b + oy)) * cellSize_y
        dz = (z_cell - (c + oz)) * cellSize_z

        # project the separation vector onto each principal axis and
        # scale by that axis's own semi-axis length -- this is what
        # turns the isotropic q = |d| / h into the ellipsoidal q
        xi0 = (e0x * dx + e0y * dy + e0z * dz) / eval0
        xi1 = (e1x * dx + e1y * dy + e1z * dz) / eval1
        xi2 = (e2x * dx + e2y * dy + e2z * dz) / eval2

        q = ti.sqrt(xi0 * xi0 + xi1 * xi1 + xi2 * xi2)

        s += coeff * kernel_prefactor * evaluate_fn(q, dim)

    return s * cellSize_x * cellSize_y * cellSize_z


# =============================================================================
# ||                                                                         ||
# ||                          STATIC KERNELS                                 ||
# ||                                                                         ||
# =============================================================================


# =============================================================================
# NGP
# =============================================================================
@ti.kernel
def _ngp_1d(
    periodic: ti.template(),
    particle_positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 1)
    particle_fields: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    boxsizes: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (1,)
    grid_fields: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (Nx, F)
    grid_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (Nx,)
):
    Nx = grid_weights.shape[0]
    F = grid_fields.shape[1]

    inv_dx = Nx / boxsizes[0]

    for p in range(particle_positions.shape[0]):
        ix = ti.floor(particle_positions[p, 0] * inv_dx, int)

        keep = True
        if ti.static(periodic):
            ix = _apply_pbc(ix, Nx)
        else:
            keep = 0 <= ix < Nx

        if keep:
            w = particle_weights[p]
            ti.atomic_add(grid_weights[ix], w)
            for f in range(F):
                ti.atomic_add(grid_fields[ix, f], particle_fields[p, f] * w)


@ti.kernel
def _ngp_2d(
    periodic: ti.template(),
    particle_positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 2)
    particle_fields: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    boxsizes: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (2,)
    grid_fields: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (Nx, Ny, F)
    grid_weights: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (Nx, Ny)
):
    Nx = grid_weights.shape[0]
    Ny = grid_weights.shape[1]
    F = grid_fields.shape[2]

    inv_dx = ti.Vector([Nx / boxsizes[0], Ny / boxsizes[1]])

    for p in range(particle_positions.shape[0]):
        ix = ti.floor(particle_positions[p, 0] * inv_dx[0], int)
        iy = ti.floor(particle_positions[p, 1] * inv_dx[1], int)

        keep = True
        if ti.static(periodic):
            ix = _apply_pbc(ix, Nx)
            iy = _apply_pbc(iy, Ny)
        else:
            keep = (0 <= ix < Nx) and (0 <= iy < Ny)

        if keep:
            w = particle_weights[p]
            ti.atomic_add(grid_weights[ix, iy], w)
            for f in range(F):
                ti.atomic_add(grid_fields[ix, iy, f], particle_fields[p, f] * w)


@ti.kernel
def _ngp_3d(
    periodic: ti.template(),
    particle_positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 3)
    particle_fields: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    boxsizes: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (3,)
    grid_fields: ti.types.ndarray(dtype=ti.f32, ndim=4),  # (Nx, Ny, Nz, F)
    grid_weights: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (Nx, Ny, Nz)
):
    Nx = grid_weights.shape[0]
    Ny = grid_weights.shape[1]
    Nz = grid_weights.shape[2]
    F = grid_fields.shape[3]

    inv_dx = ti.Vector([Nx / boxsizes[0], Ny / boxsizes[1], Nz / boxsizes[2]])

    for p in range(particle_positions.shape[0]):
        ix = ti.floor(particle_positions[p, 0] * inv_dx[0], int)
        iy = ti.floor(particle_positions[p, 1] * inv_dx[1], int)
        iz = ti.floor(particle_positions[p, 2] * inv_dx[2], int)

        keep = True
        if ti.static(periodic):
            ix = _apply_pbc(ix, Nx)
            iy = _apply_pbc(iy, Ny)
            iz = _apply_pbc(iz, Nz)
        else:
            keep = (0 <= ix < Nx) and (0 <= iy < Ny) and (0 <= iz < Nz)  # BUGFIX (a)

        if keep:
            w = particle_weights[p]
            ti.atomic_add(grid_weights[ix, iy, iz], w)
            for f in range(F):
                ti.atomic_add(grid_fields[ix, iy, iz, f], particle_fields[p, f] * w)


# =============================================================================
# CIC
# =============================================================================
@ti.kernel
def _cic_1d(
    periodic: ti.template(),
    particle_positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 1)
    particle_fields: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    boxsizes: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (1,)
    grid_fields: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (Nx, F)
    grid_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (Nx,)
):
    Nx = grid_weights.shape[0]
    F = grid_fields.shape[1]
    SHIFT = 0
    K = 2

    inv_dx = Nx / boxsizes[0]

    for p in range(particle_positions.shape[0]):
        gx = particle_positions[p, 0] * inv_dx - 0.5

        ix = ti.floor(gx, int)
        fx = gx - ix

        wx = ti.Vector([1.0 - fx, fx])

        ox_min, ox_max = 0, K - 1
        if ti.static(not periodic):
            ox_min = ti.max(0, SHIFT - ix)
            ox_max = ti.min(K - 1, SHIFT - ix + Nx - 1)

        w_particle = particle_weights[p]

        for ox in range(ox_min, ox_max + 1):
            x = ix + ox - SHIFT
            if ti.static(periodic):
                x = _apply_pbc(x, Nx)

            w = wx[ox] * w_particle
            ti.atomic_add(grid_weights[x], w)
            for f in range(F):
                ti.atomic_add(grid_fields[x, f], particle_fields[p, f] * w)


@ti.kernel
def _cic_2d(
    periodic: ti.template(),
    particle_positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 2)
    particle_fields: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    boxsizes: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (2,)
    grid_fields: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (Nx, Ny, F)
    grid_weights: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (Nx, Ny)
):
    Nx = grid_weights.shape[0]
    Ny = grid_weights.shape[1]
    F = grid_fields.shape[2]
    SHIFT = 0
    K = 2

    inv_dx = ti.Vector([Nx / boxsizes[0], Ny / boxsizes[1]])

    for p in range(particle_positions.shape[0]):
        gx = particle_positions[p, 0] * inv_dx[0] - 0.5
        gy = particle_positions[p, 1] * inv_dx[1] - 0.5

        ix = ti.floor(gx, int)
        iy = ti.floor(gy, int)

        fx = gx - ix
        fy = gy - iy

        wx = ti.Vector([1.0 - fx, fx])
        wy = ti.Vector([1.0 - fy, fy])

        ox_min, ox_max = 0, K - 1
        oy_min, oy_max = 0, K - 1
        if ti.static(not periodic):
            ox_min = ti.max(0, SHIFT - ix)
            ox_max = ti.min(K - 1, SHIFT - ix + Nx - 1)
            oy_min = ti.max(0, SHIFT - iy)
            oy_max = ti.min(K - 1, SHIFT - iy + Ny - 1)

        w_particle = particle_weights[p]

        for ox in range(ox_min, ox_max + 1):
            x = ix + ox - SHIFT
            if ti.static(periodic):
                x = _apply_pbc(x, Nx)

            for oy in range(oy_min, oy_max + 1):
                y = iy + oy - SHIFT
                if ti.static(periodic):
                    y = _apply_pbc(y, Ny)

                w = wx[ox] * wy[oy] * w_particle
                ti.atomic_add(grid_weights[x, y], w)
                for f in range(F):
                    ti.atomic_add(grid_fields[x, y, f], particle_fields[p, f] * w)


@ti.kernel
def _cic_3d(
    periodic: ti.template(),
    particle_positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 3)
    particle_fields: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    boxsizes: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (3,)
    grid_fields: ti.types.ndarray(dtype=ti.f32, ndim=4),  # (Nx, Ny, Nz, F)
    grid_weights: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (Nx, Ny, Nz)
):
    Nx = grid_weights.shape[0]
    Ny = grid_weights.shape[1]
    Nz = grid_weights.shape[2]
    F = grid_fields.shape[3]
    SHIFT = 0
    K = 2

    inv_dx = ti.Vector([Nx / boxsizes[0], Ny / boxsizes[1], Nz / boxsizes[2]])

    for p in range(particle_positions.shape[0]):
        gx = particle_positions[p, 0] * inv_dx[0] - 0.5
        gy = particle_positions[p, 1] * inv_dx[1] - 0.5
        gz = particle_positions[p, 2] * inv_dx[2] - 0.5

        ix = ti.floor(gx, int)
        iy = ti.floor(gy, int)
        iz = ti.floor(gz, int)

        fx = gx - ix
        fy = gy - iy
        fz = gz - iz

        wx = ti.Vector([1.0 - fx, fx])
        wy = ti.Vector([1.0 - fy, fy])
        wz = ti.Vector([1.0 - fz, fz])

        ox_min, ox_max = 0, K - 1
        oy_min, oy_max = 0, K - 1
        oz_min, oz_max = 0, K - 1
        if ti.static(not periodic):
            ox_min = ti.max(0, SHIFT - ix)
            ox_max = ti.min(K - 1, SHIFT - ix + Nx - 1)
            oy_min = ti.max(0, SHIFT - iy)
            oy_max = ti.min(K - 1, SHIFT - iy + Ny - 1)
            oz_min = ti.max(0, SHIFT - iz)
            oz_max = ti.min(
                K - 1, SHIFT - iz + Nz - 1
            )  # BUGFIX (b) -- was `0 <= Nz < Nz`

        w_particle = particle_weights[p]

        for ox in range(ox_min, ox_max + 1):
            x = ix + ox - SHIFT
            if ti.static(periodic):
                x = _apply_pbc(x, Nx)

            for oy in range(oy_min, oy_max + 1):
                y = iy + oy - SHIFT
                if ti.static(periodic):
                    y = _apply_pbc(y, Ny)

                for oz in range(oz_min, oz_max + 1):
                    z = iz + oz - SHIFT
                    if ti.static(periodic):
                        z = _apply_pbc(z, Nz)

                    w = wx[ox] * wy[oy] * wz[oz] * w_particle
                    ti.atomic_add(grid_weights[x, y, z], w)
                    for f in range(F):
                        ti.atomic_add(
                            grid_fields[x, y, z, f], particle_fields[p, f] * w
                        )


# =============================================================================
# TSC
# =============================================================================
@ti.kernel
def _tsc_1d(
    periodic: ti.template(),
    particle_positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 1)
    particle_fields: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    boxsizes: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (1,)
    grid_fields: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (Nx, F)
    grid_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (Nx,)
):
    Nx = grid_weights.shape[0]
    F = grid_fields.shape[1]
    SHIFT = 1
    K = 3

    inv_dx = ti.Vector([Nx / boxsizes[0]])

    for p in range(particle_positions.shape[0]):
        gx = particle_positions[p, 0] * inv_dx[0] - 0.5
        ix = ti.round(gx, int)
        dx = gx - ix
        wx = ti.Vector([0.5 * (0.5 - dx) ** 2, 0.75 - dx * dx, 0.5 * (0.5 + dx) ** 2])

        ox_min, ox_max = 0, K - 1
        if ti.static(not periodic):
            ox_min = ti.max(0, SHIFT - ix)
            ox_max = ti.min(K - 1, SHIFT - ix + Nx - 1)

        w_particle = particle_weights[p]

        for ox in range(ox_min, ox_max + 1):
            x = ix + ox - SHIFT
            if ti.static(periodic):
                x = _apply_pbc(x, Nx)

            w = wx[ox] * w_particle
            ti.atomic_add(grid_weights[x], w)
            for f in range(F):
                ti.atomic_add(grid_fields[x, f], particle_fields[p, f] * w)


@ti.kernel
def _tsc_2d(
    periodic: ti.template(),
    particle_positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 2)
    particle_fields: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    boxsizes: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (2,)
    grid_fields: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (Nx, Ny, F)
    grid_weights: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (Nx, Ny)
):
    Nx = grid_weights.shape[0]
    Ny = grid_weights.shape[1]
    F = grid_fields.shape[2]
    SHIFT = 1
    K = 3

    inv_dx = ti.Vector([Nx / boxsizes[0], Ny / boxsizes[1]])

    for p in range(particle_positions.shape[0]):
        gx = particle_positions[p, 0] * inv_dx[0] - 0.5
        gy = particle_positions[p, 1] * inv_dx[1] - 0.5

        ix = ti.round(gx, int)
        iy = ti.round(gy, int)

        dx = gx - ix
        dy = gy - iy

        wx = ti.Vector([0.5 * (0.5 - dx) ** 2, 0.75 - dx * dx, 0.5 * (0.5 + dx) ** 2])
        wy = ti.Vector([0.5 * (0.5 - dy) ** 2, 0.75 - dy * dy, 0.5 * (0.5 + dy) ** 2])

        ox_min, ox_max = 0, K - 1
        oy_min, oy_max = 0, K - 1
        if ti.static(not periodic):
            ox_min = ti.max(0, SHIFT - ix)
            ox_max = ti.min(K - 1, SHIFT - ix + Nx - 1)
            oy_min = ti.max(0, SHIFT - iy)
            oy_max = ti.min(K - 1, SHIFT - iy + Ny - 1)

        w_particle = particle_weights[p]

        for ox in range(ox_min, ox_max + 1):
            x = ix + ox - SHIFT
            if ti.static(periodic):
                x = _apply_pbc(x, Nx)

            for oy in range(oy_min, oy_max + 1):
                y = iy + oy - SHIFT
                if ti.static(periodic):
                    y = _apply_pbc(y, Ny)

                w = wx[ox] * wy[oy] * w_particle
                ti.atomic_add(grid_weights[x, y], w)
                for f in range(F):
                    ti.atomic_add(grid_fields[x, y, f], particle_fields[p, f] * w)


@ti.kernel
def _tsc_3d(
    periodic: ti.template(),
    particle_positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 3)
    particle_fields: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    boxsizes: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (3,)
    grid_fields: ti.types.ndarray(dtype=ti.f32, ndim=4),  # (Nx, Ny, Nz, F)
    grid_weights: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (Nx, Ny, Nz)
):
    Nx = grid_weights.shape[0]
    Ny = grid_weights.shape[1]
    Nz = grid_weights.shape[2]
    F = grid_fields.shape[3]
    SHIFT = 1
    K = 3

    inv_dx = ti.Vector([Nx / boxsizes[0], Ny / boxsizes[1], Nz / boxsizes[2]])

    for p in range(particle_positions.shape[0]):
        gx = particle_positions[p, 0] * inv_dx[0] - 0.5
        gy = particle_positions[p, 1] * inv_dx[1] - 0.5
        gz = particle_positions[p, 2] * inv_dx[2] - 0.5

        ix = ti.round(gx, int)
        iy = ti.round(gy, int)
        iz = ti.round(gz, int)

        dx = gx - ix
        dy = gy - iy
        dz = gz - iz

        wx = ti.Vector([0.5 * (0.5 - dx) ** 2, 0.75 - dx * dx, 0.5 * (0.5 + dx) ** 2])
        wy = ti.Vector([0.5 * (0.5 - dy) ** 2, 0.75 - dy * dy, 0.5 * (0.5 + dy) ** 2])
        wz = ti.Vector([0.5 * (0.5 - dz) ** 2, 0.75 - dz * dz, 0.5 * (0.5 + dz) ** 2])

        ox_min, ox_max = 0, K - 1
        oy_min, oy_max = 0, K - 1
        oz_min, oz_max = 0, K - 1
        if ti.static(not periodic):
            ox_min = ti.max(0, SHIFT - ix)
            ox_max = ti.min(K - 1, SHIFT - ix + Nx - 1)
            oy_min = ti.max(0, SHIFT - iy)
            oy_max = ti.min(K - 1, SHIFT - iy + Ny - 1)
            oz_min = ti.max(0, SHIFT - iz)
            oz_max = ti.min(K - 1, SHIFT - iz + Nz - 1)

        w_particle = particle_weights[p]

        for ox in range(ox_min, ox_max + 1):
            x = ix + ox - SHIFT
            if ti.static(periodic):
                x = _apply_pbc(x, Nx)

            for oy in range(oy_min, oy_max + 1):
                y = iy + oy - SHIFT
                if ti.static(periodic):
                    y = _apply_pbc(y, Ny)

                for oz in range(oz_min, oz_max + 1):
                    z = iz + oz - SHIFT
                    if ti.static(periodic):
                        z = _apply_pbc(z, Nz)

                    w = wx[ox] * wy[oy] * wz[oz] * w_particle
                    ti.atomic_add(grid_weights[x, y, z], w)
                    for f in range(F):
                        ti.atomic_add(
                            grid_fields[x, y, z, f], particle_fields[p, f] * w
                        )


# =============================================================================
# PCS
# =============================================================================
@ti.kernel
def _pcs_1d(
    periodic: ti.template(),
    particle_positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 1)
    particle_fields: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    boxsizes: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (1,)
    grid_fields: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (Nx, F)
    grid_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (Nx,)
):
    Nx = grid_weights.shape[0]
    F = grid_fields.shape[1]
    SHIFT = 1
    K = 4

    inv_dx = ti.Vector([Nx / boxsizes[0]])

    for p in range(particle_positions.shape[0]):
        gx = particle_positions[p, 0] * inv_dx[0] - 0.5

        ix = ti.floor(gx, int)  # BUGFIX (c) -- was ti.round

        dx = gx - ix

        wx = ti.Vector(
            [
                (1.0 - dx) ** 3 / 6.0,
                (4.0 - 6.0 * dx**2 + 3.0 * dx**3) / 6.0,
                (1.0 + 3.0 * dx + 3.0 * dx**2 - 3.0 * dx**3) / 6.0,
                dx**3 / 6.0,
            ]
        )

        ox_min, ox_max = 0, K - 1
        if ti.static(not periodic):
            ox_min = ti.max(0, SHIFT - ix)
            ox_max = ti.min(K - 1, SHIFT - ix + Nx - 1)

        w_particle = particle_weights[p]

        for ox in range(ox_min, ox_max + 1):
            x = ix + ox - SHIFT
            if ti.static(periodic):
                x = _apply_pbc(x, Nx)

            w = wx[ox] * w_particle
            ti.atomic_add(grid_weights[x], w)
            for f in range(F):
                ti.atomic_add(grid_fields[x, f], particle_fields[p, f] * w)


@ti.kernel
def _pcs_2d(
    periodic: ti.template(),
    particle_positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 2)
    particle_fields: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    boxsizes: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (2,)
    grid_fields: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (Nx, Ny, F)
    grid_weights: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (Nx, Ny)
):
    Nx = grid_weights.shape[0]
    Ny = grid_weights.shape[1]
    F = grid_fields.shape[2]
    SHIFT = 1
    K = 4

    inv_dx = ti.Vector([Nx / boxsizes[0], Ny / boxsizes[1]])

    for p in range(particle_positions.shape[0]):
        gx = particle_positions[p, 0] * inv_dx[0] - 0.5
        gy = particle_positions[p, 1] * inv_dx[1] - 0.5

        ix = ti.floor(gx, int)  # BUGFIX (c) -- was ti.round
        iy = ti.floor(gy, int)  # BUGFIX (c) -- was ti.round

        dx = gx - ix
        dy = gy - iy

        wx = ti.Vector(
            [
                (1.0 - dx) ** 3 / 6.0,
                (4.0 - 6.0 * dx**2 + 3.0 * dx**3) / 6.0,
                (1.0 + 3.0 * dx + 3.0 * dx**2 - 3.0 * dx**3) / 6.0,
                dx**3 / 6.0,
            ]
        )
        wy = ti.Vector(
            [
                (1.0 - dy) ** 3 / 6.0,
                (4.0 - 6.0 * dy**2 + 3.0 * dy**3) / 6.0,
                (1.0 + 3.0 * dy + 3.0 * dy**2 - 3.0 * dy**3) / 6.0,
                dy**3 / 6.0,
            ]
        )

        ox_min, ox_max = 0, K - 1
        oy_min, oy_max = 0, K - 1
        if ti.static(not periodic):
            ox_min = ti.max(0, SHIFT - ix)
            ox_max = ti.min(K - 1, SHIFT - ix + Nx - 1)
            oy_min = ti.max(0, SHIFT - iy)
            oy_max = ti.min(K - 1, SHIFT - iy + Ny - 1)

        w_particle = particle_weights[p]

        for ox in range(ox_min, ox_max + 1):
            x = ix + ox - SHIFT
            if ti.static(periodic):
                x = _apply_pbc(x, Nx)

            for oy in range(oy_min, oy_max + 1):
                y = iy + oy - SHIFT
                if ti.static(periodic):
                    y = _apply_pbc(y, Ny)

                w = wx[ox] * wy[oy] * w_particle
                ti.atomic_add(grid_weights[x, y], w)
                for f in range(F):
                    ti.atomic_add(grid_fields[x, y, f], particle_fields[p, f] * w)


@ti.kernel
def _pcs_3d(
    periodic: ti.template(),
    particle_positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 3)
    particle_fields: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    boxsizes: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (3,)
    grid_fields: ti.types.ndarray(dtype=ti.f32, ndim=4),  # (Nx, Ny, Nz, F)
    grid_weights: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (Nx, Ny, Nz)
):
    Nx = grid_weights.shape[0]
    Ny = grid_weights.shape[1]
    Nz = grid_weights.shape[2]
    F = grid_fields.shape[3]
    SHIFT = 1
    K = 4

    inv_dx = ti.Vector([Nx / boxsizes[0], Ny / boxsizes[1], Nz / boxsizes[2]])

    for p in range(particle_positions.shape[0]):
        gx = particle_positions[p, 0] * inv_dx[0] - 0.5
        gy = particle_positions[p, 1] * inv_dx[1] - 0.5
        gz = particle_positions[p, 2] * inv_dx[2] - 0.5

        ix = ti.floor(gx, int)  # BUGFIX (c) -- was ti.round
        iy = ti.floor(gy, int)  # BUGFIX (c) -- was ti.round
        iz = ti.floor(gz, int)  # BUGFIX (c) -- was ti.round

        dx = gx - ix
        dy = gy - iy
        dz = gz - iz

        wx = ti.Vector(
            [
                (1.0 - dx) ** 3 / 6.0,
                (4.0 - 6.0 * dx**2 + 3.0 * dx**3) / 6.0,
                (1.0 + 3.0 * dx + 3.0 * dx**2 - 3.0 * dx**3) / 6.0,
                dx**3 / 6.0,
            ]
        )
        wy = ti.Vector(
            [
                (1.0 - dy) ** 3 / 6.0,
                (4.0 - 6.0 * dy**2 + 3.0 * dy**3) / 6.0,
                (1.0 + 3.0 * dy + 3.0 * dy**2 - 3.0 * dy**3) / 6.0,
                dy**3 / 6.0,
            ]
        )
        wz = ti.Vector(
            [
                (1.0 - dz) ** 3 / 6.0,
                (4.0 - 6.0 * dz**2 + 3.0 * dz**3) / 6.0,
                (1.0 + 3.0 * dz + 3.0 * dz**2 - 3.0 * dz**3) / 6.0,
                dz**3 / 6.0,
            ]
        )

        ox_min, ox_max = 0, K - 1
        oy_min, oy_max = 0, K - 1
        oz_min, oz_max = 0, K - 1
        if ti.static(not periodic):
            ox_min = ti.max(0, SHIFT - ix)
            ox_max = ti.min(K - 1, SHIFT - ix + Nx - 1)
            oy_min = ti.max(0, SHIFT - iy)
            oy_max = ti.min(K - 1, SHIFT - iy + Ny - 1)
            oz_min = ti.max(0, SHIFT - iz)
            oz_max = ti.min(K - 1, SHIFT - iz + Nz - 1)

        w_particle = particle_weights[p]

        for ox in range(ox_min, ox_max + 1):
            x = ix + ox - SHIFT
            if ti.static(periodic):
                x = _apply_pbc(x, Nx)

            for oy in range(oy_min, oy_max + 1):
                y = iy + oy - SHIFT
                if ti.static(periodic):
                    y = _apply_pbc(y, Ny)

                for oz in range(oz_min, oz_max + 1):
                    z = iz + oz - SHIFT
                    if ti.static(periodic):
                        z = _apply_pbc(z, Nz)

                    w = wx[ox] * wy[oy] * wz[oz] * w_particle
                    ti.atomic_add(grid_weights[x, y, z], w)
                    for f in range(F):
                        ti.atomic_add(
                            grid_fields[x, y, z, f], particle_fields[p, f] * w
                        )


# =============================================================================
# PQS
# =============================================================================
@ti.kernel
def _pqs_1d(
    periodic: ti.template(),
    particle_positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 1)
    particle_fields: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    boxsizes: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (1,)
    grid_fields: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (Nx, F)
    grid_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (Nx,)
):
    Nx = grid_weights.shape[0]
    F = grid_fields.shape[1]
    SHIFT = 2  # BUGFIX (d) -- was 1
    K = 6

    inv_dx = ti.Vector([Nx / boxsizes[0]])

    for p in range(particle_positions.shape[0]):
        gx = particle_positions[p, 0] * inv_dx[0] - 0.5
        ix = ti.floor(gx, int)  # BUGFIX (c) -- was ti.round
        dx = gx - ix

        wx = ti.Vector(
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

        ox_min, ox_max = 0, K - 1
        if ti.static(not periodic):
            ox_min = ti.max(0, SHIFT - ix)
            ox_max = ti.min(K - 1, SHIFT - ix + Nx - 1)

        w_particle = particle_weights[p]

        for ox in range(ox_min, ox_max + 1):
            x = ix + ox - SHIFT
            if ti.static(periodic):
                x = _apply_pbc(x, Nx)

            w = wx[ox] * w_particle
            ti.atomic_add(grid_weights[x], w)
            for f in range(F):
                ti.atomic_add(grid_fields[x, f], particle_fields[p, f] * w)


@ti.kernel
def _pqs_2d(
    periodic: ti.template(),
    particle_positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 2)
    particle_fields: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    boxsizes: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (2,)
    grid_fields: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (Nx, Ny, F)
    grid_weights: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (Nx, Ny)
):
    Nx = grid_weights.shape[0]
    Ny = grid_weights.shape[1]
    F = grid_fields.shape[2]
    SHIFT = 2  # BUGFIX (d) -- was 1
    K = 6

    inv_dx = ti.Vector([Nx / boxsizes[0], Ny / boxsizes[1]])

    for p in range(particle_positions.shape[0]):
        gx = particle_positions[p, 0] * inv_dx[0] - 0.5
        gy = particle_positions[p, 1] * inv_dx[1] - 0.5

        ix = ti.floor(gx, int)  # BUGFIX (c) -- was ti.round
        iy = ti.floor(gy, int)  # BUGFIX (c) -- was ti.round

        dx = gx - ix
        dy = gy - iy

        wx = ti.Vector(
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
        wy = ti.Vector(
            [
                (1.0 - dy) ** 5 / 120.0,
                ((2.0 - dy) ** 5 - 6.0 * (1.0 - dy) ** 5) / 120.0,
                ((3.0 - dy) ** 5 - 6.0 * (2.0 - dy) ** 5 + 15.0 * (1.0 - dy) ** 5)
                / 120.0,
                ((2.0 + dy) ** 5 - 6.0 * (1.0 + dy) ** 5 + 15.0 * dy**5) / 120.0,
                ((1.0 + dy) ** 5 - 6.0 * dy**5) / 120.0,
                dy**5 / 120.0,
            ]
        )

        ox_min, ox_max = 0, K - 1
        oy_min, oy_max = 0, K - 1
        if ti.static(not periodic):
            ox_min = ti.max(0, SHIFT - ix)
            ox_max = ti.min(K - 1, SHIFT - ix + Nx - 1)
            oy_min = ti.max(0, SHIFT - iy)
            oy_max = ti.min(K - 1, SHIFT - iy + Ny - 1)

        w_particle = particle_weights[p]

        for ox in range(ox_min, ox_max + 1):
            x = ix + ox - SHIFT
            if ti.static(periodic):
                x = _apply_pbc(x, Nx)

            for oy in range(oy_min, oy_max + 1):
                y = iy + oy - SHIFT
                if ti.static(periodic):
                    y = _apply_pbc(y, Ny)

                w = wx[ox] * wy[oy] * w_particle
                ti.atomic_add(grid_weights[x, y], w)
                for f in range(F):
                    ti.atomic_add(grid_fields[x, y, f], particle_fields[p, f] * w)


@ti.kernel
def _pqs_3d(
    periodic: ti.template(),
    particle_positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 3)
    particle_fields: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    boxsizes: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (3,)
    grid_fields: ti.types.ndarray(dtype=ti.f32, ndim=4),  # (Nx, Ny, Nz, F)
    grid_weights: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (Nx, Ny, Nz)
):
    Nx = grid_weights.shape[0]
    Ny = grid_weights.shape[1]
    Nz = grid_weights.shape[2]
    F = grid_fields.shape[3]
    SHIFT = 2  # BUGFIX (d) -- was 1
    K = 6

    inv_dx = ti.Vector([Nx / boxsizes[0], Ny / boxsizes[1], Nz / boxsizes[2]])

    for p in range(particle_positions.shape[0]):
        gx = particle_positions[p, 0] * inv_dx[0] - 0.5
        gy = particle_positions[p, 1] * inv_dx[1] - 0.5
        gz = particle_positions[p, 2] * inv_dx[2] - 0.5

        ix = ti.floor(gx, int)  # BUGFIX (c) -- was ti.round
        iy = ti.floor(gy, int)  # BUGFIX (c) -- was ti.round
        iz = ti.floor(gz, int)  # BUGFIX (c) -- was ti.round

        dx = gx - ix
        dy = gy - iy
        dz = gz - iz

        wx = ti.Vector(
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
        wy = ti.Vector(
            [  # BUGFIX (e) -- was built from dx
                (1.0 - dy) ** 5 / 120.0,
                ((2.0 - dy) ** 5 - 6.0 * (1.0 - dy) ** 5) / 120.0,
                ((3.0 - dy) ** 5 - 6.0 * (2.0 - dy) ** 5 + 15.0 * (1.0 - dy) ** 5)
                / 120.0,
                ((2.0 + dy) ** 5 - 6.0 * (1.0 + dy) ** 5 + 15.0 * dy**5) / 120.0,
                ((1.0 + dy) ** 5 - 6.0 * dy**5) / 120.0,
                dy**5 / 120.0,
            ]
        )
        wz = ti.Vector(
            [  # BUGFIX (e) -- was built from dx
                (1.0 - dz) ** 5 / 120.0,
                ((2.0 - dz) ** 5 - 6.0 * (1.0 - dz) ** 5) / 120.0,
                ((3.0 - dz) ** 5 - 6.0 * (2.0 - dz) ** 5 + 15.0 * (1.0 - dz) ** 5)
                / 120.0,
                ((2.0 + dz) ** 5 - 6.0 * (1.0 + dz) ** 5 + 15.0 * dz**5) / 120.0,
                ((1.0 + dz) ** 5 - 6.0 * dz**5) / 120.0,
                dz**5 / 120.0,
            ]
        )

        ox_min, ox_max = 0, K - 1
        oy_min, oy_max = 0, K - 1
        oz_min, oz_max = 0, K - 1
        if ti.static(not periodic):
            ox_min = ti.max(0, SHIFT - ix)
            ox_max = ti.min(K - 1, SHIFT - ix + Nx - 1)
            oy_min = ti.max(0, SHIFT - iy)
            oy_max = ti.min(K - 1, SHIFT - iy + Ny - 1)
            oz_min = ti.max(0, SHIFT - iz)
            oz_max = ti.min(K - 1, SHIFT - iz + Nz - 1)

        w_particle = particle_weights[p]

        for ox in range(ox_min, ox_max + 1):
            x = ix + ox - SHIFT
            if ti.static(periodic):
                x = _apply_pbc(x, Nx)

            for oy in range(oy_min, oy_max + 1):
                y = iy + oy - SHIFT
                if ti.static(periodic):
                    y = _apply_pbc(y, Ny)

                for oz in range(oz_min, oz_max + 1):
                    z = iz + oz - SHIFT
                    if ti.static(periodic):
                        z = _apply_pbc(z, Nz)

                    w = wx[ox] * wy[oy] * wz[oz] * w_particle
                    ti.atomic_add(grid_weights[x, y, z], w)
                    for f in range(F):
                        ti.atomic_add(
                            grid_fields[x, y, z, f], particle_fields[p, f] * w
                        )


# =============================================================================
# ||                                                                         ||
# ||                       ADAPTIVE KERNELS                                  ||
# ||                                                                         ||
# =============================================================================


# =============================================================================
# Separable Kernels
# =============================================================================
@ti.kernel
def _separable_1d(
    F_1d_fn: ti.template(),
    sigma_fn: ti.template(),
    kernel_support: ti.f32,
    dim: ti.template(),   # only new line in the signature
    periodic: ti.template(),
    positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 1)
    quantities: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
    smoothing_lengths: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 1)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    cellSize_x_inv: ti.f32,
    max_support: ti.f32,
    MAX_STENCIL: ti.template(),
    Nx: ti.i32,
    fields: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (Nx, F)
    weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (Nx,)
):
    F = fields.shape[1]
    sig = sigma_fn(dim)

    for n in range(positions.shape[0]):

        wx = ti.Vector.zero(ti.f32, MAX_STENCIL)
        wj = particle_weights[n]

        x_phys = positions[n, 0]

        hsm_x_phys = smoothing_lengths[n, 0]

        if ti.static(periodic):
            hsm_max = hsm_x_phys
            if hsm_max > max_support:
                scale = max_support / hsm_max
                hsm_x_phys *= scale

        x_cell = x_phys * cellSize_x_inv

        hsm_x_cell = hsm_x_phys * cellSize_x_inv

        inv_hx = 1.0 / hsm_x_cell

        support_x_cell = kernel_support * hsm_x_cell

        i_min = ti.cast(ti.floor(x_cell - support_x_cell), ti.i32)
        i_max = ti.cast(ti.ceil(x_cell + support_x_cell), ti.i32) - 1

        if ti.static(not periodic):
            i_min = ti.max(i_min, 0)
            i_max = ti.min(i_max, Nx - 1)

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

        for si in ti.ndrange(nx):

            i = i_min + si

            ii = i
            if ti.static(periodic):
                ii = _apply_pbc(i, Nx)

            w = wx[si]
            ti.atomic_add(weights[ii], w)
            for f in range(F):
                ti.atomic_add(
                    fields[ii, f],
                    quantities[n, f] * w,
                )


@ti.kernel
def _separable_2d(
    F_1d_fn: ti.template(),
    sigma_fn: ti.template(),
    kernel_support: ti.f32,
    dim: ti.template(),   # only new line in the signature
    periodic: ti.template(),
    positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 2)
    quantities: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
    smoothing_lengths: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 2)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    cellSize_x_inv: ti.f32,
    cellSize_y_inv: ti.f32,
    max_support: ti.f32,
    MAX_STENCIL: ti.template(),
    Nx: ti.i32,
    Ny: ti.i32,
    fields: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (Nx, Ny, F)
    weights: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (Nx, Ny)
):
    F = fields.shape[2]
    sig = sigma_fn(dim)

    for n in range(positions.shape[0]):

        wx = ti.Vector.zero(ti.f32, MAX_STENCIL)
        wy = ti.Vector.zero(ti.f32, MAX_STENCIL)
        wj = particle_weights[n]

        x_phys = positions[n, 0]
        y_phys = positions[n, 1]

        hsm_x_phys = smoothing_lengths[n, 0]
        hsm_y_phys = smoothing_lengths[n, 1]

        if ti.static(periodic):
            hsm_max = ti.max(hsm_x_phys, hsm_y_phys)
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

        i_min = ti.cast(ti.floor(x_cell - support_x_cell), ti.i32)
        i_max = ti.cast(ti.ceil(x_cell + support_x_cell), ti.i32) - 1
        j_min = ti.cast(ti.floor(y_cell - support_y_cell), ti.i32)
        j_max = ti.cast(ti.ceil(y_cell + support_y_cell), ti.i32) - 1

        if ti.static(not periodic):
            i_min = ti.max(i_min, 0)
            i_max = ti.min(i_max, Nx - 1)
            j_min = ti.max(j_min, 0)
            j_max = ti.min(j_max, Ny - 1)

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

        for si, sj in ti.ndrange(nx, ny):

            i = i_min + si
            j = j_min + sj

            ii = i
            jj = j
            if ti.static(periodic):
                ii = _apply_pbc(i, Nx)
                jj = _apply_pbc(j, Ny)

            w = wx[si] * wy[sj]
            ti.atomic_add(weights[ii, jj], w)
            for f in range(F):
                ti.atomic_add(
                    fields[ii, jj, f],
                    quantities[n, f] * w,
                )


@ti.kernel
def _separable_3d(
    F_1d_fn: ti.template(),
    sigma_fn: ti.template(),
    kernel_support: ti.f32,
    dim: ti.template(),   # only new line in the signature
    periodic: ti.template(),
    positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 3)
    quantities: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
    smoothing_lengths: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 3)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    cellSize_x_inv: ti.f32,
    cellSize_y_inv: ti.f32,
    cellSize_z_inv: ti.f32,
    max_support: ti.f32,
    MAX_STENCIL: ti.template(),
    Nx: ti.i32,
    Ny: ti.i32,
    Nz: ti.i32,
    fields: ti.types.ndarray(dtype=ti.f32, ndim=4),  # (Nx, Ny, Nz, F)
    weights: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (Nx, Ny, Nz)
):
    F = fields.shape[3]
    sig = sigma_fn(dim)

    for n in range(positions.shape[0]):

        wx = ti.Vector.zero(ti.f32, MAX_STENCIL)
        wy = ti.Vector.zero(ti.f32, MAX_STENCIL)
        wz = ti.Vector.zero(ti.f32, MAX_STENCIL)
        wj = particle_weights[n]

        x_phys = positions[n, 0]
        y_phys = positions[n, 1]
        z_phys = positions[n, 2]

        hsm_x_phys = smoothing_lengths[n, 0]
        hsm_y_phys = smoothing_lengths[n, 1]
        hsm_z_phys = smoothing_lengths[n, 2]

        if ti.static(periodic):
            hsm_max = ti.max(hsm_x_phys, ti.max(hsm_y_phys, hsm_z_phys))
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

        i_min = ti.cast(ti.floor(x_cell - support_x_cell), ti.i32)
        i_max = ti.cast(ti.ceil(x_cell + support_x_cell), ti.i32) - 1
        j_min = ti.cast(ti.floor(y_cell - support_y_cell), ti.i32)
        j_max = ti.cast(ti.ceil(y_cell + support_y_cell), ti.i32) - 1
        k_min = ti.cast(ti.floor(z_cell - support_z_cell), ti.i32)
        k_max = ti.cast(ti.ceil(z_cell + support_z_cell), ti.i32) - 1

        if ti.static(not periodic):
            i_min = ti.max(i_min, 0)
            i_max = ti.min(i_max, Nx - 1)
            j_min = ti.max(j_min, 0)
            j_max = ti.min(j_max, Ny - 1)
            k_min = ti.max(k_min, 0)
            k_max = ti.min(k_max, Nz - 1)

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

        for si, sj, sk in ti.ndrange(nx, ny, nz):

            i = i_min + si
            j = j_min + sj
            k = k_min + sk

            ii = i
            jj = j
            kk = k
            if ti.static(periodic):
                ii = _apply_pbc(i, Nx)
                jj = _apply_pbc(j, Ny)
                kk = _apply_pbc(k, Nz)

            w = wx[si] * wy[sj] * wz[sk]

            ti.atomic_add(weights[ii, jj, kk], w)
            for f in range(F):
                ti.atomic_add(
                    fields[ii, jj, kk, f],
                    quantities[n, f] * w,
                )


# =============================================================================
# Isotropic Kernels
# =============================================================================
@ti.kernel
def _isotropic_1d(
    evaluate_fn: ti.template(),
    sigma_fn: ti.template(),
    kernel_support: ti.f32,
    dim: ti.template(),   # only new line in the signature
    quad_points: ti.template(),
    periodic: ti.template(),
    eta_crit: ti.f32,
    positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 1)
    quantities: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
    smoothing_lengths: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    sample_coords: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (S, 1)
    sample_integrals: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (S,)
    num_samples: ti.i32,
    cellSize_x: ti.f32,
    cellSize_x_inv: ti.f32,
    max_support: ti.f32,
    Nx: ti.i32,
    fields: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (Nx, F)
    weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (Nx)
):
    F = fields.shape[1]
    sig = sigma_fn(dim)

    for n in range(positions.shape[0]):
        x_phys = positions[n, 0]

        hsm_phys = smoothing_lengths[n]
        if ti.static(periodic):
            if hsm_phys > max_support:
                hsm_phys = max_support

        kernel_prefactor = sig / hsm_phys

        x_cell = x_phys * cellSize_x_inv

        eta = hsm_phys * cellSize_x_inv

        wj = particle_weights[n]

        if eta < eta_crit:
            # --- anti-aliasing branch: deposit via precomputed kernel samples ---
            for s in range(num_samples):
                x_sample = x_phys + sample_coords[s, 0] * hsm_phys

                ix, ix_valid = _cell_index_from_pos(
                    x_sample, cellSize_x_inv, Nx, periodic
                )

                if ix_valid:
                    integral = sample_integrals[s]
                    if integral != 0.0:
                        w = wj * integral
                        ti.atomic_add(weights[ix], w)
                        for f in range(F):
                            ti.atomic_add(fields[ix, f], quantities[n, f] * w)

        else:
            # --- cell-quadrature branch, mass-conserving via two-pass renormalization ---
            support_phys = kernel_support * hsm_phys
            support_x_cell = support_phys * cellSize_x_inv

            i_min = ti.floor(x_cell - support_x_cell, int)
            i_max = ti.ceil(x_cell + support_x_cell, int)

            if ti.static(not periodic):
                i_min = ti.max(i_min, 0)
                i_max = ti.min(i_max, Nx - 1)

            # pass 1: total mass over the affected cells (no per-cell storage)
            total_weight = 0.0
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
                total_weight += wj * integral

            if total_weight > 0.0:  # BUGFIX: C++ divides unconditionally
                correction = 1.0 / total_weight

                # pass 2: recompute (cheap) + deposit normalized weight
                for a in range(i_min, i_max + 1):
                    aa = a
                    if ti.static(periodic):
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

                    ti.atomic_add(weights[aa], w_corr)
                    for f in range(F):
                        ti.atomic_add(fields[aa, f], quantities[n, f] * w_corr)


@ti.kernel
def _isotropic_2d(
    evaluate_fn: ti.template(),
    sigma_fn: ti.template(),
    kernel_support: ti.f32,
    dim: ti.template(),   # only new line in the signature
    quad_points: ti.template(),
    periodic: ti.template(),
    eta_crit: ti.f32,
    positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 2)
    quantities: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
    smoothing_lengths: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    sample_coords: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (S, 2)
    sample_integrals: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (S,)
    num_samples: ti.i32,
    cellSize_x: ti.f32,
    cellSize_y: ti.f32,
    cellSize_x_inv: ti.f32,
    cellSize_y_inv: ti.f32,
    max_support: ti.f32,
    Nx: ti.i32,
    Ny: ti.i32,
    fields: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (Nx, Ny, F)
    weights: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (Nx, Ny)
):
    F = fields.shape[2]
    sig = sigma_fn(dim)

    for n in range(positions.shape[0]):
        x_phys = positions[n, 0]
        y_phys = positions[n, 1]

        hsm_phys = smoothing_lengths[n]
        if ti.static(periodic):
            if hsm_phys > max_support:
                hsm_phys = max_support

        kernel_prefactor = sig / (hsm_phys * hsm_phys)

        x_cell = x_phys * cellSize_x_inv
        y_cell = y_phys * cellSize_y_inv

        eta = ti.min(hsm_phys * cellSize_x_inv, hsm_phys * cellSize_y_inv)

        wj = particle_weights[n]

        if eta < eta_crit:
            # --- anti-aliasing branch: deposit via precomputed kernel samples ---
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
                        ti.atomic_add(weights[ix, iy], w)
                        for f in range(F):
                            ti.atomic_add(fields[ix, iy, f], quantities[n, f] * w)

        else:
            # --- cell-quadrature branch, mass-conserving via two-pass renormalization ---
            support_phys = kernel_support * hsm_phys
            support_x_cell = support_phys * cellSize_x_inv
            support_y_cell = support_phys * cellSize_y_inv

            i_min = ti.floor(x_cell - support_x_cell, int)
            i_max = ti.ceil(x_cell + support_x_cell, int)
            j_min = ti.floor(y_cell - support_y_cell, int)
            j_max = ti.ceil(y_cell + support_y_cell, int)

            if ti.static(not periodic):
                i_min = ti.max(i_min, 0)
                i_max = ti.min(i_max, Nx - 1)
                j_min = ti.max(j_min, 0)
                j_max = ti.min(j_max, Ny - 1)

            # pass 1: total mass over the affected cells (no per-cell storage)
            total_weight = 0.0
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
                    total_weight += wj * integral

            if total_weight > 0.0:  # BUGFIX: C++ divides unconditionally
                correction = 1.0 / total_weight

                # pass 2: recompute (cheap) + deposit normalized weight
                for a in range(i_min, i_max + 1):
                    aa = a
                    if ti.static(periodic):
                        aa = _apply_pbc(a, Nx)

                    for b in range(j_min, j_max + 1):
                        bb = b
                        if ti.static(periodic):
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

                        ti.atomic_add(weights[aa, bb], w_corr)
                        for f in range(F):
                            ti.atomic_add(fields[aa, bb, f], quantities[n, f] * w_corr)


@ti.kernel
def _isotropic_3d(
    evaluate_fn: ti.template(),
    sigma_fn: ti.template(),
    kernel_support: ti.f32,
    dim: ti.template(),   # only new line in the signature
    quad_points: ti.template(),
    periodic: ti.template(),
    eta_crit: ti.f32,
    positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 3)
    quantities: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
    smoothing_lengths: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    sample_coords: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (S, 3)
    sample_integrals: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (S,)
    num_samples: ti.i32,
    cellSize_x: ti.f32,
    cellSize_y: ti.f32,
    cellSize_z: ti.f32,
    cellSize_x_inv: ti.f32,
    cellSize_y_inv: ti.f32,
    cellSize_z_inv: ti.f32,
    max_support: ti.f32,
    Nx: ti.i32,
    Ny: ti.i32,
    Nz: ti.i32,
    fields: ti.types.ndarray(dtype=ti.f32, ndim=4),  # (Nx, Ny, Nz, F)
    weights: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (Nx, Ny, Nz)
):
    F = fields.shape[3]
    sig = sigma_fn(dim)

    for n in range(positions.shape[0]):
        x_phys = positions[n, 0]
        y_phys = positions[n, 1]
        z_phys = positions[n, 2]

        hsm_phys = smoothing_lengths[n]
        if ti.static(periodic):
            if hsm_phys > max_support:
                hsm_phys = max_support

        kernel_prefactor = sig / (hsm_phys * hsm_phys * hsm_phys)

        x_cell = x_phys * cellSize_x_inv
        y_cell = y_phys * cellSize_y_inv
        z_cell = z_phys * cellSize_z_inv

        eta = ti.min(
            hsm_phys * cellSize_x_inv,
            ti.min(hsm_phys * cellSize_y_inv, hsm_phys * cellSize_z_inv),
        )
        wj = particle_weights[n]

        if eta < eta_crit:
            # --- anti-aliasing branch: deposit via precomputed kernel samples ---
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
                        ti.atomic_add(weights[ix, iy, iz], w)
                        for f in range(F):
                            ti.atomic_add(fields[ix, iy, iz, f], quantities[n, f] * w)

        else:
            # --- cell-quadrature branch, mass-conserving via two-pass renormalization ---
            support_phys = kernel_support * hsm_phys
            support_x_cell = support_phys * cellSize_x_inv
            support_y_cell = support_phys * cellSize_y_inv
            support_z_cell = support_phys * cellSize_z_inv

            i_min = ti.floor(x_cell - support_x_cell, int)
            i_max = ti.ceil(x_cell + support_x_cell, int)
            j_min = ti.floor(y_cell - support_y_cell, int)
            j_max = ti.ceil(y_cell + support_y_cell, int)
            k_min = ti.floor(z_cell - support_z_cell, int)
            k_max = ti.ceil(z_cell + support_z_cell, int)

            if ti.static(not periodic):
                i_min = ti.max(i_min, 0)
                i_max = ti.min(i_max, Nx - 1)
                j_min = ti.max(j_min, 0)
                j_max = ti.min(j_max, Ny - 1)
                k_min = ti.max(k_min, 0)
                k_max = ti.min(k_max, Nz - 1)

            # pass 1: total mass over the affected cells (no per-cell storage)
            total_weight = 0.0
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
                        total_weight += wj * integral

            if total_weight > 0.0:  # BUGFIX: C++ divides unconditionally
                correction = 1.0 / total_weight

                # pass 2: recompute (cheap) + deposit normalized weight
                for a in range(i_min, i_max + 1):
                    aa = a
                    if ti.static(periodic):
                        aa = _apply_pbc(a, Nx)

                    for b in range(j_min, j_max + 1):
                        bb = b
                        if ti.static(periodic):
                            bb = _apply_pbc(b, Ny)

                        for c in range(k_min, k_max + 1):
                            cc = c
                            if ti.static(periodic):
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

                            ti.atomic_add(weights[aa, bb, cc], w_corr)
                            for f in range(F):
                                ti.atomic_add(
                                    fields[aa, bb, cc, f], quantities[n, f] * w_corr
                                )


# =============================================================================
# Covariant Kernels
# =============================================================================
@ti.kernel
def _covariant_1d(
    evaluate_fn: ti.template(),
    sigma_fn: ti.template(),
    kernel_support: ti.f32,
    dim: ti.template(),   # only new line in the signature
    quad_points: ti.template(),
    periodic: ti.template(),
    eta_crit: ti.f32,
    positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N,1)
    quantities: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N,F)
    hmat_eigvecs: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N,1)
    hmat_eigvals: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N,1)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),
    sample_coords: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (S,1)
    sample_integrals: ti.types.ndarray(dtype=ti.f32, ndim=1),
    num_samples: ti.i32,
    cellSize_x: ti.f32,
    cellSize_x_inv: ti.f32,
    max_support: ti.f32,
    Nx: ti.i32,
    fields: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (Nx,F)
    weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (Nx,)
):
    F = fields.shape[1]
    sig = sigma_fn(dim)

    for n in range(positions.shape[0]):

        x_phys = positions[n, 0]

        # always 1 in 1D, but kept for API symmetry
        e0 = hmat_eigvecs[n, 0]

        eval0 = hmat_eigvals[n, 0]

        if ti.static(periodic):
            if eval0 > max_support:
                eval0 = max_support

        eval0_cell = eval0 * cellSize_x_inv

        eta = eval0_cell
        wj = particle_weights[n]

        if eta < eta_crit:

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

                        ti.atomic_add(weights[ix], w)

                        for f in range(F):
                            ti.atomic_add(
                                fields[ix, f],
                                quantities[n, f] * w,
                            )

        else:

            x_cell = x_phys * cellSize_x_inv

            detH = eval0
            kernel_prefactor = sig / detH

            support_x_cell = kernel_support * ti.abs(eval0_cell)

            i_min = ti.floor(x_cell - support_x_cell, int)
            i_max = ti.ceil(x_cell + support_x_cell, int)

            if ti.static(not periodic):
                i_min = ti.max(i_min, 0)
                i_max = ti.min(i_max, Nx - 1)

            total_weight = 0.0

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

                total_weight += wj * integral

            if total_weight > 0.0:

                correction = 1.0 / total_weight

                for a in range(i_min, i_max + 1):

                    aa = a
                    if ti.static(periodic):
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

                    ti.atomic_add(weights[aa], w_corr)

                    for f in range(F):
                        ti.atomic_add(
                            fields[aa, f],
                            quantities[n, f] * w_corr,
                        )


@ti.kernel
def _covariant_2d(
    evaluate_fn: ti.template(),
    sigma_fn: ti.template(),
    kernel_support: ti.f32,
    dim: ti.template(),   # only new line in the signature
    quad_points: ti.template(),
    periodic: ti.template(),
    eta_crit: ti.f32,
    positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N,2)
    quantities: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N,F)
    hmat_eigvecs: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N,4) column-major
    hmat_eigvals: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N,2)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),
    sample_coords: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (S,2)
    sample_integrals: ti.types.ndarray(dtype=ti.f32, ndim=1),
    num_samples: ti.i32,
    cellSize_x: ti.f32,
    cellSize_y: ti.f32,
    cellSize_x_inv: ti.f32,
    cellSize_y_inv: ti.f32,
    max_support: ti.f32,
    Nx: ti.i32,
    Ny: ti.i32,
    fields: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (Nx,Ny,F)
    weights: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (Nx,Ny)
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

        if ti.static(periodic):
            max_eval = ti.max(eval0, eval1)
            if max_eval > max_support:
                scale = max_support / max_eval
                eval0 *= scale
                eval1 *= scale

        eval0_cell = eval0 * cellSize_x_inv
        eval1_cell = eval1 * cellSize_y_inv

        eta = ti.min(eval0_cell, eval1_cell)

        wj = particle_weights[n]

        if eta < eta_crit:

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
                        ti.atomic_add(weights[ix, iy], w)

                        for f in range(F):
                            ti.atomic_add(
                                fields[ix, iy, f],
                                quantities[n, f] * w,
                            )

        else:

            x_cell = x_phys * cellSize_x_inv
            y_cell = y_phys * cellSize_y_inv

            detH = eval0 * eval1
            kernel_prefactor = sig / detH

            support_x_cell = kernel_support * ti.sqrt(
                (e0x * eval0_cell) ** 2 + (e1x * eval1_cell) ** 2
            )

            support_y_cell = kernel_support * ti.sqrt(
                (e0y * eval0_cell) ** 2 + (e1y * eval1_cell) ** 2
            )

            i_min = ti.floor(x_cell - support_x_cell, int)
            i_max = ti.ceil(x_cell + support_x_cell, int)

            j_min = ti.floor(y_cell - support_y_cell, int)
            j_max = ti.ceil(y_cell + support_y_cell, int)

            if ti.static(not periodic):
                i_min = ti.max(i_min, 0)
                i_max = ti.min(i_max, Nx - 1)
                j_min = ti.max(j_min, 0)
                j_max = ti.min(j_max, Ny - 1)

            total_weight = 0.0

            for a in range(i_min, i_max + 1):
                for b in range(j_min, j_max + 1):

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

                    total_weight += wj * integral

            if total_weight > 0.0:

                correction = 1.0 / total_weight

                for a in range(i_min, i_max + 1):

                    aa = a
                    if ti.static(periodic):
                        aa = _apply_pbc(a, Nx)

                    for b in range(j_min, j_max + 1):

                        bb = b
                        if ti.static(periodic):
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

                        ti.atomic_add(weights[aa, bb], w_corr)

                        for f in range(F):
                            ti.atomic_add(
                                fields[aa, bb, f],
                                quantities[n, f] * w_corr,
                            )


@ti.kernel
def _covariant_3d(
    evaluate_fn: ti.template(),
    sigma_fn: ti.template(),
    kernel_support: ti.f32,
    dim: ti.template(),   # only new line in the signature
    quad_points: ti.template(),
    periodic: ti.template(),
    eta_crit: ti.f32,
    positions: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 3)
    quantities: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, F)
    hmat_eigvecs: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 9) column-major
    hmat_eigvals: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 3)
    particle_weights: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (N,)
    sample_coords: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (S, 3)
    sample_integrals: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (S,)
    num_samples: ti.i32,
    cellSize_x: ti.f32,
    cellSize_y: ti.f32,
    cellSize_z: ti.f32,
    cellSize_x_inv: ti.f32,
    cellSize_y_inv: ti.f32,
    cellSize_z_inv: ti.f32,
    max_support: ti.f32,
    Nx: ti.i32,
    Ny: ti.i32,
    Nz: ti.i32,
    fields: ti.types.ndarray(dtype=ti.f32, ndim=4),  # (Nx, Ny, Nz, F)
    weights: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (Nx, Ny, Nz)
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

        if ti.static(periodic):
            max_eval = ti.max(eval0, ti.max(eval1, eval2))
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

        eta = ti.min(eval0_cell, ti.min(eval1_cell, eval2_cell))
        wj = particle_weights[n]

        if eta < eta_crit:
            # --- anti-aliasing branch: deposit via precomputed kernel samples ---
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
                        ti.atomic_add(weights[ix, iy, iz], w)
                        for f in range(F):
                            ti.atomic_add(fields[ix, iy, iz, f], quantities[n, f] * w)

        else:
            # --- cell-quadrature branch, mass-conserving via two-pass renormalization ---
            x_cell = x_phys * cellSize_x_inv
            y_cell = y_phys * cellSize_y_inv
            z_cell = z_phys * cellSize_z_inv

            detH = eval0 * eval1 * eval2
            kernel_prefactor = sig / detH

            support_x_cell = kernel_support * ti.sqrt(
                (e0x * eval0_cell) ** 2
                + (e1x * eval1_cell) ** 2
                + (e2x * eval2_cell) ** 2
            )
            support_y_cell = kernel_support * ti.sqrt(
                (e0y * eval0_cell) ** 2
                + (e1y * eval1_cell) ** 2
                + (e2y * eval2_cell) ** 2
            )
            support_z_cell = kernel_support * ti.sqrt(
                (e0z * eval0_cell) ** 2
                + (e1z * eval1_cell) ** 2
                + (e2z * eval2_cell) ** 2
            )

            i_min = ti.floor(x_cell - support_x_cell, int)
            i_max = ti.ceil(x_cell + support_x_cell, int)
            j_min = ti.floor(y_cell - support_y_cell, int)
            j_max = ti.ceil(y_cell + support_y_cell, int)
            k_min = ti.floor(z_cell - support_z_cell, int)
            k_max = ti.ceil(z_cell + support_z_cell, int)

            if ti.static(not periodic):
                i_min = ti.max(i_min, 0)
                i_max = ti.min(i_max, Nx - 1)
                j_min = ti.max(j_min, 0)
                j_max = ti.min(j_max, Ny - 1)
                k_min = ti.max(k_min, 0)
                k_max = ti.min(k_max, Nz - 1)

            # pass 1: total mass over the affected cells (no per-cell storage)
            total_weight = 0.0
            for a in range(i_min, i_max + 1):
                for b in range(j_min, j_max + 1):
                    for c in range(k_min, k_max + 1):
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
                        total_weight += wj * integral

            if total_weight > 0.0:  # BUGFIX: C++ divides unconditionally
                correction = 1.0 / total_weight

                # pass 2: recompute (cheap) + deposit normalized weight
                for a in range(i_min, i_max + 1):
                    aa = a
                    if ti.static(periodic):
                        aa = _apply_pbc(a, Nx)

                    for b in range(j_min, j_max + 1):
                        bb = b
                        if ti.static(periodic):
                            bb = _apply_pbc(b, Ny)

                        for c in range(k_min, k_max + 1):
                            cc = c
                            if ti.static(periodic):
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

                            ti.atomic_add(weights[aa, bb, cc], w_corr)
                            for f in range(F):
                                ti.atomic_add(
                                    fields[aa, bb, cc, f], quantities[n, f] * w_corr
                                )


# =============================================================================
# ||                                                                         ||
# ||                               WRAPPERS                                  ||
# ||                                                                         ||
# =============================================================================
_KERNELS = {
    False: {  # static (fixed-stencil) methods
        ("ngp", 1): _ngp_1d,
        ("ngp", 2): _ngp_2d,
        ("ngp", 3): _ngp_3d,
        ("cic", 1): _cic_1d,
        ("cic", 2): _cic_2d,
        ("cic", 3): _cic_3d,
        ("tsc", 1): _tsc_1d,
        ("tsc", 2): _tsc_2d,
        ("tsc", 3): _tsc_3d,
        ("pcs", 1): _pcs_1d,
        ("pcs", 2): _pcs_2d,
        ("pcs", 3): _pcs_3d,
        ("pqs", 1): _pqs_1d,
        ("pqs", 2): _pqs_2d,
        ("pqs", 3): _pqs_3d,
    },
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

FIXED_GRID_KERNELS = ["ngp", "cic", "tsc", "pcs", "pqs"]
ADAPTIVE_STRUCTURES = ["separable", "isotropic", "covariant"]


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
    kernel: Any  # the @ti.kernel to invoke
    args: tuple  # exact positional args for `kernel`

    # -------------------------------------------------------------------
    # Output arrays (also embedded in `args`; kept here too since Taichi
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
    particle_hsml = _as_float32(particle_hsml)

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
    num_kernel_evaluations_per_axis,
    eta_crit,
):
    DIM = particle_positions.shape[1]
    particle_hsml = _as_float32(particle_hsml)

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

    grid = build_kernel_sample_grid(kernel_name, DIM, num_kernel_evaluations_per_axis)
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
    num_kernel_evaluations_per_axis,
    eta_crit,
):
    DIM = particle_positions.shape[1]

    # ti.kernels read the flattened matrix as [e0x,e0y,e0z, e1x,e1y,e1z, ...]
    # i.e. column j (eigenvector j) contiguous -- transpose(0, 2, 1) turns
    # (N, row, col) into (N, col, row) so reshape(-1, D*D) lays out each
    # column contiguously.
    particle_hmat_eigvecs = _as_float32(
        particle_hmat_eigvecs.transpose(0, 2, 1).reshape(-1, DIM * DIM)
    )
    particle_hmat_eigvals = _as_float32(particle_hmat_eigvals)

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

    grid = build_kernel_sample_grid(kernel_name, DIM, num_kernel_evaluations_per_axis)
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
    num_kernel_evaluations_per_axis=5,
    eta_crit=1.0,
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
        Required when ``adaptive=True``. Separable kernels use names such as
        ``"tsc_rect"``; isotropic and covariant kernels are spherical.
        - fixed-grid: `kernel_name` is one of "ngp"/"cic"/"tsc"/"pcs"/"pqs".
        - separable: needs `particle_hsml` (N, D). Integrated exactly;
          `integration_method` is ignored.
        - isotropic: needs `particle_hsml` (N,), a spherical kernel such as
          "cubic_spline", `integration_method`,
          `num_kernel_evaluations_per_axis`, `eta_crit`.
        - covariant: needs `particle_hmat_eigvecs` (N, D, D),
          `particle_hmat_eigvals` (N, D), plus the same kernel/quadrature
          options as "isotropic".

    Returns
    -------
    grid_fields : ndarray, shape (*gridnums, F)
    grid_weights : ndarray, shape (*gridnums,)

    """
    particle_positions = _as_float32(particle_positions)
    particle_fields = _as_float32(particle_fields)
    particle_weights = _as_float32(particle_weights)
    boxsizes = _as_float32(boxsizes)
    dim = particle_positions.shape[-1]

    if not adaptive:
        if structure is not None:
            raise ValueError("structure is only valid when adaptive=True.")
        if kernel_name not in FIXED_GRID_KERNELS:
            raise ValueError(
                f"kernel_name={kernel_name!r} is not a fixed-grid stencil. "
                f"Available: {FIXED_GRID_KERNELS}"
            )
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
            f"Available: {ADAPTIVE_STRUCTURES}"
        )

    missing = [name for name in _REQUIRED_KWARGS[structure] if locals()[name] is None]
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
            num_kernel_evaluations_per_axis=num_kernel_evaluations_per_axis,
            eta_crit=eta_crit,
        )
    else:  # "covariant"
        prep = _prepare_covariant(
            **common,
            particle_hmat_eigvecs=particle_hmat_eigvecs,
            particle_hmat_eigvals=particle_hmat_eigvals,
            integration_method=integration_method,
            num_kernel_evaluations_per_axis=num_kernel_evaluations_per_axis,
            eta_crit=eta_crit,
        )

    prep.kernel(*prep.args)
    return np.moveaxis(prep.grid_fields, -1, 0), prep.grid_weights


# ============================================================
if __name__ == "__main__":
    dim = 3
    N = int(1e6)
    F = 4
    structure = "separable"
    kernel_name = "tsc_rect"

    particle_positions = np.random.uniform(0, 1, (N, dim))
    particle_fields = np.ones((N, F))
    particle_weights = np.ones(N)
    particle_hsml = np.ones((N, dim)) * 0.01
    particle_hmat_eigvecs = np.tile(np.eye(dim), (N, 1, 1))
    particle_hmat_eigvals = np.ones((N, dim)) * 0.01
    boxsizes = np.array([1.0] * dim)
    gridnums = [256] * dim
    periodic = True

    from timeit import repeat

    N_REPEATS = 3
    N_LOOPS = 3

    kwargs = dict(
        particle_positions=particle_positions,
        particle_fields=particle_fields,
        particle_weights=particle_weights,
        particle_hsml=particle_hsml,
        particle_hmat_eigvecs=particle_hmat_eigvecs,
        particle_hmat_eigvals=particle_hmat_eigvals,
        boxsizes=boxsizes,
        gridnums=gridnums,
        periodic=periodic,
        adaptive=True,
        structure=structure,
        kernel_name=kernel_name,
        integration_method="midpoint",
        num_kernel_evaluations_per_axis=5,
        eta_crit=1.0,
    )

    fields, weights = deposit(**kwargs)
    print("fields:", fields.shape)
    print("weights:", weights.shape)

    best = min(
        repeat(
            stmt=lambda: deposit(**kwargs),
            repeat=N_REPEATS,
            number=N_LOOPS,
        )
    )
    print(f"Best runtime: {best / N_LOOPS * 1e3:.3f} ms")
