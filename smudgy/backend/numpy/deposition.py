from itertools import product

import numpy as np


def _as_float32(array):
    """Return a float32 view of ``array`` without copying when possible.

    Parameters
    ----------
    array
        Input array-like object.

    Returns
    -------
    numpy.ndarray
        ``float32`` view or copy of ``array``.

    """
    return np.asarray(array, dtype=np.float32)


# =============================================================================
# NGP
# =============================================================================
def _ngp_1d(
    particle_positions, particle_fields, particle_weights, boxsizes, gridnums, periodic
):
    """Deposit particle particle_fields onto a 1D grid with Nearest-Grid-Point weighting.

    Parameters
    ----------
    particle_positions : ndarray, shape (N, 1)
        Cartesian particle coordinates, where ``N`` is the number of particles.
    particle_fields : ndarray, shape (N, F)
        Per-particle fields to accumulate, with ``F`` fields per particle.
    particle_weights : ndarray, shape (N,)
        Per-particle weights (e.g., masses) to use during deposition.
    boxsizes : array_like of length 1
        Domain sizes per axis, assuming ``[0, boxsize]`` in each dimension.
    gridnums : array_like of length 1
        Number of grid cells for each axis.
    periodic : bool
        Dummy variable for input consistency with other methods

    Returns
    -------
    grid_fields : ndarray
        Accumulated field values on the grid.
    grid_weights : ndarray
        Particle counts per cell (used as normalization weights).

    """
    particle_positions = _as_float32(particle_positions)
    particle_fields = _as_float32(particle_fields)
    particle_weights = _as_float32(particle_weights)
    boxsizes = _as_float32(boxsizes)

    Nx = gridnums[0]
    F = particle_fields.shape[1]

    inv_dx = np.array([Nx], dtype=np.float32) / boxsizes
    grid_pos = (particle_positions * inv_dx).astype(np.int32)

    x_idx = grid_pos[:, 0]

    valid = (0 <= x_idx) & (x_idx < Nx)

    if not np.any(valid):
        return (
            np.zeros((F, Nx), dtype=np.float32),
            np.zeros((Nx,), dtype=np.float32),
        )

    x_idx = x_idx[valid]
    particle_fields = particle_fields[valid]
    particle_weights = particle_weights[valid]

    flat_idx = x_idx

    weighted_particle_fields = particle_fields * particle_weights[:, None]

    grid_fields = np.empty((F, Nx), dtype=np.float32)

    for f in range(F):
        grid_fields[f] = np.bincount(
            flat_idx,
            weights=weighted_particle_fields[:, f],
            minlength=Nx,
        )

    grid_weights = np.bincount(
        flat_idx,
        weights=particle_weights,
        minlength=Nx,
    ).astype(np.float32)

    grid_fields = grid_fields.reshape(F, Nx)
    grid_weights = grid_weights.reshape(Nx)

    return grid_fields, grid_weights


def _ngp_2d(
    particle_positions, particle_fields, particle_weights, boxsizes, gridnums, periodic
):
    """Deposit particle particle_fields onto a 2D grid with Nearest-Grid-Point weighting.

    Parameters
    ----------
    particle_positions : ndarray, shape (N, 2)
        Cartesian particle coordinates, where ``N`` is the number of particles.
    particle_fields : ndarray, shape (N, F)
        Per-particle fields to accumulate, with ``F`` fields per particle.
    particle_weights : ndarray, shape (N,)
        Per-particle weights (e.g., masses) to use during deposition.
    boxsizes : array_like of length 2
        Domain sizes per axis, assuming ``[0, boxsize]`` in each dimension.
    gridnums : array_like of length 2
        Number of grid cells for each axis.
    periodic : bool
        Dummy variable for input consistency with other methods

    Returns
    -------
    grid_fields : ndarray
        Accumulated field values on the grid.
    grid_weights : ndarray
        Particle counts per cell (used as normalization weights).

    """
    particle_positions = _as_float32(particle_positions)
    particle_fields = _as_float32(particle_fields)
    particle_weights = _as_float32(particle_weights)
    boxsizes = _as_float32(boxsizes)

    Nx, Ny = gridnums
    F = particle_fields.shape[1]

    inv_dx = np.array([Nx, Ny], dtype=np.float32) / boxsizes
    grid_pos = (particle_positions * inv_dx).astype(np.int32)

    x_idx = grid_pos[:, 0]
    y_idx = grid_pos[:, 1]

    valid = (0 <= x_idx) & (x_idx < Nx) & (0 <= y_idx) & (y_idx < Ny)

    if not np.any(valid):
        return (
            np.zeros((F, Nx, Ny), dtype=np.float32),
            np.zeros((Nx, Ny), dtype=np.float32),
        )

    x_idx = x_idx[valid]
    y_idx = y_idx[valid]
    particle_fields = particle_fields[valid]
    particle_weights = particle_weights[valid]

    flat_idx = x_idx * Ny + y_idx

    weighted_particle_fields = particle_fields * particle_weights[:, None]

    grid_fields = np.empty((F, Nx * Ny), dtype=np.float32)

    for f in range(F):
        grid_fields[f] = np.bincount(
            flat_idx,
            weights=weighted_particle_fields[:, f],
            minlength=Nx * Ny,
        )

    grid_weights = np.bincount(
        flat_idx,
        weights=particle_weights,
        minlength=Nx * Ny,
    ).astype(np.float32)

    grid_fields = grid_fields.reshape(F, Nx, Ny)
    grid_weights = grid_weights.reshape(Nx, Ny)

    return grid_fields, grid_weights


def _ngp_3d(
    particle_positions, particle_fields, particle_weights, boxsizes, gridnums, periodic
):
    """Deposit particle particle_fields onto a 3D grid with Nearest-Grid-Point weighting.

    Parameters
    ----------
    particle_positions : ndarray, shape (N, 3)
        Cartesian particle coordinates, where ``N`` is the number of particles.
    particle_fields : ndarray, shape (N, F)
        Per-particle fields to accumulate, with ``F`` fields per particle.
    particle_weights : ndarray, shape (N,)
        Per-particle weights (e.g., masses) to use during deposition.
    boxsizes : array_like of length 3
        Domain sizes per axis, assuming ``[0, boxsize]`` in each dimension.
    gridnums : array_like of length 3
        Number of grid cells for each axis.
    periodic : bool
        Dummy variable for input consistency with other methods

    Returns
    -------
    grid_fields : ndarray
        Accumulated field values on the grid.
    grid_weights : ndarray
        Particle counts per cell (used as normalization weights).

    """
    particle_positions = _as_float32(particle_positions)
    particle_fields = _as_float32(particle_fields)
    particle_weights = _as_float32(particle_weights)
    boxsizes = _as_float32(boxsizes)

    Nx, Ny, Nz = gridnums
    F = particle_fields.shape[1]

    inv_dx = np.array([Nx, Ny, Nz], dtype=np.float32) / boxsizes
    grid_pos = (particle_positions * inv_dx).astype(np.int32)

    x_idx = grid_pos[:, 0]
    y_idx = grid_pos[:, 1]
    z_idx = grid_pos[:, 2]

    valid = (
        (0 <= x_idx)
        & (x_idx < Nx)
        & (0 <= y_idx)
        & (y_idx < Ny)
        & (0 <= z_idx)
        & (z_idx < Nz)
    )

    if not np.any(valid):
        return (
            np.zeros((F, Nx, Ny, Nz), dtype=np.float32),
            np.zeros((Nx, Ny, Nz), dtype=np.float32),
        )

    x_idx = x_idx[valid]
    y_idx = y_idx[valid]
    z_idx = z_idx[valid]
    particle_fields = particle_fields[valid]
    particle_weights = particle_weights[valid]

    flat_idx = x_idx * Ny * Nz + y_idx * Nz + z_idx

    weighted_particle_fields = particle_fields * particle_weights[:, None]

    grid_fields = np.empty((F, Nx * Ny * Nz), dtype=np.float32)

    for f in range(F):
        grid_fields[f] = np.bincount(
            flat_idx,
            weights=weighted_particle_fields[:, f],
            minlength=Nx * Ny * Nz,
        )

    grid_weights = np.bincount(
        flat_idx,
        weights=particle_weights,
        minlength=Nx * Ny * Nz,
    ).astype(np.float32)

    grid_fields = grid_fields.reshape(F, Nx, Ny, Nz)
    grid_weights = grid_weights.reshape(Nx, Ny, Nz)

    return grid_fields, grid_weights


# =============================================================================
# CIC
# =============================================================================
def _cic_1d(
    particle_positions,
    particle_fields,
    particle_weights,
    boxsizes,
    gridnums,
    periodic,
):
    """Deposit particle particle_fields onto a 1D grid with Nearest-Grid-Point weighting.

    Parameters
    ----------
    particle_positions : ndarray, shape (N, 1)
        Cartesian particle coordinates, where ``N`` is the number of particles.
    particle_fields : ndarray, shape (N, F)
        Per-particle fields to accumulate, with ``F`` fields per particle.
    particle_weights : ndarray, shape (N,)
        Per-particle weights (e.g., masses) to use during deposition.
    boxsizes : array_like of length 1
        Domain sizes per axis, assuming ``[0, boxsize]`` in each dimension.
    gridnums : array_like of length 1
        Number of grid cells for each axis.
    periodic : bool
        Whether to wrap particles that leave the domain (applies to all axes).

    Returns
    -------
    grid_fields : ndarray
        Accumulated field values on the grid.
    grid_weights : ndarray
        Particle counts per cell (used as normalization weights).

    """
    particle_positions = _as_float32(particle_positions)
    particle_fields = _as_float32(particle_fields)
    particle_weights = _as_float32(particle_weights)
    boxsizes = _as_float32(boxsizes)

    Nx = gridnums[0]
    F = particle_fields.shape[1]

    inv_dx = np.array([Nx], dtype=np.float32) / boxsizes

    # Cell-centered coordinates
    grid_pos = particle_positions * inv_dx - 0.5

    if periodic:
        grid_pos = np.mod(grid_pos, [Nx])
    else:
        eps = 1e-6
        grid_pos = np.clip(
            grid_pos,
            0.0,
            [Nx - 1 - eps],
        )

    base = np.floor(grid_pos).astype(np.int32)
    frac = grid_pos - base

    x0 = base[:, 0]

    x1 = x0 + 1

    fx = frac[:, 0]

    wx0 = 1.0 - fx
    wx1 = fx

    flat_size = Nx

    grid_fields = np.zeros((F, flat_size), dtype=np.float32)
    grid_weights = np.zeros(flat_size, dtype=np.float32)

    neighbors = (
        (x0, wx0),
        (x1, wx1),
    )

    weighted_fields = particle_fields * particle_weights[:, None]

    for xi, w in neighbors:

        if periodic:
            xi = xi % Nx
            valid = slice(None)

        else:
            valid = (0 <= xi) & (xi < Nx)

            if not np.any(valid):
                continue

            xi = xi[valid]
            w = w[valid]

        flat_idx = xi

        deposit_weight = particle_weights if periodic else particle_weights[valid]
        deposit_field = weighted_fields if periodic else weighted_fields[valid]

        np.add.at(
            grid_weights,
            flat_idx,
            deposit_weight * w,
        )

        for f in range(F):
            np.add.at(
                grid_fields[f],
                flat_idx,
                deposit_field[:, f] * w,
            )

    grid_fields = grid_fields.reshape(F, Nx)
    grid_weights = grid_weights.reshape(
        Nx,
    )

    return grid_fields, grid_weights


def _cic_2d(
    particle_positions,
    particle_fields,
    particle_weights,
    boxsizes,
    gridnums,
    periodic,
):
    """Deposit particle particle_fields onto a 2D grid with Nearest-Grid-Point weighting.

    Parameters
    ----------
    particle_positions : ndarray, shape (N, 3)
        Cartesian particle coordinates, where ``N`` is the number of particles.
    particle_fields : ndarray, shape (N, F)
        Per-particle fields to accumulate, with ``F`` fields per particle.
    particle_weights : ndarray, shape (N,)
        Per-particle weights (e.g., masses) to use during deposition.
    boxsizes : array_like of length 3
        Domain sizes per axis, assuming ``[0, boxsize]`` in each dimension.
    gridnums : array_like of length 3
        Number of grid cells for each axis.
    periodic : bool
        Whether to wrap particles that leave the domain (applies to all axes).

    Returns
    -------
    grid_fields : ndarray
        Accumulated field values on the grid.
    grid_weights : ndarray
        Particle counts per cell (used as normalization weights).

    """
    particle_positions = _as_float32(particle_positions)
    particle_fields = _as_float32(particle_fields)
    particle_weights = _as_float32(particle_weights)
    boxsizes = _as_float32(boxsizes)

    Nx, Ny = gridnums
    F = particle_fields.shape[1]

    inv_dx = np.array([Nx, Ny], dtype=np.float32) / boxsizes

    # Cell-centered coordinates
    grid_pos = particle_positions * inv_dx - 0.5

    if periodic:
        grid_pos = np.mod(grid_pos, [Nx, Ny])
    else:
        eps = 1e-6
        grid_pos = np.clip(
            grid_pos,
            0.0,
            [Nx - 1 - eps, Ny - 1 - eps],
        )

    base = np.floor(grid_pos).astype(np.int32)
    frac = grid_pos - base

    x0 = base[:, 0]
    y0 = base[:, 1]

    x1 = x0 + 1
    y1 = y0 + 1

    fx = frac[:, 0]
    fy = frac[:, 1]

    wx0 = 1.0 - fx
    wx1 = fx

    wy0 = 1.0 - fy
    wy1 = fy

    flat_size = Nx * Ny

    grid_fields = np.zeros((F, flat_size), dtype=np.float32)
    grid_weights = np.zeros(flat_size, dtype=np.float32)

    neighbors = (
        (x0, y0, wx0 * wy0),
        (x1, y0, wx1 * wy0),
        (x0, y1, wx0 * wy1),
        (x1, y1, wx1 * wy1),
    )

    weighted_fields = particle_fields * particle_weights[:, None]

    for xi, yi, w in neighbors:

        if periodic:
            xi = xi % Nx
            yi = yi % Ny
            valid = slice(None)

        else:
            valid = (0 <= xi) & (xi < Nx) & (0 <= yi) & (yi < Ny)

            if not np.any(valid):
                continue

            xi = xi[valid]
            yi = yi[valid]
            w = w[valid]

        flat_idx = xi * Ny + yi

        deposit_weight = particle_weights if periodic else particle_weights[valid]
        deposit_field = weighted_fields if periodic else weighted_fields[valid]

        np.add.at(
            grid_weights,
            flat_idx,
            deposit_weight * w,
        )

        for f in range(F):
            np.add.at(
                grid_fields[f],
                flat_idx,
                deposit_field[:, f] * w,
            )

    grid_fields = grid_fields.reshape(F, Nx, Ny)
    grid_weights = grid_weights.reshape(Nx, Ny)

    return grid_fields, grid_weights


def _cic_3d(
    particle_positions,
    particle_fields,
    particle_weights,
    boxsizes,
    gridnums,
    periodic,
):
    particle_positions = _as_float32(particle_positions)
    particle_fields = _as_float32(particle_fields)
    particle_weights = _as_float32(particle_weights)
    boxsizes = _as_float32(boxsizes)

    Nx, Ny, Nz = gridnums
    F = particle_fields.shape[1]

    inv_dx = np.array([Nx, Ny, Nz], dtype=np.float32) / boxsizes

    # Cell-centered coordinates
    grid_pos = particle_positions * inv_dx - 0.5

    if periodic:
        grid_pos = np.mod(grid_pos, [Nx, Ny, Nz])
    else:
        eps = 1e-6
        grid_pos = np.clip(
            grid_pos,
            0.0,
            [Nx - 1 - eps, Ny - 1 - eps, Nz - 1 - eps],
        )

    base = np.floor(grid_pos).astype(np.int32)
    frac = grid_pos - base

    x0 = base[:, 0]
    y0 = base[:, 1]
    z0 = base[:, 2]

    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    fx = frac[:, 0]
    fy = frac[:, 1]
    fz = frac[:, 2]

    wx0 = 1.0 - fx
    wx1 = fx

    wy0 = 1.0 - fy
    wy1 = fy

    wz0 = 1.0 - fz
    wz1 = fz

    flat_size = Nx * Ny * Nz

    grid_fields = np.zeros((F, flat_size), dtype=np.float32)
    grid_weights = np.zeros(flat_size, dtype=np.float32)

    neighbors = (
        (x0, y0, z0, wx0 * wy0 * wz0),
        (x1, y0, z0, wx1 * wy0 * wz0),
        (x0, y1, z0, wx0 * wy1 * wz0),
        (x1, y1, z0, wx1 * wy1 * wz0),
        (x0, y0, z1, wx0 * wy0 * wz1),
        (x1, y0, z1, wx1 * wy0 * wz1),
        (x0, y1, z1, wx0 * wy1 * wz1),
        (x1, y1, z1, wx1 * wy1 * wz1),
    )

    weighted_fields = particle_fields * particle_weights[:, None]

    for xi, yi, zi, w in neighbors:

        if periodic:
            xi = xi % Nx
            yi = yi % Ny
            zi = zi % Nz
            valid = slice(None)

        else:
            valid = (
                (0 <= xi) & (xi < Nx) & (0 <= yi) & (yi < Ny) & (0 <= zi) & (zi < Nz)
            )

            if not np.any(valid):
                continue

            xi = xi[valid]
            yi = yi[valid]
            zi = zi[valid]
            w = w[valid]

        flat_idx = (xi * Ny + yi) * Nz + zi

        deposit_weight = particle_weights if periodic else particle_weights[valid]
        deposit_fields = weighted_fields if periodic else weighted_fields[valid]

        np.add.at(
            grid_weights,
            flat_idx,
            deposit_weight * w,
        )

        for f in range(F):
            np.add.at(
                grid_fields[f],
                flat_idx,
                deposit_fields[:, f] * w,
            )

    grid_fields = grid_fields.reshape(F, Nx, Ny, Nz)
    grid_weights = grid_weights.reshape(Nx, Ny, Nz)

    return grid_fields, grid_weights


# =============================================================================
# TSC
# =============================================================================
def _tsc_1d(
    particle_positions,
    particle_fields,
    particle_weights,
    boxsizes,
    gridnums,
    periodic,
):
    particle_positions = _as_float32(particle_positions)
    particle_fields = _as_float32(particle_fields)
    particle_weights = _as_float32(particle_weights)
    boxsizes = _as_float32(boxsizes)

    Nx = gridnums[0]
    F = particle_fields.shape[1]

    inv_dx = np.array([Nx], dtype=np.float32) / boxsizes

    # Cell-centered coordinates
    grid_pos = particle_positions * inv_dx - 0.5

    if periodic:
        grid_pos = np.mod(grid_pos, [Nx])

    ix = np.rint(grid_pos[:, 0]).astype(np.int32)
    dx = grid_pos[:, 0] - ix

    def tsc_weights(d):
        wm = 0.5 * (0.5 - d) ** 2
        w0 = 0.75 - d**2
        wp = 0.5 * (0.5 + d) ** 2
        return wm, w0, wp

    wxm, wx0, wxp = tsc_weights(dx)
    wx = (wxm, wx0, wxp)

    ixs = (ix - 1, ix, ix + 1)
    flat_size = Nx

    grid_fields = np.zeros((F, flat_size), dtype=np.float32)
    grid_weights = np.zeros(flat_size, dtype=np.float32)

    weighted_fields = particle_fields * particle_weights[:, None]

    for ox in range(3):

        xi = ixs[ox]
        w = wx[ox]

        if periodic:
            xi = xi % Nx
            valid = slice(None)
        else:
            valid = (0 <= xi) & (xi < Nx)

            if not np.any(valid):
                continue

            xi = xi[valid]
            w = w[valid]

        flat_idx = xi

        deposit_weight = particle_weights if periodic else particle_weights[valid]
        deposit_fields = weighted_fields if periodic else weighted_fields[valid]

        np.add.at(grid_weights, flat_idx, deposit_weight * w)

        for f in range(F):
            np.add.at(
                grid_fields[f],
                flat_idx,
                deposit_fields[:, f] * w,
            )

    return (
        grid_fields.reshape(F, Nx),
        grid_weights.reshape(
            Nx,
        ),
    )


def _tsc_2d(
    particle_positions,
    particle_fields,
    particle_weights,
    boxsizes,
    gridnums,
    periodic,
):
    particle_positions = _as_float32(particle_positions)
    particle_fields = _as_float32(particle_fields)
    particle_weights = _as_float32(particle_weights)
    boxsizes = _as_float32(boxsizes)

    Nx, Ny = gridnums
    F = particle_fields.shape[1]

    inv_dx = np.array([Nx, Ny], dtype=np.float32) / boxsizes

    # Cell-centered coordinates
    grid_pos = particle_positions * inv_dx - 0.5

    if periodic:
        grid_pos = np.mod(grid_pos, [Nx, Ny])

    ix = np.rint(grid_pos[:, 0]).astype(np.int32)
    iy = np.rint(grid_pos[:, 1]).astype(np.int32)

    dx = grid_pos[:, 0] - ix
    dy = grid_pos[:, 1] - iy

    def tsc_weights(d):
        wm = 0.5 * (0.5 - d) ** 2
        w0 = 0.75 - d**2
        wp = 0.5 * (0.5 + d) ** 2
        return wm, w0, wp

    wxm, wx0, wxp = tsc_weights(dx)
    wym, wy0, wyp = tsc_weights(dy)

    wx = (wxm, wx0, wxp)
    wy = (wym, wy0, wyp)

    ixs = (ix - 1, ix, ix + 1)
    iys = (iy - 1, iy, iy + 1)

    flat_size = Nx * Ny

    grid_fields = np.zeros((F, flat_size), dtype=np.float32)
    grid_weights = np.zeros(flat_size, dtype=np.float32)

    weighted_fields = particle_fields * particle_weights[:, None]

    for ox, oy in product(range(3), repeat=2):

        xi = ixs[ox]
        yi = iys[oy]
        w = wx[ox] * wy[oy]

        if periodic:
            xi = xi % Nx
            yi = yi % Ny
            valid = slice(None)
        else:
            valid = (0 <= xi) & (xi < Nx) & (0 <= yi) & (yi < Ny)

            if not np.any(valid):
                continue

            xi = xi[valid]
            yi = yi[valid]
            w = w[valid]

        flat_idx = xi * Ny + yi

        deposit_weight = particle_weights if periodic else particle_weights[valid]
        deposit_fields = weighted_fields if periodic else weighted_fields[valid]

        np.add.at(grid_weights, flat_idx, deposit_weight * w)

        for f in range(F):
            np.add.at(
                grid_fields[f],
                flat_idx,
                deposit_fields[:, f] * w,
            )

    return (
        grid_fields.reshape(F, Nx, Ny),
        grid_weights.reshape(Nx, Ny),
    )


def _tsc_3d(
    particle_positions,
    particle_fields,
    particle_weights,
    boxsizes,
    gridnums,
    periodic,
):
    particle_positions = _as_float32(particle_positions)
    particle_fields = _as_float32(particle_fields)
    particle_weights = _as_float32(particle_weights)
    boxsizes = _as_float32(boxsizes)

    Nx, Ny, Nz = gridnums
    F = particle_fields.shape[1]

    inv_dx = np.array([Nx, Ny, Nz], dtype=np.float32) / boxsizes

    # Cell-centered coordinates
    grid_pos = particle_positions * inv_dx - 0.5

    if periodic:
        grid_pos = np.mod(grid_pos, [Nx, Ny, Nz])

    ix = np.rint(grid_pos[:, 0]).astype(np.int32)
    iy = np.rint(grid_pos[:, 1]).astype(np.int32)
    iz = np.rint(grid_pos[:, 2]).astype(np.int32)

    dx = grid_pos[:, 0] - ix
    dy = grid_pos[:, 1] - iy
    dz = grid_pos[:, 2] - iz

    def tsc_weights(d):
        wm = 0.5 * (0.5 - d) ** 2
        w0 = 0.75 - d**2
        wp = 0.5 * (0.5 + d) ** 2
        return wm, w0, wp

    wxm, wx0, wxp = tsc_weights(dx)
    wym, wy0, wyp = tsc_weights(dy)
    wzm, wz0, wzp = tsc_weights(dz)

    wx = (wxm, wx0, wxp)
    wy = (wym, wy0, wyp)
    wz = (wzm, wz0, wzp)

    ixs = (ix - 1, ix, ix + 1)
    iys = (iy - 1, iy, iy + 1)
    izs = (iz - 1, iz, iz + 1)

    flat_size = Nx * Ny * Nz

    grid_fields = np.zeros((F, flat_size), dtype=np.float32)
    grid_weights = np.zeros(flat_size, dtype=np.float32)

    weighted_fields = particle_fields * particle_weights[:, None]

    for ox, oy, oz in product(range(3), repeat=3):

        xi = ixs[ox]
        yi = iys[oy]
        zi = izs[oz]

        w = wx[ox] * wy[oy] * wz[oz]

        if periodic:
            xi = xi % Nx
            yi = yi % Ny
            zi = zi % Nz
            valid = slice(None)
        else:
            valid = (
                (0 <= xi) & (xi < Nx) & (0 <= yi) & (yi < Ny) & (0 <= zi) & (zi < Nz)
            )

            if not np.any(valid):
                continue

            xi = xi[valid]
            yi = yi[valid]
            zi = zi[valid]
            w = w[valid]

        flat_idx = (xi * Ny + yi) * Nz + zi

        deposit_weight = particle_weights if periodic else particle_weights[valid]
        deposit_fields = weighted_fields if periodic else weighted_fields[valid]

        np.add.at(grid_weights, flat_idx, deposit_weight * w)

        for f in range(F):
            np.add.at(
                grid_fields[f],
                flat_idx,
                deposit_fields[:, f] * w,
            )

    return (
        grid_fields.reshape(F, Nx, Ny, Nz),
        grid_weights.reshape(Nx, Ny, Nz),
    )


# =============================================================================
# PCS
# =============================================================================
def _pcs_1d(
    particle_positions,
    particle_fields,
    particle_weights,
    boxsizes,
    gridnums,
    periodic,
):
    particle_positions = _as_float32(particle_positions)
    particle_fields = _as_float32(particle_fields)
    particle_weights = _as_float32(particle_weights)
    boxsizes = _as_float32(boxsizes)

    Nx = gridnums[0]
    F = particle_fields.shape[1]

    inv_dx = np.array([Nx], dtype=np.float32) / boxsizes

    # Cell-centered coordinates
    grid_pos = particle_positions * inv_dx - 0.5

    if periodic:
        grid_pos = np.mod(grid_pos, [Nx])

    ix = np.floor(grid_pos[:, 0]).astype(np.int32)
    dx = grid_pos[:, 0] - ix

    def pcs_weights(d):
        w0 = ((1.0 - d) ** 3) / 6.0
        w1 = (4.0 - 6.0 * d**2 + 3.0 * d**3) / 6.0
        w2 = (1.0 + 3.0 * d + 3.0 * d**2 - 3.0 * d**3) / 6.0
        w3 = d**3 / 6.0
        return w0, w1, w2, w3

    wx0, wx1, wx2, wx3 = pcs_weights(dx)
    wx = (wx0, wx1, wx2, wx3)
    ixs = (ix - 1, ix, ix + 1, ix + 2)

    flat_size = Nx

    grid_fields = np.zeros((F, flat_size), dtype=np.float32)
    grid_weights = np.zeros(flat_size, dtype=np.float32)

    weighted_fields = particle_fields * particle_weights[:, None]

    for ox in range(4):

        xi = ixs[ox]
        w = wx[ox]

        if periodic:
            xi = xi % Nx
            valid = slice(None)
        else:
            valid = (0 <= xi) & (xi < Nx)

            if not np.any(valid):
                continue

            xi = xi[valid]
            w = w[valid]

        flat_idx = xi

        deposit_weight = particle_weights if periodic else particle_weights[valid]
        deposit_fields = weighted_fields if periodic else weighted_fields[valid]

        np.add.at(grid_weights, flat_idx, deposit_weight * w)

        for f in range(F):
            np.add.at(
                grid_fields[f],
                flat_idx,
                deposit_fields[:, f] * w,
            )

    return (
        grid_fields.reshape(F, Nx),
        grid_weights.reshape(
            Nx,
        ),
    )


def _pcs_2d(
    particle_positions,
    particle_fields,
    particle_weights,
    boxsizes,
    gridnums,
    periodic,
):
    particle_positions = _as_float32(particle_positions)
    particle_fields = _as_float32(particle_fields)
    particle_weights = _as_float32(particle_weights)
    boxsizes = _as_float32(boxsizes)

    Nx, Ny = gridnums
    F = particle_fields.shape[1]

    inv_dx = np.array([Nx, Ny], dtype=np.float32) / boxsizes

    # Cell-centered coordinates
    grid_pos = particle_positions * inv_dx - 0.5

    if periodic:
        grid_pos = np.mod(grid_pos, [Nx, Ny])

    ix = np.floor(grid_pos[:, 0]).astype(np.int32)
    iy = np.floor(grid_pos[:, 1]).astype(np.int32)

    dx = grid_pos[:, 0] - ix
    dy = grid_pos[:, 1] - iy

    def pcs_weights(d):
        w0 = ((1.0 - d) ** 3) / 6.0
        w1 = (4.0 - 6.0 * d**2 + 3.0 * d**3) / 6.0
        w2 = (1.0 + 3.0 * d + 3.0 * d**2 - 3.0 * d**3) / 6.0
        w3 = d**3 / 6.0
        return w0, w1, w2, w3

    wx0, wx1, wx2, wx3 = pcs_weights(dx)
    wy0, wy1, wy2, wy3 = pcs_weights(dy)

    wx = (wx0, wx1, wx2, wx3)
    wy = (wy0, wy1, wy2, wy3)

    ixs = (ix - 1, ix, ix + 1, ix + 2)
    iys = (iy - 1, iy, iy + 1, iy + 2)

    flat_size = Nx * Ny

    grid_fields = np.zeros((F, flat_size), dtype=np.float32)
    grid_weights = np.zeros(flat_size, dtype=np.float32)

    weighted_fields = particle_fields * particle_weights[:, None]

    for ox, oy in product(range(4), repeat=2):

        xi = ixs[ox]
        yi = iys[oy]
        w = wx[ox] * wy[oy]

        if periodic:
            xi = xi % Nx
            yi = yi % Ny
            valid = slice(None)
        else:
            valid = (0 <= xi) & (xi < Nx) & (0 <= yi) & (yi < Ny)

            if not np.any(valid):
                continue

            xi = xi[valid]
            yi = yi[valid]
            w = w[valid]

        flat_idx = xi * Ny + yi

        deposit_weight = particle_weights if periodic else particle_weights[valid]
        deposit_fields = weighted_fields if periodic else weighted_fields[valid]

        np.add.at(grid_weights, flat_idx, deposit_weight * w)

        for f in range(F):
            np.add.at(
                grid_fields[f],
                flat_idx,
                deposit_fields[:, f] * w,
            )

    return (
        grid_fields.reshape(F, Nx, Ny),
        grid_weights.reshape(Nx, Ny),
    )


def _pcs_3d(
    particle_positions,
    particle_fields,
    particle_weights,
    boxsizes,
    gridnums,
    periodic,
):
    particle_positions = _as_float32(particle_positions)
    particle_fields = _as_float32(particle_fields)
    particle_weights = _as_float32(particle_weights)
    boxsizes = _as_float32(boxsizes)

    Nx, Ny, Nz = gridnums
    F = particle_fields.shape[1]

    inv_dx = np.array([Nx, Ny, Nz], dtype=np.float32) / boxsizes

    # Cell-centered coordinates
    grid_pos = particle_positions * inv_dx - 0.5

    if periodic:
        grid_pos = np.mod(grid_pos, [Nx, Ny, Nz])

    ix = np.floor(grid_pos[:, 0]).astype(np.int32)
    iy = np.floor(grid_pos[:, 1]).astype(np.int32)
    iz = np.floor(grid_pos[:, 2]).astype(np.int32)

    dx = grid_pos[:, 0] - ix
    dy = grid_pos[:, 1] - iy
    dz = grid_pos[:, 2] - iz

    def pcs_weights(d):
        w0 = ((1.0 - d) ** 3) / 6.0
        w1 = (4.0 - 6.0 * d**2 + 3.0 * d**3) / 6.0
        w2 = (1.0 + 3.0 * d + 3.0 * d**2 - 3.0 * d**3) / 6.0
        w3 = d**3 / 6.0
        return w0, w1, w2, w3

    wx0, wx1, wx2, wx3 = pcs_weights(dx)
    wy0, wy1, wy2, wy3 = pcs_weights(dy)
    wz0, wz1, wz2, wz3 = pcs_weights(dz)

    wx = (wx0, wx1, wx2, wx3)
    wy = (wy0, wy1, wy2, wy3)
    wz = (wz0, wz1, wz2, wz3)

    ixs = (ix - 1, ix, ix + 1, ix + 2)
    iys = (iy - 1, iy, iy + 1, iy + 2)
    izs = (iz - 1, iz, iz + 1, iz + 2)

    flat_size = Nx * Ny * Nz

    grid_fields = np.zeros((F, flat_size), dtype=np.float32)
    grid_weights = np.zeros(flat_size, dtype=np.float32)

    weighted_fields = particle_fields * particle_weights[:, None]

    for ox, oy, oz in product(range(4), repeat=3):

        xi = ixs[ox]
        yi = iys[oy]
        zi = izs[oz]

        w = wx[ox] * wy[oy] * wz[oz]

        if periodic:
            xi = xi % Nx
            yi = yi % Ny
            zi = zi % Nz
            valid = slice(None)
        else:
            valid = (
                (0 <= xi) & (xi < Nx) & (0 <= yi) & (yi < Ny) & (0 <= zi) & (zi < Nz)
            )

            if not np.any(valid):
                continue

            xi = xi[valid]
            yi = yi[valid]
            zi = zi[valid]
            w = w[valid]

        flat_idx = (xi * Ny + yi) * Nz + zi

        deposit_weight = particle_weights if periodic else particle_weights[valid]
        deposit_fields = weighted_fields if periodic else weighted_fields[valid]

        np.add.at(grid_weights, flat_idx, deposit_weight * w)

        for f in range(F):
            np.add.at(
                grid_fields[f],
                flat_idx,
                deposit_fields[:, f] * w,
            )

    return (
        grid_fields.reshape(F, Nx, Ny, Nz),
        grid_weights.reshape(Nx, Ny, Nz),
    )


# =============================================================================
# PQS
# =============================================================================
def _pqs_1d(
    particle_positions,
    particle_fields,
    particle_weights,
    boxsizes,
    gridnums,
    periodic,
):
    particle_positions = _as_float32(particle_positions)
    particle_fields = _as_float32(particle_fields)
    particle_weights = _as_float32(particle_weights)
    boxsizes = _as_float32(boxsizes)

    Nx = gridnums[0]
    F = particle_fields.shape[1]

    inv_dx = np.array([Nx], dtype=np.float32) / boxsizes

    # Cell-centered coordinates
    grid_pos = particle_positions * inv_dx - 0.5

    if periodic:
        grid_pos = np.mod(grid_pos, [Nx])

    ix = np.floor(grid_pos[:, 0]).astype(np.int32)

    dx = grid_pos[:, 0] - ix

    def pqs_weights(d):
        w0 = (1.0 - d) ** 5 / 120.0
        w1 = ((2.0 - d) ** 5 - 6.0 * (1.0 - d) ** 5) / 120.0
        w2 = ((3.0 - d) ** 5 - 6.0 * (2.0 - d) ** 5 + 15.0 * (1.0 - d) ** 5) / 120.0
        w3 = ((2.0 + d) ** 5 - 6.0 * (1.0 + d) ** 5 + 15.0 * d**5) / 120.0
        w4 = ((1.0 + d) ** 5 - 6.0 * d**5) / 120.0
        w5 = d**5 / 120.0
        return w0, w1, w2, w3, w4, w5

    wx0, wx1, wx2, wx3, wx4, wx5 = pqs_weights(dx)
    wx = (wx0, wx1, wx2, wx3, wx4, wx5)
    ixs = (ix - 2, ix - 1, ix, ix + 1, ix + 2, ix + 3)
    flat_size = Nx

    grid_fields = np.zeros((F, flat_size), dtype=np.float32)
    grid_weights = np.zeros(flat_size, dtype=np.float32)

    weighted_fields = particle_fields * particle_weights[:, None]

    for ox in range(6):
        xi = ixs[ox]
        w = wx[ox]

        if periodic:
            xi = xi % Nx
            valid = slice(None)
        else:
            valid = (0 <= xi) & (xi < Nx)

            if not np.any(valid):
                continue

            xi = xi[valid]
            w = w[valid]

        flat_idx = xi

        deposit_weight = particle_weights if periodic else particle_weights[valid]
        deposit_fields = weighted_fields if periodic else weighted_fields[valid]

        np.add.at(grid_weights, flat_idx, deposit_weight * w)

        for f in range(F):
            np.add.at(
                grid_fields[f],
                flat_idx,
                deposit_fields[:, f] * w,
            )

    return (
        grid_fields.reshape(F, Nx),
        grid_weights.reshape(
            Nx,
        ),
    )


def _pqs_2d(
    particle_positions,
    particle_fields,
    particle_weights,
    boxsizes,
    gridnums,
    periodic,
):
    particle_positions = _as_float32(particle_positions)
    particle_fields = _as_float32(particle_fields)
    particle_weights = _as_float32(particle_weights)
    boxsizes = _as_float32(boxsizes)

    Nx, Ny = gridnums
    F = particle_fields.shape[1]

    inv_dx = np.array([Nx, Ny], dtype=np.float32) / boxsizes

    # Cell-centered coordinates
    grid_pos = particle_positions * inv_dx - 0.5

    if periodic:
        grid_pos = np.mod(grid_pos, [Nx, Ny])

    ix = np.floor(grid_pos[:, 0]).astype(np.int32)
    iy = np.floor(grid_pos[:, 1]).astype(np.int32)

    dx = grid_pos[:, 0] - ix
    dy = grid_pos[:, 1] - iy

    def pqs_weights(d):
        w0 = (1.0 - d) ** 5 / 120.0
        w1 = ((2.0 - d) ** 5 - 6.0 * (1.0 - d) ** 5) / 120.0
        w2 = ((3.0 - d) ** 5 - 6.0 * (2.0 - d) ** 5 + 15.0 * (1.0 - d) ** 5) / 120.0
        w3 = ((2.0 + d) ** 5 - 6.0 * (1.0 + d) ** 5 + 15.0 * d**5) / 120.0
        w4 = ((1.0 + d) ** 5 - 6.0 * d**5) / 120.0
        w5 = d**5 / 120.0
        return w0, w1, w2, w3, w4, w5

    wx0, wx1, wx2, wx3, wx4, wx5 = pqs_weights(dx)
    wy0, wy1, wy2, wy3, wy4, wy5 = pqs_weights(dy)

    wx = (wx0, wx1, wx2, wx3, wx4, wx5)
    wy = (wy0, wy1, wy2, wy3, wy4, wy5)

    ixs = (ix - 2, ix - 1, ix, ix + 1, ix + 2, ix + 3)
    iys = (iy - 2, iy - 1, iy, iy + 1, iy + 2, iy + 3)

    flat_size = Nx * Ny

    grid_fields = np.zeros((F, flat_size), dtype=np.float32)
    grid_weights = np.zeros(flat_size, dtype=np.float32)

    weighted_fields = particle_fields * particle_weights[:, None]

    for ox, oy in product(range(6), repeat=2):

        xi = ixs[ox]
        yi = iys[oy]
        w = wx[ox] * wy[oy]

        if periodic:
            xi = xi % Nx
            yi = yi % Ny
            valid = slice(None)
        else:
            valid = (0 <= xi) & (xi < Nx) & (0 <= yi) & (yi < Ny)

            if not np.any(valid):
                continue

            xi = xi[valid]
            yi = yi[valid]
            w = w[valid]

        flat_idx = xi * Ny + yi

        deposit_weight = particle_weights if periodic else particle_weights[valid]
        deposit_fields = weighted_fields if periodic else weighted_fields[valid]

        np.add.at(grid_weights, flat_idx, deposit_weight * w)

        for f in range(F):
            np.add.at(
                grid_fields[f],
                flat_idx,
                deposit_fields[:, f] * w,
            )

    return (
        grid_fields.reshape(F, Nx, Ny),
        grid_weights.reshape(Nx, Ny),
    )


def _pqs_3d(
    particle_positions,
    particle_fields,
    particle_weights,
    boxsizes,
    gridnums,
    periodic,
):
    particle_positions = _as_float32(particle_positions)
    particle_fields = _as_float32(particle_fields)
    particle_weights = _as_float32(particle_weights)
    boxsizes = _as_float32(boxsizes)

    Nx, Ny, Nz = gridnums
    F = particle_fields.shape[1]

    inv_dx = np.array([Nx, Ny, Nz], dtype=np.float32) / boxsizes

    # Cell-centered coordinates
    grid_pos = particle_positions * inv_dx - 0.5

    if periodic:
        grid_pos = np.mod(grid_pos, [Nx, Ny, Nz])

    ix = np.floor(grid_pos[:, 0]).astype(np.int32)
    iy = np.floor(grid_pos[:, 1]).astype(np.int32)
    iz = np.floor(grid_pos[:, 2]).astype(np.int32)

    dx = grid_pos[:, 0] - ix
    dy = grid_pos[:, 1] - iy
    dz = grid_pos[:, 2] - iz

    def pqs_weights(d):
        w0 = (1.0 - d) ** 5 / 120.0
        w1 = ((2.0 - d) ** 5 - 6.0 * (1.0 - d) ** 5) / 120.0
        w2 = ((3.0 - d) ** 5 - 6.0 * (2.0 - d) ** 5 + 15.0 * (1.0 - d) ** 5) / 120.0
        w3 = ((2.0 + d) ** 5 - 6.0 * (1.0 + d) ** 5 + 15.0 * d**5) / 120.0
        w4 = ((1.0 + d) ** 5 - 6.0 * d**5) / 120.0
        w5 = d**5 / 120.0
        return w0, w1, w2, w3, w4, w5

    wx0, wx1, wx2, wx3, wx4, wx5 = pqs_weights(dx)
    wy0, wy1, wy2, wy3, wy4, wy5 = pqs_weights(dy)
    wz0, wz1, wz2, wz3, wz4, wz5 = pqs_weights(dz)

    wx = (wx0, wx1, wx2, wx3, wx4, wx5)
    wy = (wy0, wy1, wy2, wy3, wy4, wy5)
    wz = (wz0, wz1, wz2, wz3, wz4, wz5)

    ixs = (ix - 2, ix - 1, ix, ix + 1, ix + 2, ix + 3)
    iys = (iy - 2, iy - 1, iy, iy + 1, iy + 2, iy + 3)
    izs = (iz - 2, iz - 1, iz, iz + 1, iz + 2, iz + 3)

    flat_size = Nx * Ny * Nz

    grid_fields = np.zeros((F, flat_size), dtype=np.float32)
    grid_weights = np.zeros(flat_size, dtype=np.float32)

    weighted_fields = particle_fields * particle_weights[:, None]

    for ox, oy, oz in product(range(6), repeat=3):

        xi = ixs[ox]
        yi = iys[oy]
        zi = izs[oz]
        w = wx[ox] * wy[oy] * wz[oz]

        if periodic:
            xi = xi % Nx
            yi = yi % Ny
            zi = zi % Nz
            valid = slice(None)
        else:
            valid = (
                (0 <= xi) & (xi < Nx) & (0 <= yi) & (yi < Ny) & (0 <= zi) & (zi < Nz)
            )

            if not np.any(valid):
                continue

            xi = xi[valid]
            yi = yi[valid]
            zi = zi[valid]
            w = w[valid]

        flat_idx = xi * Ny * Ny + yi * Nz + zi

        deposit_weight = particle_weights if periodic else particle_weights[valid]
        deposit_fields = weighted_fields if periodic else weighted_fields[valid]

        np.add.at(grid_weights, flat_idx, deposit_weight * w)

        for f in range(F):
            np.add.at(
                grid_fields[f],
                flat_idx,
                deposit_fields[:, f] * w,
            )

    return (
        grid_fields.reshape(F, Nx, Ny, Nz),
        grid_weights.reshape(Nx, Ny, Nz),
    )


# =============================================================================
# Python-level dispatch
# =============================================================================
FIXED_GRID_KERNELS = ["ngp", "cic", "tsc", "pcs", "pqs"]

_KERNELS = {
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
}


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
    if adaptive:
        raise ValueError(
            "backend='numpy' does not support adaptive deposition; "
            "use backend='taichi'."
        )
    if structure is not None:
        raise ValueError("structure is only valid when adaptive=True.")
    if kernel_name not in FIXED_GRID_KERNELS:
        raise ValueError(
            f"kernel_name={kernel_name!r} is not available on backend='numpy'. "
            f"Available: {FIXED_GRID_KERNELS}"
        )

    particle_positions = _as_float32(particle_positions)
    particle_fields = _as_float32(particle_fields)
    particle_weights = _as_float32(particle_weights)
    boxsizes = _as_float32(boxsizes)
    dim = particle_positions.shape[-1]

    key = (kernel_name, dim)
    if key not in _KERNELS:
        raise ValueError(
            f"Unsupported combination: kernel_name={kernel_name!r}, dim={dim}. "
            f"Available: {sorted(_KERNELS.keys())}"
        )
    kernel = _KERNELS[key]

    grid_fields, grid_weights = kernel(
        particle_positions,
        particle_fields,
        particle_weights,
        boxsizes,
        gridnums,
        bool(periodic),
    )

    return grid_fields, grid_weights


# ============================================================
if __name__ == "__main__":

    kernel_name = "pqs"
    dim = 1
    N = int(1e6)
    F = 3
    particle_positions = np.random.uniform(0, 1, (N, dim))
    particle_fields = np.ones((len(particle_positions), F))
    particle_weights = np.ones(len(particle_positions))
    boxsizes = np.array([1.0] * dim)
    gridnums = [512] * dim
    periodic = True

    from timeit import repeat

    N_REPEATS = 3
    N_LOOPS = 3

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------
    fields, weights = deposit(
        particle_positions,
        particle_fields,
        particle_weights,
        boxsizes,
        gridnums,
        periodic,
        kernel_name=kernel_name,
    )
    print("fields:", fields.shape)
    print("weights:", weights.shape)

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------
    best = min(
        repeat(
            stmt=lambda: deposit(
                particle_positions,
                particle_fields,
                particle_weights,
                boxsizes,
                gridnums,
                periodic,
                kernel_name=kernel_name,
            ),
            repeat=N_REPEATS,
            number=N_LOOPS,
        )
    )
    print(f"Best runtime: {best / N_LOOPS * 1e3:.3f} ms")
