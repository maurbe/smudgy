import numpy as np


def _wrap_or_mask_indices(indices: np.ndarray, gridnum: int, periodic_flag: bool):
    """Wrap indices for periodic axes or mark validity for non-periodic axes.

    Parameters
    ----------
    indices
        Index array to wrap or validate.
    gridnum
        Number of grid cells along the axis.
    periodic_flag
        Whether periodic wrapping is enabled for the axis.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        ``(indices, valid_mask)`` where ``indices`` may be wrapped and
        ``valid_mask`` is ``True`` for in-domain entries.

    """
    if periodic_flag:
        return np.mod(indices, gridnum), np.ones_like(indices, dtype=bool)
    valid = (indices >= 0) & (indices < gridnum)
    return indices, valid


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


def _ngp_2d(positions, quantities, boxsizes, gridnums, periodic):
    """Deposit particle quantities onto a 2D grid with Nearest-Grid-Point weighting.

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Cartesian particle coordinates, where ``N`` is the number of particles.
    quantities : ndarray, shape (N, F)
        Per-particle fields to accumulate, with ``F`` fields per particle.
    boxsizes : array_like of length 2
        Domain sizes per axis, assuming ``[0, boxsize]`` in each dimension.
    gridnums : array_like of length 2
        Number of grid cells for each axis.
    periodic : bool
        Whether to wrap particles that leave the domain (applies to all axes).

    Returns
    -------
    fields : ndarray
        Accumulated field values on the grid.
    weights : ndarray
        Particle counts per cell (used as normalization weights).

    """
    positions = _as_float32(positions)
    quantities = _as_float32(quantities)
    boxsizes = _as_float32(boxsizes)

    gridnum_x, gridnum_y = gridnums
    inv_dx = np.array([gridnum_x, gridnum_y], dtype=np.float32) / boxsizes
    grid_pos = (positions * inv_dx).astype(int)

    fields = np.zeros((gridnum_x, gridnum_y, quantities.shape[1]), dtype=np.float32)
    weights = np.zeros((gridnum_x, gridnum_y), dtype=np.float32)

    # Split indices
    x_idx = grid_pos[:, 0]
    y_idx = grid_pos[:, 1]

    valid = np.ones(x_idx.shape[0], dtype=bool)
    valid &= (x_idx >= 0) & (x_idx < gridnum_x)
    valid &= (y_idx >= 0) & (y_idx < gridnum_y)

    if not np.all(valid):
        x_idx = x_idx[valid]
        y_idx = y_idx[valid]
        quantities = quantities[valid]
    if x_idx.size == 0:
        return fields, weights

    for f in range(quantities.shape[1]):
        np.add.at(fields[:, :, f], (x_idx, y_idx), quantities[:, f])
    np.add.at(weights, (x_idx, y_idx), 1)
    return fields, weights


def _ngp_3d(positions, quantities, boxsizes, gridnums, periodic):
    """Deposit particle quantities onto a 3D grid via Nearest-Grid-Point weighting.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Cartesian particle coordinates, where ``N`` is the number of particles.
    quantities : ndarray, shape (N, F)
        Per-particle fields to accumulate, with ``F`` fields per particle.
    boxsizes : array_like of length 3
        Domain sizes per axis, assuming ``[0, boxsize]`` in each dimension.
    gridnums : array_like of length 3
        Number of grid cells for each axis.
    periodic : bool
        Whether to wrap particles that leave the domain (applies to all axes).

    Returns
    -------
    fields : ndarray
        Accumulated field values on the grid.
    weights : ndarray
        Particle counts per cell (used as normalization weights).

    """
    positions = _as_float32(positions)
    quantities = _as_float32(quantities)
    boxsizes = _as_float32(boxsizes)

    gridnum_x, gridnum_y, gridnum_z = gridnums
    inv_dx = np.array([gridnum_x, gridnum_y, gridnum_z], dtype=np.float32) / boxsizes
    grid_pos = (positions * inv_dx).astype(int)

    fields = np.zeros(
        (gridnum_x, gridnum_y, gridnum_z, quantities.shape[1]), dtype=np.float32
    )
    weights = np.zeros((gridnum_x, gridnum_y, gridnum_z), dtype=np.float32)

    # Split indices
    x_idx = grid_pos[:, 0]
    y_idx = grid_pos[:, 1]
    z_idx = grid_pos[:, 2]

    valid = np.ones(x_idx.shape[0], dtype=bool)

    valid &= (x_idx >= 0) & (x_idx < gridnum_x)
    valid &= (y_idx >= 0) & (y_idx < gridnum_y)
    valid &= (z_idx >= 0) & (z_idx < gridnum_z)

    if not np.all(valid):
        x_idx = x_idx[valid]
        y_idx = y_idx[valid]
        z_idx = z_idx[valid]
        quantities = quantities[valid]
    if x_idx.size == 0:
        return fields, weights

    for f in range(quantities.shape[1]):
        np.add.at(fields[:, :, :, f], (x_idx, y_idx, z_idx), quantities[:, f])
    np.add.at(weights, (x_idx, y_idx, z_idx), 1)
    return fields, weights


def _tophat_2d(positions, quantities, boxsizes, gridnums, periodic):
    """2D Cloud-In-Cell (cell-centered grid convention)."""

    positions = _as_float32(positions)
    quantities = _as_float32(quantities)
    boxsizes = _as_float32(boxsizes)

    Nx, Ny = gridnums
    inv_dx = np.array([Nx, Ny], dtype=np.float32) / boxsizes

    # --------------------------------------------------
    # Map to grid coordinates (CELL-CENTERED!)
    # --------------------------------------------------
    grid_pos = positions * inv_dx - 0.5

    if periodic:
        grid_pos = np.mod(grid_pos, [Nx, Ny])
    else:
        eps = 1e-6
        grid_pos = np.clip(grid_pos, 0, [Nx - 1 - eps, Ny - 1 - eps])

    base = np.floor(grid_pos).astype(np.int32)
    frac = grid_pos - base

    x0, y0 = base[:, 0], base[:, 1]
    x1, y1 = x0 + 1, y0 + 1

    dx, dy = frac[:, 0], frac[:, 1]

    # --------------------------------------------------
    # Handle indices
    # --------------------------------------------------
    x0_idx, x0_valid = _wrap_or_mask_indices(x0, Nx, periodic)
    x1_idx, x1_valid = _wrap_or_mask_indices(x1, Nx, periodic)
    y0_idx, y0_valid = _wrap_or_mask_indices(y0, Ny, periodic)
    y1_idx, y1_valid = _wrap_or_mask_indices(y1, Ny, periodic)

    # --------------------------------------------------
    # Weights (bilinear)
    # --------------------------------------------------
    wx0 = 1.0 - dx
    wx1 = dx
    wy0 = 1.0 - dy
    wy1 = dy

    fields = np.zeros((Nx, Ny, quantities.shape[1]), dtype=np.float32)
    weights = np.zeros((Nx, Ny), dtype=np.float32)

    neighbors = [
        (x0_idx, x0_valid, y0_idx, y0_valid, wx0 * wy0),
        (x1_idx, x1_valid, y0_idx, y0_valid, wx1 * wy0),
        (x0_idx, x0_valid, y1_idx, y1_valid, wx0 * wy1),
        (x1_idx, x1_valid, y1_idx, y1_valid, wx1 * wy1),
    ]

    # --------------------------------------------------
    # Accumulate
    # --------------------------------------------------
    for x_idx, x_mask, y_idx, y_mask, w in neighbors:
        valid = x_mask & y_mask
        if not np.any(valid):
            continue

        xi = x_idx[valid]
        yi = y_idx[valid]
        wv = w[valid]

        np.add.at(fields, (xi, yi), quantities[valid] * wv[:, None])
        np.add.at(weights, (xi, yi), wv)

    return fields, weights


def _tophat_3d(positions, quantities, boxsizes, gridnums, periodic):
    positions = _as_float32(positions)
    quantities = _as_float32(quantities)
    boxsizes = _as_float32(boxsizes)

    Nx, Ny, Nz = gridnums
    inv_dx = np.array([Nx, Ny, Nz], dtype=np.float32) / boxsizes

    # Normalize to grid space
    grid_pos = positions * inv_dx - 0.5

    if periodic:
        grid_pos = np.mod(grid_pos, [Nx, Ny, Nz])
    else:
        # Clamp to valid range
        eps = 1e-6
        grid_pos = np.clip(grid_pos, 0, [Nx - 1 - eps, Ny - 1 - eps, Nz - 1 - eps])

    base = np.floor(grid_pos).astype(np.int32)
    frac = grid_pos - base

    x0, y0, z0 = base[:, 0], base[:, 1], base[:, 2]
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    dx, dy, dz = frac[:, 0], frac[:, 1], frac[:, 2]

    # Wrap or mask
    x0_idx, x0_valid = _wrap_or_mask_indices(x0, Nx, periodic)
    x1_idx, x1_valid = _wrap_or_mask_indices(x1, Nx, periodic)
    y0_idx, y0_valid = _wrap_or_mask_indices(y0, Ny, periodic)
    y1_idx, y1_valid = _wrap_or_mask_indices(y1, Ny, periodic)
    z0_idx, z0_valid = _wrap_or_mask_indices(z0, Nz, periodic)
    z1_idx, z1_valid = _wrap_or_mask_indices(z1, Nz, periodic)

    # Weights
    wx0 = 1 - dx
    wx1 = dx
    wy0 = 1 - dy
    wy1 = dy
    wz0 = 1 - dz
    wz1 = dz

    fields = np.zeros((Nx, Ny, Nz, quantities.shape[1]), dtype=np.float32)
    weights = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    neighbors = [
        (x0_idx, x0_valid, y0_idx, y0_valid, z0_idx, z0_valid, wx0 * wy0 * wz0),
        (x1_idx, x1_valid, y0_idx, y0_valid, z0_idx, z0_valid, wx1 * wy0 * wz0),
        (x0_idx, x0_valid, y1_idx, y1_valid, z0_idx, z0_valid, wx0 * wy1 * wz0),
        (x1_idx, x1_valid, y1_idx, y1_valid, z0_idx, z0_valid, wx1 * wy1 * wz0),
        (x0_idx, x0_valid, y0_idx, y0_valid, z1_idx, z1_valid, wx0 * wy0 * wz1),
        (x1_idx, x1_valid, y0_idx, y0_valid, z1_idx, z1_valid, wx1 * wy0 * wz1),
        (x0_idx, x0_valid, y1_idx, y1_valid, z1_idx, z1_valid, wx0 * wy1 * wz1),
        (x1_idx, x1_valid, y1_idx, y1_valid, z1_idx, z1_valid, wx1 * wy1 * wz1),
    ]

    for x_idx, x_mask, y_idx, y_mask, z_idx, z_mask, w in neighbors:
        valid = x_mask & y_mask & z_mask
        if not np.any(valid):
            continue

        xi = x_idx[valid]
        yi = y_idx[valid]
        zi = z_idx[valid]
        wv = w[valid]

        # Vectorized accumulation over all fields
        np.add.at(fields, (xi, yi, zi), quantities[valid] * wv[:, None])
        np.add.at(weights, (xi, yi, zi), wv)

    return fields, weights

