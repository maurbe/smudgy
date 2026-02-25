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


def _cic_2d(positions, quantities, boxsizes, gridnums, periodic):
    """Deposit 2D particle quantities using bilinear Cloud-In-Cell weights.

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
        Sum of interpolation weights per cell (for later normalization).

    """
    positions = _as_float32(positions)
    quantities = _as_float32(quantities)
    boxsizes = _as_float32(boxsizes)

    gridnum_x, gridnum_y = gridnums
    inv_dx = np.array([gridnum_x, gridnum_y], dtype=np.float32) / boxsizes

    grid_pos = positions * inv_dx
    base_idx = np.floor(grid_pos).astype(int)
    frac = grid_pos - base_idx

    dx, dy = frac[:, 0], frac[:, 1]
    x0, y0 = base_idx[:, 0], base_idx[:, 1]
    x1 = x0 + 1
    y1 = y0 + 1

    x0_idx, x0_valid = _wrap_or_mask_indices(x0, gridnum_x, periodic)
    x1_idx, x1_valid = _wrap_or_mask_indices(x1, gridnum_x, periodic)
    y0_idx, y0_valid = _wrap_or_mask_indices(y0, gridnum_y, periodic)
    y1_idx, y1_valid = _wrap_or_mask_indices(y1, gridnum_y, periodic)

    w00 = (1 - dx) * (1 - dy)
    w10 = dx * (1 - dy)
    w01 = (1 - dx) * dy
    w11 = dx * dy

    fields = np.zeros((gridnum_x, gridnum_y, quantities.shape[1]), dtype=np.float32)
    weights = np.zeros((gridnum_x, gridnum_y), dtype=np.float32)

    neighbors = [
        (x0_idx, x0_valid, y0_idx, y0_valid, w00),
        (x1_idx, x1_valid, y0_idx, y0_valid, w10),
        (x0_idx, x0_valid, y1_idx, y1_valid, w01),
        (x1_idx, x1_valid, y1_idx, y1_valid, w11),
    ]

    for x_idx, x_mask, y_idx, y_mask, weight_vals in neighbors:
        valid = x_mask & y_mask
        if not np.any(valid):
            continue
        for i in range(quantities.shape[1]):
            q = quantities[:, i]
            np.add.at(
                fields[:, :, i],
                (x_idx[valid], y_idx[valid]),
                q[valid] * weight_vals[valid],
            )
        np.add.at(weights, (x_idx[valid], y_idx[valid]), weight_vals[valid])

    return fields, weights


def _cic_3d(positions, quantities, boxsizes, gridnums, periodic):
    """Deposit 3D particle quantities using trilinear Cloud-In-Cell weights.

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
        Sum of interpolation weights per cell (for later normalization).

    """
    positions = _as_float32(positions)
    quantities = _as_float32(quantities)
    boxsizes = _as_float32(boxsizes)

    gridnum_x, gridnum_y, gridnum_z = gridnums
    inv_dx = np.array([gridnum_x, gridnum_y, gridnum_z], dtype=np.float32) / boxsizes

    grid_pos = positions * inv_dx
    base_idx = np.floor(grid_pos).astype(int)
    frac = grid_pos - base_idx

    dx, dy, dz = frac[:, 0], frac[:, 1], frac[:, 2]
    x0, y0, z0 = base_idx[:, 0], base_idx[:, 1], base_idx[:, 2]
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0_idx, x0_valid = _wrap_or_mask_indices(x0, gridnum_x, periodic)
    x1_idx, x1_valid = _wrap_or_mask_indices(x1, gridnum_x, periodic)
    y0_idx, y0_valid = _wrap_or_mask_indices(y0, gridnum_y, periodic)
    y1_idx, y1_valid = _wrap_or_mask_indices(y1, gridnum_y, periodic)
    z0_idx, z0_valid = _wrap_or_mask_indices(z0, gridnum_z, periodic)
    z1_idx, z1_valid = _wrap_or_mask_indices(z1, gridnum_z, periodic)

    # Compute weights for 8 surrounding grid points
    w000 = (1 - dx) * (1 - dy) * (1 - dz)
    w100 = dx * (1 - dy) * (1 - dz)
    w010 = (1 - dx) * dy * (1 - dz)
    w110 = dx * dy * (1 - dz)
    w001 = (1 - dx) * (1 - dy) * dz
    w101 = dx * (1 - dy) * dz
    w011 = (1 - dx) * dy * dz
    w111 = dx * dy * dz

    fields = np.zeros(
        (gridnum_x, gridnum_y, gridnum_z, quantities.shape[1]), dtype=np.float32
    )
    weights = np.zeros((gridnum_x, gridnum_y, gridnum_z), dtype=np.float32)

    neighbors = [
        (x0_idx, x0_valid, y0_idx, y0_valid, z0_idx, z0_valid, w000),
        (x1_idx, x1_valid, y0_idx, y0_valid, z0_idx, z0_valid, w100),
        (x0_idx, x0_valid, y1_idx, y1_valid, z0_idx, z0_valid, w010),
        (x1_idx, x1_valid, y1_idx, y1_valid, z0_idx, z0_valid, w110),
        (x0_idx, x0_valid, y0_idx, y0_valid, z1_idx, z1_valid, w001),
        (x1_idx, x1_valid, y0_idx, y0_valid, z1_idx, z1_valid, w101),
        (x0_idx, x0_valid, y1_idx, y1_valid, z1_idx, z1_valid, w011),
        (x1_idx, x1_valid, y1_idx, y1_valid, z1_idx, z1_valid, w111),
    ]

    for x_idx, x_mask, y_idx, y_mask, z_idx, z_mask, weight_vals in neighbors:
        valid = x_mask & y_mask & z_mask
        if not np.any(valid):
            continue
        for i in range(quantities.shape[1]):
            q = quantities[:, i]
            np.add.at(
                fields[:, :, :, i],
                (x_idx[valid], y_idx[valid], z_idx[valid]),
                q[valid] * weight_vals[valid],
            )
        np.add.at(
            weights, (x_idx[valid], y_idx[valid], z_idx[valid]), weight_vals[valid]
        )

    return fields, weights


def _weights_tsc(d):
    """Return Triangular Shaped Cloud weights for displacement array ``d``.

    Parameters
    ----------
    d
        Displacements in cell units.

    Returns
    -------
    numpy.ndarray
        Weights of shape ``(len(d), 3)`` for offsets ``[-1, 0, 1]``.

    """
    w = np.empty((len(d), 3), dtype=np.float32)
    w[:, 0] = 0.5 * (1.5 - d) ** 2
    w[:, 1] = 0.75 - (d - 1) ** 2
    w[:, 2] = 0.5 * (d - 0.5) ** 2
    return w


def _tsc_2d(positions, quantities, boxsizes, gridnums, periodic):
    """Deposit 2D particle quantities using the Triangular Shaped Cloud kernel.

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
        Sum of interpolation weights per cell (for later normalization).

    """
    positions = _as_float32(positions)
    quantities = _as_float32(quantities)
    boxsizes = _as_float32(boxsizes)

    gridnum_x, gridnum_y = gridnums
    inv_dx = np.array([gridnum_x, gridnum_y], dtype=np.float32) / boxsizes

    # Convert positions to grid coordinates
    grid_pos = positions * inv_dx

    base_idx = np.floor(grid_pos).astype(int)  # (N, 2)
    frac = grid_pos - base_idx  # (N, 2)
    nf = quantities.shape[1]

    fields = np.zeros((gridnum_x, gridnum_y, nf), dtype=np.float32)
    weights = np.zeros((gridnum_x, gridnum_y), dtype=np.float32)
    offsets = np.array([-1, 0, 1])

    # Precompute the TSC weight function for each axis and particle for neighbors
    wx = _weights_tsc(frac[:, 0])
    wy = _weights_tsc(frac[:, 1])

    for dx_i, dx in enumerate(offsets):
        for dy_i, dy in enumerate(offsets):
            ix = base_idx[:, 0] + dx
            iy = base_idx[:, 1] + dy

            valid = np.ones(ix.shape[0], dtype=bool)
            if periodic:
                ix = np.mod(ix, gridnum_x)
                iy = np.mod(iy, gridnum_y)
            else:
                valid &= (ix >= 0) & (ix < gridnum_x)
                valid &= (iy >= 0) & (iy < gridnum_y)

            if not np.any(valid):
                continue

            w = wx[:, dx_i] * wy[:, dy_i]

            for f in range(nf):
                np.add.at(
                    fields[:, :, f],
                    (ix[valid], iy[valid]),
                    quantities[:, f][valid] * w[valid],
                )

            np.add.at(weights, (ix[valid], iy[valid]), w[valid])

    return fields, weights


def _tsc_3d(positions, quantities, boxsizes, gridnums, periodic):
    """Deposit 3D particle quantities using the Triangular Shaped Cloud kernel.

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
        Sum of interpolation weights per cell (for later normalization).

    """
    positions = _as_float32(positions)
    quantities = _as_float32(quantities)
    boxsizes = _as_float32(boxsizes)

    gridnum_x, gridnum_y, gridnum_z = gridnums
    inv_dx = np.array([gridnum_x, gridnum_y, gridnum_z], dtype=np.float32) / boxsizes

    # Normalize positions to grid coordinates (float)
    grid_pos = positions * inv_dx

    # Integer cell indices (center cell)
    base_idx = np.floor(grid_pos).astype(int)  # shape (N,3)

    # Fractional offset inside cell
    frac = grid_pos - base_idx  # shape (N,3)
    nf = quantities.shape[1]

    fields = np.zeros((gridnum_x, gridnum_y, gridnum_z, nf), dtype=np.float32)
    weights = np.zeros((gridnum_x, gridnum_y, gridnum_z), dtype=np.float32)
    offsets = np.array([-1, 0, 1])

    # Precompute the TSC weight function for each axis and particle for neighbors
    wx = _weights_tsc(frac[:, 0])
    wy = _weights_tsc(frac[:, 1])
    wz = _weights_tsc(frac[:, 2])

    # Loop over each neighbor offset in 3D (27 neighbors)

    for dx_i, dx in enumerate(offsets):
        for dy_i, dy in enumerate(offsets):
            for dz_i, dz in enumerate(offsets):
                ix = base_idx[:, 0] + dx
                iy = base_idx[:, 1] + dy
                iz = base_idx[:, 2] + dz

                valid = np.ones(ix.shape[0], dtype=bool)
                if periodic:
                    ix = np.mod(ix, gridnum_x)
                    iy = np.mod(iy, gridnum_y)
                    iz = np.mod(iz, gridnum_z)
                else:
                    valid &= (ix >= 0) & (ix < gridnum_x)
                    valid &= (iy >= 0) & (iy < gridnum_y)
                    valid &= (iz >= 0) & (iz < gridnum_z)

                if not np.any(valid):
                    continue

                w = wx[:, dx_i] * wy[:, dy_i] * wz[:, dz_i]

                for f in range(nf):
                    np.add.at(
                        fields[:, :, :, f],
                        (ix[valid], iy[valid], iz[valid]),
                        quantities[:, f][valid] * w[valid],
                    )

                np.add.at(weights, (ix[valid], iy[valid], iz[valid]), w[valid])

    return fields, weights
