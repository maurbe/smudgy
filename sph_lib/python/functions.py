import numpy as np


def ngp_2d(positions, quantities, extent, gridnum, periodic):

    boxsize = extent[1] - extent[0]
    inv_dx = gridnum / boxsize
    grid_pos = ((positions - extent[0]) * inv_dx).astype(int)

    fields  = np.zeros((gridnum, gridnum, quantities.shape[1]), dtype=np.float32)
    weights = np.zeros((gridnum, gridnum), dtype=np.int32)
    
    # Split indices
    x_idx = grid_pos[:, 0]
    y_idx = grid_pos[:, 1]

    for f in range(quantities.shape[1]):
        np.add.at(fields[:, :, f], (x_idx, y_idx), quantities[:, f])
    np.add.at(weights, (x_idx, y_idx), 1)
    
    return fields, weights

def ngp_3d(positions, quantities, extent, gridnum, periodic):

    boxsize = extent[1] - extent[0]
    inv_dx = gridnum / boxsize
    grid_pos = ((positions - extent[0]) * inv_dx).astype(int)

    fields  = np.zeros((gridnum, gridnum, gridnum, quantities.shape[1]), dtype=np.float32)
    weights = np.zeros((gridnum, gridnum, gridnum), dtype=np.int32)

    # Split indices
    x_idx = grid_pos[:, 0]
    y_idx = grid_pos[:, 1]
    z_idx = grid_pos[:, 2]

    for f in range(quantities.shape[1]):
        np.add.at(fields[:, :, :, f], (x_idx, y_idx, z_idx), quantities[:, f])
    np.add.at(weights, (x_idx, y_idx, z_idx), 1)

    return fields, weights

def cic_2d(positions, quantities, extent, gridnum, periodic):
    boxsize = extent[1] - extent[0]
    inv_dx = gridnum / boxsize

    grid_pos = (positions - extent[0]) * inv_dx
    base_idx = np.floor(grid_pos).astype(int)
    frac = grid_pos - base_idx

    dx, dy = frac[:, 0], frac[:, 1]
    x0, y0 = base_idx[:, 0], base_idx[:, 1]
    x1 = x0 + 1
    y1 = y0 + 1

    if periodic:
        x0 %= gridnum
        x1 %= gridnum
        y0 %= gridnum
        y1 %= gridnum
    else:
        x1 = np.clip(x1, 0, gridnum - 1)
        y1 = np.clip(y1, 0, gridnum - 1)

    w00 = (1 - dx) * (1 - dy)
    w10 = dx * (1 - dy)
    w01 = (1 - dx) * dy
    w11 = dx * dy

    fields  = np.zeros((gridnum, gridnum, quantities.shape[1]), dtype=np.float32)
    weights = np.zeros((gridnum, gridnum), dtype=np.float32)

    for i in range(quantities.shape[1]):
        q = quantities[:, i]
        np.add.at(fields[:, :, i], (x0, y0), q * w00)
        np.add.at(fields[:, :, i], (x1, y0), q * w10)
        np.add.at(fields[:, :, i], (x0, y1), q * w01)
        np.add.at(fields[:, :, i], (x1, y1), q * w11)

    np.add.at(weights, (x0, y0), w00)
    np.add.at(weights, (x1, y0), w10)
    np.add.at(weights, (x0, y1), w01)
    np.add.at(weights, (x1, y1), w11)

    return fields, weights

def cic_3d(positions, quantities, extent, gridnum, periodic):
    boxsize = extent[1] - extent[0]
    inv_dx = gridnum / boxsize

    grid_pos = (positions - extent[0]) * inv_dx
    base_idx = np.floor(grid_pos).astype(int)
    frac = grid_pos - base_idx

    dx, dy, dz = frac[:, 0], frac[:, 1], frac[:, 2]
    x0, y0, z0 = base_idx[:, 0], base_idx[:, 1], base_idx[:, 2]
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    if periodic:
        x0 %= gridnum
        x1 %= gridnum
        y0 %= gridnum
        y1 %= gridnum
        z0 %= gridnum
        z1 %= gridnum
    else:
        #TODO: this is wrong to do... need to skip particles outside the box
        x1 = np.clip(x1, 0, gridnum - 1)
        y1 = np.clip(y1, 0, gridnum - 1)
        z1 = np.clip(z1, 0, gridnum - 1)

    # Compute weights for 8 surrounding grid points
    w000 = (1 - dx) * (1 - dy) * (1 - dz)
    w100 = dx * (1 - dy) * (1 - dz)
    w010 = (1 - dx) * dy * (1 - dz)
    w110 = dx * dy * (1 - dz)
    w001 = (1 - dx) * (1 - dy) * dz
    w101 = dx * (1 - dy) * dz
    w011 = (1 - dx) * dy * dz
    w111 = dx * dy * dz

    fields  = np.zeros((gridnum, gridnum, gridnum, quantities.shape[1]), dtype=np.float32)
    weights = np.zeros((gridnum, gridnum, gridnum), dtype=np.float32)

    for i in range(quantities.shape[1]):
        q = quantities[:, i]
        np.add.at(fields[:, :, :, i], (x0, y0, z0), q * w000)
        np.add.at(fields[:, :, :, i], (x1, y0, z0), q * w100)
        np.add.at(fields[:, :, :, i], (x0, y1, z0), q * w010)
        np.add.at(fields[:, :, :, i], (x1, y1, z0), q * w110)
        np.add.at(fields[:, :, :, i], (x0, y0, z1), q * w001)
        np.add.at(fields[:, :, :, i], (x1, y0, z1), q * w101)
        np.add.at(fields[:, :, :, i], (x0, y1, z1), q * w011)
        np.add.at(fields[:, :, :, i], (x1, y1, z1), q * w111)

    np.add.at(weights, (x0, y0, z0), w000)
    np.add.at(weights, (x1, y0, z0), w100)
    np.add.at(weights, (x0, y1, z0), w010)
    np.add.at(weights, (x1, y1, z0), w110)
    np.add.at(weights, (x0, y0, z1), w001)
    np.add.at(weights, (x1, y0, z1), w101)
    np.add.at(weights, (x0, y1, z1), w011)
    np.add.at(weights, (x1, y1, z1), w111)

    return fields, weights

def _weights_tsc(d):
        w = np.empty((len(d), 3))
        w[:, 0] = 0.5 * (1.5 - d)**2
        w[:, 1] = 0.75 - (d - 1)**2
        w[:, 2] = 0.5 * (d - 0.5)**2
        return w

def tsc_2d(positions, quantities, extent, gridnum, periodic):
    boxsize = extent[1] - extent[0]
    inv_dx = gridnum / boxsize

    # Convert positions to grid coordinates
    grid_pos = (positions - extent[0]) * inv_dx

    base_idx = np.floor(grid_pos).astype(int)  # (N, 2)
    frac = grid_pos - base_idx                 # (N, 2)
    nf = quantities.shape[1]

    fields = np.zeros((gridnum, gridnum, nf), dtype=np.float32)
    weights = np.zeros((gridnum, gridnum), dtype=np.float32)
    offsets = np.array([-1, 0, 1])

    # Precompute the TSC weight function for each axis and particle for neighbors
    wx = _weights_tsc(frac[:, 0])
    wy = _weights_tsc(frac[:, 1])

    for dx_i, dx in enumerate(offsets):
        for dy_i, dy in enumerate(offsets):
            ix = base_idx[:, 0] + dx
            iy = base_idx[:, 1] + dy

            if periodic:
                ix %= gridnum
                iy %= gridnum
            #else:
            #    ix = np.clip(ix, 0, gridnum - 1)
            #    iy = np.clip(iy, 0, gridnum - 1)

            w = wx[:, dx_i] * wy[:, dy_i]  # (N,)

            for f in range(nf):
                np.add.at(fields[:, :, f], (ix, iy), quantities[:, f] * w)

            np.add.at(weights, (ix, iy), w)

    return fields, weights

def tsc_3d(positions, quantities, extent, gridnum, periodic):
    boxsize = extent[1] - extent[0]
    inv_dx = gridnum / boxsize

    # Normalize positions to grid coordinates (float)
    grid_pos = (positions - extent[0]) * inv_dx

    # Integer cell indices (center cell)
    base_idx = np.floor(grid_pos).astype(int)  # shape (N,3)

    # Fractional offset inside cell
    frac = grid_pos - base_idx  # shape (N,3)
    nf = quantities.shape[1]

    fields = np.zeros((gridnum, gridnum, gridnum, nf), dtype=np.float32)
    weights = np.zeros((gridnum, gridnum, gridnum), dtype=np.float32)
    offsets = np.array([-1, 0, 1])

    # Precompute the TSC weight function for each axis and particle for neighbors
    wx = _weights_tsc(frac[:, 0])
    wy = _weights_tsc(frac[:, 1])
    wz = _weights_tsc(frac[:, 2])

    # Loop over each neighbor offset in 3D (27 neighbors)
    for dx_i, dx in enumerate(offsets):
        for dy_i, dy in enumerate(offsets):
            for dz_i, dz in enumerate(offsets):
                # Compute neighbor cell indices
                ix = base_idx[:, 0] + dx
                iy = base_idx[:, 1] + dy
                iz = base_idx[:, 2] + dz

                if periodic:
                    ix %= gridnum
                    iy %= gridnum
                    iz %= gridnum
                else:
                    # Clip to grid boundaries
                    ix = np.clip(ix, 0, gridnum - 1)
                    iy = np.clip(iy, 0, gridnum - 1)
                    iz = np.clip(iz, 0, gridnum - 1)

                # Compute weights for this neighbor
                w = wx[:, dx_i] * wy[:, dy_i] * wz[:, dz_i]  # shape (N,)

                # Deposit quantities weighted by w into fields
                for f in range(nf):
                    np.add.at(fields[:, :, :, f], (ix, iy, iz), quantities[:, f] * w)

                # Deposit weights (scalar)
                np.add.at(weights, (ix, iy, iz), w)

    return fields, weights
