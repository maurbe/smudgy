"""Backend-independent helpers shared by the numpy and Quadrants backends.

Input normalisation (dtype / contiguity / box size), shared constants and
argument validation live here once, so that the two backends differ only in
how they compute.
"""

import numpy as np

# Regulariser in q = r / (h + EPS) and in the covariant second-moment tensor.
EPS = 1e-7

FIXED_GRID_KERNELS = ("ngp", "cic", "tsc", "bspline4", "bspline5", "bspline6")
ADAPTIVE_STRUCTURES = ("separable", "isotropic", "covariant")
INTERPOLATION_MODES = ("field", "gradient", "divergence", "curl")

# Rows are the unit vectors e_x, e_y, e_z (plane projections).
BASIS_VECTORS = np.eye(3, dtype=np.float32)


# =============================================================================
# Input normalisation
# =============================================================================
def as_f32(array):
    """C-contiguous float32 array; no copy if `array` already is one."""
    return np.ascontiguousarray(array, dtype=np.float32)


def as_i32(array):
    """C-contiguous int32 array; no copy if `array` already is one."""
    return np.ascontiguousarray(array, dtype=np.int32)


def normalize_boxsize(boxsize, dim):
    """Return ``(box, periodic)`` for a scalar/(D,) box size or None.

    `box` is a float32 array of shape (dim,), all zeros when non-periodic
    (``boxsize is None``), so it can always be passed to a kernel.
    """
    if boxsize is None:
        return np.zeros(dim, dtype=np.float32), False
    return as_f32(np.broadcast_to(boxsize, (dim,))), True


def support_scaled(h, support):
    """Rescale h so a kernel's compact support does not extend past q = 1.

    In particle mode `h` is the distance to the K-th neighbour (isotropic) or
    the tensor whose ellipsoid passes through the farthest neighbour
    (covariant), and only those K neighbours are summed. A kernel whose
    support is `s` > 1 times h would be truncated there, so h -> h / s.
    Kernels with s <= 1 are left as they are. Never modifies `h` in place.
    """
    if support <= 1.0:
        return h
    return np.asarray(h, dtype=np.float32) / np.float32(support)


# =============================================================================
# Argument validation
# =============================================================================
def check_particle_structure(structure, operation):
    """Particle density/interpolation support isotropic and covariant h only."""
    if structure == "separable":
        raise ValueError(
            f"structure='separable' is not supported for particle {operation}; "
            "use 'isotropic' or 'covariant'."
        )
    if structure not in ("isotropic", "covariant"):
        raise ValueError(
            f"Invalid structure {structure!r} for particle {operation}. "
            "Expected 'isotropic' or 'covariant'."
        )


def check_interpolation_mode(mode, dim):
    """Validate an interpolation mode for the given dimension."""
    if mode not in INTERPOLATION_MODES:
        raise ValueError(
            f"Unsupported interpolation mode {mode!r}. "
            f"Expected one of {INTERPOLATION_MODES}."
        )
    if mode == "curl" and dim == 1:
        raise ValueError("mode='curl' is undefined in 1D.")


def check_hmat_args(dim, ratio_max, min_axis, volume_preserving, hsml, n_query):
    """Validate the smoothing-tensor options shared by both backends.

    Returns `min_axis` as a float, with 0.0 meaning "no lower bound".
    """
    if dim not in (1, 2, 3):
        raise ValueError(
            "[smudgy] Only 1D, 2D and 3D positions are supported for "
            "anisotropic smoothing tensors."
        )
    if not ratio_max >= 1.0:
        raise ValueError(f"[smudgy] ratio_max must be >= 1, got {ratio_max}.")
    if min_axis is not None and not min_axis >= 0.0:
        raise ValueError(f"[smudgy] min_axis must be >= 0, got {min_axis}.")
    if volume_preserving:
        if hsml is None:
            raise ValueError("[smudgy] volume_preserving=True requires hsml.")
        if np.shape(hsml) != (n_query,):
            raise ValueError(
                f"[smudgy] hsml must have shape ({n_query},), got {np.shape(hsml)}."
            )
    return 0.0 if min_axis is None else float(min_axis)


def check_fixed_grid_kernel(kernel_name, structure):
    """Validate a fixed-grid (``adaptive=False``) deposition request."""
    if structure is not None:
        raise ValueError("structure is only valid when adaptive=True.")
    if kernel_name not in FIXED_GRID_KERNELS:
        raise ValueError(
            f"kernel_name={kernel_name!r} is not a fixed-grid stencil. "
            f"Available: {list(FIXED_GRID_KERNELS)}"
        )
