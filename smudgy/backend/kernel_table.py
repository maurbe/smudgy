"""Kernel constants shared by all backends: support radius and normalisation.

One entry per kernel: `support` is the compact-support radius in units of h
(q = r / h), `sigma[d - 1]` the normalisation in d dimensions, such that
``sigma * W(q) / h**d`` integrates to one. The backends only implement the
shape W(q) (and its gradient / antiderivative) and read these constants.
"""

from math import pi, sqrt
from typing import NamedTuple


class KernelConstants(NamedTuple):
    """Support radius (units of h) and normalisation in 1, 2 and 3 dimensions."""

    support: float
    sigma: tuple[float, float, float]  # 1D, 2D, 3D


def _tsc_sigma(s):
    return (1.0 / s, 3.0 / (pi * s**2), 3.0 / (pi * s**3))


_GAUSSIAN_SIGMA = (1.0 / sqrt(pi), 1.0 / pi, 1.0 / pi**1.5)

# Radially symmetric kernels W(|xi|), used by isotropic and covariant smoothing.
SPHERICAL = {
    "tophat": KernelConstants(0.5, (1.0, 4.0 / pi, 6.0 / pi)),
    "tsc": KernelConstants(1.5, _tsc_sigma(1.5)),
    "gaussian": KernelConstants(3.0, _GAUSSIAN_SIGMA),
    "lucy": KernelConstants(1.0, (5.0 / 4.0, 5.0 / pi, 105.0 / (16.0 * pi))),
    "cubic_spline": KernelConstants(1.0, (4.0 / 3.0, 40.0 / (7.0 * pi), 8.0 / pi)),
    "quintic_spline": KernelConstants(
        3.0, (1.0 / 120.0, 7.0 / (478.0 * pi), 1.0 / (120.0 * pi))
    ),
    "wendland_c2": KernelConstants(
        2.0, (5.0 / 8.0, 7.0 / (4.0 * pi), 21.0 / (16.0 * pi))
    ),
    "wendland_c4": KernelConstants(
        2.0, (3.0 / 4.0, 9.0 / (4.0 * pi), 495.0 / (256.0 * pi))
    ),
    "wendland_c6": KernelConstants(
        2.0, (55.0 / 64.0, 39.0 / (14.0 * pi), 1365.0 / (512.0 * pi))
    ),
}

# Tensor-product kernels prod_a W_1d(q_a) (separable deposition only).
SEPARABLE = {
    "tophat": KernelConstants(0.5, (1.0, 1.0, 1.0)),
    "tsc": KernelConstants(1.5, (1.0 / 1.5, 1.0 / 1.5**2, 1.0 / 1.5**3)),
    "gaussian": KernelConstants(3.0, _GAUSSIAN_SIGMA),
}


def spherical_constants(name):
    """Constants of a spherical kernel, with a helpful error for unknown names."""
    try:
        return SPHERICAL[name]
    except KeyError:
        raise ValueError(
            f"Invalid kernel_name {name!r}. Must be one of {list(SPHERICAL)}."
        ) from None
