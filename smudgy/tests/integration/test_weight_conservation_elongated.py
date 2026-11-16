"""Mass conservation for strongly elongated covariant kernels.

A rotated, needle-like smoothing tensor can span several cells along every
grid axis (so its eta exceeds eta_crit and the cell-quadrature path is
chosen) while being much thinner than a cell across its minor axis. No
quadrature node may then fall inside its support; such particles must fall
back to the (normalized) sample-table deposition instead of losing their
mass.
"""

import numpy as np
import pytest

from smudgy import PointCloud

GRIDNUM = 32


def _rotation(dim: int, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    if dim == 2:
        return np.array([[c, -s], [s, c]])
    # rotate about z, then about x, so the long axis is oblique to all axes
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return rx @ rz


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("kernel_name", ["gaussian", "quintic_spline", "cubic_spline"])
@pytest.mark.parametrize("integration_method", ["midpoint", "trapezoidal", "simpson"])
def test_needle_kernels_conserve_weight(dim, kernel_name, integration_method):
    rng = np.random.default_rng(42)
    n = 64
    positions = rng.uniform(0.2, 0.8, size=(n, dim))
    weights = rng.uniform(0.5, 2.0, size=n)

    pc = PointCloud(positions, weights, boxsize=1.0, verbose=False, arch="cpu")

    # long axis ~5 cells, minor axis(es) ~1/50 cell (axis ratio 250)
    cell = 1.0 / GRIDNUM
    eigvals = np.full((n, dim), 0.02 * cell)
    eigvals[:, 0] = 5.0 * cell
    eigvecs = np.empty((n, dim, dim))
    for i, angle in enumerate(rng.uniform(0.2, 1.3, size=n)):
        eigvecs[i] = _rotation(dim, angle)
    tensors = np.einsum("nij,nj,nkj->nik", eigvecs, eigvals, eigvecs)

    pc.set_smoothing(
        structure="covariant",
        smoothing_tensors=tensors,
        smoothing_tensors_eigvals=eigvals,
        smoothing_tensors_eigvecs=eigvecs,
    )
    grid = pc.deposit(
        fields=None,
        averaged=False,
        gridnums=GRIDNUM,
        kernel_name=kernel_name,
        structure="covariant",
        integration_method=integration_method,
    )
    # rel=1e-3 tolerates the Gaussian truncation at q=3 (~4e-4 in 3D) in
    # the sample-table path; a single dropped particle would be ~1.5%.
    assert grid.sum() == pytest.approx(weights.sum(), rel=1e-3)
