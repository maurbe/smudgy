"""Particle-mode density / interpolation are normalized for every kernel.

Particle mode sums only the K nearest neighbors, so kernels with support
> 1 must be scaled such that their compact support ends at the K-th
neighbor (isotropic) or on the smoothing-tensor ellipsoid (covariant).
Without that scaling they were truncated (e.g. 'gaussian' recovered ~43% of
the density in 3D, ~87% in 1D).
"""

import numpy as np
import pytest

from smudgy import PointCloud
from smudgy.backend.numpy.kernels import get_kernel

DIMS = [1, 2, 3]
BACKENDS = ["numpy", "taichi"]
VARIANTS = [("isotropic", False), ("covariant", False), ("covariant", True)]
KERNEL_NAMES = [
    "tophat",
    "tsc",
    "gaussian",
    "lucy",
    "cubic_spline",
    "quintic_spline",
    "wendland_c2",
    "wendland_c4",
    "wendland_c6",
]
NUM_POINTS = 5000
NUM_NEIGHBORS = {1: 16, 2: 32, 3: 32}
# The leave-one-out K-NN density is unbiased only up to O(1/K) and the
# covariant hull volume adds a few percent, hence the loose density tolerance;
# the truncation it guards against reaches 57% (gaussian, 3D).
RTOL_DENSITY = 0.1
RTOL_POU = 0.05


@pytest.fixture(scope="module")
def positions():
    rng = np.random.default_rng(0)
    return {dim: rng.random((NUM_POINTS, dim)).astype(np.float32) for dim in DIMS}


def _point_cloud(pos, backend, kernel_name, structure, volume_preserving):
    dim = pos.shape[1]
    pc = PointCloud(
        positions=pos,
        weights=np.full(len(pos), 1.0 / len(pos), dtype=np.float32),
        boxsize=1.0,
        verbose=False,
        backend=backend,
    ).global_setup(num_neighbors=NUM_NEIGHBORS[dim], kernel_name=kernel_name)
    pc.compute_smoothing(structure=structure, volume_preserving=volume_preserving)
    pc.compute_density(structure=structure)
    return pc


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kernel_name", KERNEL_NAMES)
@pytest.mark.parametrize("structure,volume_preserving", VARIANTS)
@pytest.mark.parametrize("dim", DIMS)
def test_density_normalized(positions, backend, kernel_name, structure, volume_preserving, dim):
    """Mean leave-one-out density of uniform points recovers the true density 1."""
    pos = positions[dim]
    pc = _point_cloud(pos, backend, kernel_name, structure, volume_preserving)

    kernel = get_kernel(kernel_name, dim=dim)
    w0 = kernel._kernel_sigma() * float(kernel._kernel_values(np.array(0.0)))
    if structure == "covariant":
        rho = pc.smoothing.density_covariant
        vol = np.abs(np.linalg.det(pc.smoothing.smoothing_tensors.astype(np.float64)))
    else:
        rho = pc.smoothing.density_isotropic
        vol = pc.smoothing.smoothing_lengths.astype(np.float64) ** dim
    # self-contribution m W(0) of the support-scaled kernel (h -> h / support)
    scale = max(kernel.support, 1.0) ** dim
    rho_loo = rho - (1.0 / len(pos)) * w0 * scale / vol

    np.testing.assert_allclose(np.mean(rho_loo), 1.0, rtol=RTOL_DENSITY)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kernel_name", KERNEL_NAMES)
@pytest.mark.parametrize("structure,volume_preserving", VARIANTS)
@pytest.mark.parametrize("dim", DIMS)
def test_partition_of_unity(positions, backend, kernel_name, structure, volume_preserving, dim):
    """Interpolating a constant field at the particles returns that constant."""
    pos = positions[dim]
    pc = _point_cloud(pos, backend, kernel_name, structure, volume_preserving)
    ones = np.ones(len(pc.weights), dtype=np.float32)

    pou = pc.interpolate(ones, structure=structure)

    np.testing.assert_allclose(np.median(pou), 1.0, rtol=RTOL_POU)
