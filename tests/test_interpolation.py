import numpy as np
import pytest
from sph_lib import kernels, utils

@pytest.mark.parametrize("pbc", [False, True])
@pytest.mark.parametrize("mode", ["isotropic", "anisotropic"])
@pytest.mark.parametrize("quantity", ["field", "gradient"])
def test_interpolation_modes(pbc, mode, quantity):
    np.random.seed(42)
    N = 100
    D = 3
    positions = np.random.uniform(0, 1, size=(N, D))
    values = np.random.uniform(-1, 1, size=N)
    boxsize = np.ones(D) if pbc else None
    masses = np.ones(N)

    # Create PointCloud and compute smoothing lengths
    from sph_lib.ops import PointCloud
    pc = PointCloud(positions, masses, boxsize=boxsize, verbose=False)
    pc.compute_smoothing_lengths(num_neighbors=32, mode=mode)
    kernel_name = "cubic_spline"
    
    # Interpolation
    if quantity == "field":
        result = pc.interpolate_fields(values, positions, kernel_name=kernel_name, compute_gradients=False)
        assert result.shape[0] == N
    else:  # gradient
        result = pc.interpolate_fields(values, positions, kernel_name=kernel_name, compute_gradients=True)
        assert result.shape[0] == N
        assert result.shape[-1] == D
