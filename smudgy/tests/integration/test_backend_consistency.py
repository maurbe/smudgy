"""Tests cross-backend consistency between Python and C++ deposition."""

import numpy as np
import pytest

from smudgy import PointCloud

GRIDNUM = 64
DATASETS = ["random"]
STRUCTURES = ["separable"]
KERNEL_NAMES = ["ngp", "tophat"]


def _generate_dataset(dim: int):
    """Generate a random dataset for testing."""
    np.random.seed(42)
    N = 1000
    positions = np.random.uniform(0, 1, size=(N, dim))
    weights = np.ones(N, dtype=np.float32)
    boxsize = np.ones(dim, dtype=np.float32)
    return {"positions": positions, "weights": weights, "boxsize": boxsize}


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("structure", STRUCTURES)
@pytest.mark.parametrize("kernel_name", KERNEL_NAMES)
def test_python_cpp_backend_consistency(dim, structure, kernel_name):
    """Test consistency between Python and C++ backends for deposition."""
    data = _generate_dataset(dim)

    positions = np.asarray(data["positions"], dtype=np.float32)
    weights = np.asarray(data["weights"], dtype=np.float32)
    boxsize = np.asarray(data["boxsize"], dtype=np.float32)
    fields = weights

    sim = PointCloud(
        positions=positions,
        weights=weights,
        boxsize=boxsize,
        verbose=False,
    )

    fields_py = sim.deposit_to_grid(
        fields=fields,
        averaged=[False] * len(fields),
        gridnums=GRIDNUM,
        kernel_name=kernel_name,
        structure=structure,
        adaptive=False,
        use_python=True,
    )

    fields_cpp = sim.deposit_to_grid(
        fields=fields,
        averaged=[False] * len(fields),
        gridnums=GRIDNUM,
        kernel_name=kernel_name,
        structure=structure,
        adaptive=False,
        use_python=False,
        use_openmp=False,
    )

    assert fields_py.shape == fields_cpp.shape
    np.testing.assert_allclose(fields_py, fields_cpp, rtol=1e-4, atol=1e-6)
