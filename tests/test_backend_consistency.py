"""Tests cross-backend consistency between Python and C++ deposition."""

import numpy as np
import pytest

from sph_lib import PointCloud

GRIDNUM = 64
DATASETS = ["random"]


def _generate_dataset(dim: int):
    """Generate a random dataset for testing."""
    np.random.seed(42)
    N = 1000
    positions = np.random.uniform(0, 1, size=(N, dim))
    masses = np.ones(N, dtype=np.float32)
    boxsize = np.ones(dim, dtype=np.float32)
    return {"pos": positions, "mass": masses, "boxsize": boxsize}


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("method", ["ngp", "cic", "tsc"])
@pytest.mark.parametrize("dataset", DATASETS)
def test_python_cpp_backend_consistency(dim, method, dataset):
    """Test consistency between Python and C++ backends for deposition."""
    data = _generate_dataset(dim)

    positions = np.asarray(data["pos"], dtype=np.float32)
    masses = np.asarray(data["mass"], dtype=np.float32)
    boxsize = np.asarray(data["boxsize"], dtype=np.float32)
    fields = masses[:, np.newaxis]

    sim = PointCloud(
        positions=positions,
        masses=masses,
        boxsize=boxsize,
        verbose=False,
    )

    fields_py = sim.deposit_to_grid(
        fields=fields,
        averaged=[False] * fields.shape[1],
        gridnums=GRIDNUM,
        method=method,
        use_python=True,
    )

    fields_cpp = sim.deposit_to_grid(
        fields=fields,
        averaged=[False] * fields.shape[1],
        gridnums=GRIDNUM,
        method=method,
        use_python=False,
        use_openmp=False,
    )

    assert fields_py.shape == fields_cpp.shape
    np.testing.assert_allclose(fields_py, fields_cpp, rtol=1e-4, atol=1e-6)
