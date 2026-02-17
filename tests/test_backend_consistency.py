"""Tests cross-backend consistency between Python and C++ deposition."""

import pickle
from pathlib import Path

import numpy as np
import pytest

from sph_lib import PointCloud

DATASETS = ["random", "cosmo"]
datapath = "~/Desktop/sph_lib_analysis/data/"
GRIDNUM = 64


def _load_dataset(dataset: str, dim: int):
    """Load test dataset for backend consistency tests."""
    # Expand user path and select dataset based on parameters
    dataset_path = Path(datapath).expanduser()
    dataset_path /= f"dataset_{dataset}_{dim}d.pkl"
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")
    with dataset_path.open("rb") as f:
        data = pickle.load(f)
    return data


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("method", ["ngp", "cic", "tsc"])
@pytest.mark.parametrize("dataset", DATASETS)
def test_python_cpp_backend_consistency(dim, method, dataset):
    """Test consistency between Python and C++ backends for deposition."""
    data = _load_dataset(dataset, dim)

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
