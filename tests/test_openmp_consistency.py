"""Tests OpenMP consistency across enable/disable and thread counts."""

import ctypes
import pickle
from pathlib import Path

import numpy as np
import pytest

from sph_lib import PointCloud, check_openmp



DATASETS = ["random", "cosmo"]
METHODS = ["ngp", "cic", "tsc"]
GRIDNUM = 64

# --- Module-level OpenMP check ---
openmp_available = check_openmp()
pytestmark = pytest.mark.skipif(
    not openmp_available,
    reason="OpenMP not available; skipping OpenMP consistency tests."
)


def _load_dataset(dataset: str, dim: int):
    data_dir = Path(__file__).resolve().parents[2] / "data"
    dataset_path = data_dir / f"dataset_{dataset}_{dim}d.pkl"
    if not dataset_path.exists():
        pytest.skip(f"Dataset not found: {dataset_path}")
    with dataset_path.open("rb") as f:
        data = pickle.load(f)
    return data


def _get_openmp_max_threads():
    try:
        from sph_lib.core import _cpp_functions_ext as cppfunc
    except Exception:
        return None

    try:
        lib = ctypes.CDLL(cppfunc.__file__)
        omp_get_max_threads = lib.omp_get_max_threads
        omp_get_max_threads.restype = ctypes.c_int
        return int(omp_get_max_threads())
    except Exception:
        return None


def _require_openmp():
    max_threads = _get_openmp_max_threads()
    if not max_threads or max_threads < 2:
        pytest.skip("OpenMP not available or only one thread")
    return max_threads


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dataset", DATASETS)
def test_openmp_toggle_consistency(dim, method, dataset):
    max_threads = _require_openmp()

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

    fields_serial = sim.deposit_to_grid(
        fields=fields,
        averaged=[False] * fields.shape[1],
        gridnums=GRIDNUM,
        method=method,
        use_python=False,
        use_openmp=False,
    )

    fields_omp = sim.deposit_to_grid(
        fields=fields,
        averaged=[False] * fields.shape[1],
        gridnums=GRIDNUM,
        method=method,
        use_python=False,
        use_openmp=True,
        omp_threads=min(2, max_threads),
    )

    assert fields_serial.shape == fields_omp.shape
    np.testing.assert_allclose(fields_serial, fields_omp, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dataset", DATASETS)
def test_openmp_thread_counts_consistency(dim, method, dataset):
    max_threads = _require_openmp()
    thread_counts = [t for t in (2, 4, 8) if t <= max_threads]
    if len(thread_counts) < 2:
        pytest.skip("Not enough OpenMP threads available for comparison")

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

    fields_ref = sim.deposit_to_grid(
        fields=fields,
        averaged=[False] * fields.shape[1],
        gridnums=GRIDNUM,
        method=method,
        use_python=False,
        use_openmp=True,
        omp_threads=thread_counts[0],
    )

    for threads in thread_counts[1:]:
        fields_test = sim.deposit_to_grid(
            fields=fields,
            averaged=[False] * fields.shape[1],
            gridnums=GRIDNUM,
            method=method,
            use_python=False,
            use_openmp=True,
            omp_threads=threads,
        )
        assert fields_ref.shape == fields_test.shape
        np.testing.assert_allclose(fields_ref, fields_test, rtol=1e-4, atol=1e-6)
