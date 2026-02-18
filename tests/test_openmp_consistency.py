"""Tests OpenMP consistency across enable/disable and thread counts."""

import ctypes

import numpy as np
import pytest

from sph_lib import PointCloud, check_openmp

DATASETS = ["random"]
METHODS = ["ngp", "cic", "tsc"]
GRIDNUM = 64


# --- Module-level OpenMP check ---
openmp_available = check_openmp()
pytestmark = pytest.mark.skipif(
    not openmp_available,
    reason="OpenMP not available; skipping OpenMP consistency tests.",
)


def _generate_dataset(dim: int):
    """Generate a random dataset for testing."""
    np.random.seed(42)
    N = 1000
    positions = np.random.uniform(0, 1, size=(N, dim))
    masses = np.ones(N, dtype=np.float32)
    boxsize = np.ones(dim, dtype=np.float32)
    return {"pos": positions, "mass": masses, "boxsize": boxsize}


def _get_openmp_max_threads():
    """Get the maximum number of OpenMP threads available."""
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
    """Require at least two OpenMP threads for test, else skip."""
    max_threads = _get_openmp_max_threads()
    if not max_threads or max_threads < 2:
        pytest.skip("OpenMP not available or only one thread")
    return max_threads


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dataset", DATASETS)
def test_openmp_toggle_consistency(dim, method, dataset):
    """Test OpenMP toggle consistency for deposition results."""
    max_threads = _require_openmp()

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

    fields_serial = sim.deposit_to_grid(
        fields=fields,
        averaged=[False] * fields.shape[1],
        gridnums=GRIDNUM,
        method=method,
        use_openmp=False,
    )

    fields_omp = sim.deposit_to_grid(
        fields=fields,
        averaged=[False] * fields.shape[1],
        gridnums=GRIDNUM,
        method=method,
        use_openmp=True,
        omp_threads=min(2, max_threads),
    )

    assert fields_serial.shape == fields_omp.shape
    np.testing.assert_allclose(fields_serial, fields_omp, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dataset", DATASETS)
def test_openmp_thread_counts_consistency(dim, method, dataset):
    """Test consistency of deposition with different OpenMP thread counts."""
    max_threads = _require_openmp()
    thread_counts = [t for t in (2, 4, 8) if t <= max_threads]
    if len(thread_counts) < 2:
        pytest.skip("Not enough OpenMP threads available for comparison")

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

    fields_ref = sim.deposit_to_grid(
        fields=fields,
        averaged=[False] * fields.shape[1],
        gridnums=GRIDNUM,
        method=method,
        use_openmp=True,
        omp_threads=thread_counts[0],
    )

    for threads in thread_counts[1:]:
        fields_test = sim.deposit_to_grid(
            fields=fields,
            averaged=[False] * fields.shape[1],
            gridnums=GRIDNUM,
            method=method,
            use_openmp=True,
            omp_threads=threads,
        )
        assert fields_ref.shape == fields_test.shape
        np.testing.assert_allclose(fields_ref, fields_test, rtol=1e-4, atol=1e-6)
