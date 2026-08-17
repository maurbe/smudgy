import pytest

from smudgy.backend.taichi import init as taichi_init


@pytest.fixture(scope="session", autouse=True)
def _force_taichi_cpu():
    """Force the Taichi runtime to CPU arch before any test runs.

    Without this, whichever test happens to construct the first
    PointCloud(backend="taichi") (with no explicit arch) wins the default
    "gpu" arch for the rest of the pytest session -- non-deterministic
    across machines/CI runners and, per backend/taichi/__init__.py's
    docstring, unsafe to switch mid-process once other tests have already
    compiled kernels under a different arch. Every backend-consistency test
    is meant to compare Numpy CPU against Taichi CPU anyway.
    """
    taichi_init(arch="cpu")
