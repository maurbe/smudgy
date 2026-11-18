import pytest

from smudgy.backend.quadrants import init as quadrants_init


@pytest.fixture(scope="session", autouse=True)
def _force_quadrants_cpu():
    """Force the Quadrants runtime to CPU arch before any test runs.

    Without this, whichever test happens to construct the first
    PointCloud(backend="quadrants") (with no explicit arch) wins the default
    "gpu" arch for the rest of the pytest session -- non-deterministic
    across machines/CI runners and, per backend/quadrants/__init__.py's
    docstring, unsafe to switch mid-process once other tests have already
    compiled kernels under a different arch. Every backend-consistency test
    is meant to compare Numpy CPU against Quadrants CPU anyway.
    """
    quadrants_init(arch="cpu")
