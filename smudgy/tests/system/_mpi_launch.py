"""Launch helpers shared by the MPI system tests (which spawn ``mpiexec``).

Imported as a sibling module, both under pytest (rootdir-relative import of
the test file's directory) and when a test file is re-run by ``mpiexec`` as a
script (the script's directory is on ``sys.path``).
"""

import os
import shutil
import sys


def _find_mpiexec():
    # Prefer the launcher installed next to the running interpreter: a
    # different MPI's mpiexec earlier on PATH (e.g. a PMIx-based Open MPI)
    # cannot start the mpi4py build this interpreter links against.
    local = os.path.join(os.path.dirname(sys.executable), "mpiexec")
    if os.path.isfile(local) and os.access(local, os.X_OK):
        return local
    return shutil.which("mpiexec") or "mpiexec"


MPIEXEC = _find_mpiexec()


def mpi_env():
    """Environment for the spawned ranks.

    The pytest process itself has MPI initialised (``import smudgy`` imports
    ``mpi4py.MPI``). With MPICH's default libfabric ``sock`` provider, ranks
    launched from such a process can hang indefinitely in collectives (seen
    with 5 ranks on macOS); the ``tcp`` provider does not, and is faster.
    Only a default: an explicit ``FI_PROVIDER`` in the environment wins.
    """
    env = dict(os.environ)
    env.setdefault("FI_PROVIDER", "tcp")
    return env
