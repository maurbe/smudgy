"""Sanity check that 2 MPI ranks are visible via mpi4py.

Run directly under MPI:
    mpiexec -n 2 python test_mpi_ranks.py

Run via pytest (the test spawns mpiexec itself):
    pytest test_mpi_ranks.py
"""

import subprocess
import sys


def _run_under_mpi():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    print(f"RANK {comm.Get_rank()} SIZE {comm.Get_size()}")
    if comm.Get_size() != 2:
        sys.exit(1)


def test_two_ranks_visible():
    result = subprocess.run(
        ["mpiexec", "-n", "2", sys.executable, __file__],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    seen_ranks = {
        int(line.split()[1])
        for line in result.stdout.splitlines()
        if line.startswith("RANK")
    }
    assert seen_ranks == {0, 1}


if __name__ == "__main__":
    _run_under_mpi()
