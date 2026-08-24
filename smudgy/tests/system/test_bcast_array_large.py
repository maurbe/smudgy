"""Regression test for execution._bcast_array against mpi4py's pickle-bcast
~2GB payload ceiling (see PointCloud.compute_smoothing's real-world crash
on S3IT/LUMI: MPI_ERR_ARG when broadcasting large nn_dists/nn_inds arrays).

Uses a float64 array sized just over 2**31 bytes (~2GiB) -- large enough to
exceed mpi4py's pickle-based `comm.bcast`'s 32-bit byte-count ceiling, but
with an element count (~268M) far under the 32-bit element-count ceiling
that buffer-based `comm.Bcast` (what `_bcast_array` uses) is subject to.
There is no way to exercise this bug with a meaningfully smaller payload,
since the ceiling itself is defined in bytes.

Run directly under MPI:
    mpiexec -n 2 python test_bcast_array_large.py

Run via pytest (the test spawns mpiexec itself):
    pytest test_bcast_array_large.py
"""

import subprocess
import sys


def _run_under_mpi():
    import numpy as np
    from mpi4py import MPI

    from smudgy import execution

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    n = (2**31 + 8192) // 8  # float64 elements; nbytes just over 2**31
    if rank == 0:
        arr = np.zeros(n, dtype=np.float64)
        arr[0] = 7.0
        arr[-1] = 9.0
    else:
        arr = None

    result = execution._bcast_array(comm, arr)

    ok = (
        result.shape == (n,)
        and result.dtype == np.float64
        and result.nbytes > 2**31
        and result[0] == 7.0
        and result[-1] == 9.0
    )
    print(f"RANK {rank} OK {ok}")
    if not ok:
        sys.exit(1)


def test_bcast_array_handles_over_2gb_payload():
    result = subprocess.run(
        ["mpiexec", "-n", "2", sys.executable, __file__],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    seen = {
        int(line.split()[1]): line.split()[3] == "True"
        for line in result.stdout.splitlines()
        if line.startswith("RANK")
    }
    assert seen == {0: True, 1: True}, (seen, result.stdout, result.stderr)


if __name__ == "__main__":
    _run_under_mpi()
