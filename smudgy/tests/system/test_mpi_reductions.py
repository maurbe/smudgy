"""Buffer-based MPI reductions in `smudgy.execution` (no pickling).

Checks `_reduce_sum` (Allreduce), `_reduce_sum_to_root` (Reduce) and
`_gatherv_rows` (Gatherv) against numpy references, on C-contiguous arrays, on the
``moveaxis`` views the quadrants deposition returns (reduced in place on the
contiguous base), on read-only inputs, and with a tiny chunk size that forces
the chunked path used for > 2**30-element grids. Also checks the per-rank
thread split derived from the launcher's node-local rank variables.

Run directly under MPI:
    mpiexec -n 3 python test_mpi_reductions.py

Run via pytest (the test spawns mpiexec itself):
    pytest test_mpi_reductions.py
"""

import subprocess
import sys

import numpy as np

from _mpi_launch import MPIEXEC, mpi_env


def _run_under_mpi():
    from mpi4py import MPI

    from smudgy import execution
    from smudgy.backend import _node

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    def local(r, shape):
        return np.random.default_rng(100 + r).random(shape, dtype=np.float32)

    ref_grid = sum(local(r, (5, 6, 3)) for r in range(size))  # (*g, F)
    ref_w = sum(local(r, (5, 6)) for r in range(size))

    for max_count in (2**30, 7):  # 7: chunked, with a ragged last chunk
        execution._MAX_COUNT = max_count
        # deposition-like tuple: (moveaxis view (F, *g), contiguous weights)
        f, w = execution._reduce_sum(
            comm, (np.moveaxis(local(rank, (5, 6, 3)), -1, 0), local(rank, (5, 6)))
        )
        np.testing.assert_allclose(f, np.moveaxis(ref_grid, -1, 0), rtol=1e-6)
        np.testing.assert_allclose(w, ref_w, rtol=1e-6)

        # read-only input is copied, not written
        ro = local(rank, (5, 6))
        ro.flags.writeable = False
        np.testing.assert_allclose(execution._reduce_sum(comm, ro), ref_w, rtol=1e-6)

        for root in (0, size - 1):
            f, w = execution._reduce_sum_to_root(
                comm,
                (np.moveaxis(local(rank, (5, 6, 3)), -1, 0), local(rank, (5, 6))),
                root=root,
            )
            if rank == root:
                np.testing.assert_allclose(f, np.moveaxis(ref_grid, -1, 0), rtol=1e-6)
                np.testing.assert_allclose(w, ref_w, rtol=1e-6)
            else:
                assert f is None and w is None

    # Gatherv with unequal (and zero) row counts, multi-dim rows
    rows = np.full((rank, 2, 3), rank, dtype=np.int64)
    out = execution._gatherv_rows(comm, rows, root=0)
    if rank == 0:
        expect = np.concatenate([np.full((r, 2, 3), r) for r in range(size)])
        np.testing.assert_array_equal(out, expect)
    else:
        assert out is None

    # every rank sees its node-local rank from the launcher
    local_rank, local_size = _node.node_local()
    assert local_size == size and local_rank == rank, (local_rank, local_size)
    threads = _node.cpu_threads()
    assert threads is not None and 1 <= threads <= max(1, __import__("os").cpu_count())
    print(f"OK {rank}")


def test_buffer_reductions():
    result = subprocess.run(
        [MPIEXEC, "-n", "3", sys.executable, __file__],
        capture_output=True,
        env=mpi_env(),
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    ok = sorted(line for line in result.stdout.splitlines() if line.startswith("OK "))
    assert ok == ["OK 0", "OK 1", "OK 2"]


if __name__ == "__main__":
    _run_under_mpi()
