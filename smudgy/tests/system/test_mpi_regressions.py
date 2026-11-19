"""MPI regressions: calls that used to deadlock when ranks diverged.

- ``add_fields`` with several names, where non-root ranks pass ``values=None``
  (values are only read on rank 0).
- ``deposit(extent=...)`` where some ranks own no particle inside the extent
  (separable deposition raised on those ranks only).
"""

import subprocess
import sys

from _mpi_launch import MPIEXEC, mpi_env


def _run_under_mpi():
    import warnings

    import numpy as np
    from mpi4py import MPI

    from smudgy import PointCloud

    warnings.simplefilter("ignore")
    rank = MPI.COMM_WORLD.Get_rank()
    rng = np.random.default_rng(0)
    n = 900
    pos = rng.uniform(0, 1, (n, 2))
    a, b = rng.standard_normal(n), rng.standard_normal((n, 2))

    pc = PointCloud(pos if rank == 0 else None, verbose=False, arch="cpu")
    pc.add_fields(["a", "b"], [a, b] if rank == 0 else None)
    assert np.array_equal(pc.gather_particles(pc.a), a.astype(np.float32)) or rank != 0

    pc.global_setup(num_neighbors=8, kernel_name="tsc")
    pc.compute_smoothing(structure="isotropic")
    # a corner of the domain: most ranks of the Hilbert partition own nothing in it
    extent = [[0.0, 0.2], [0.0, 0.2]]
    for structure in ("separable", "isotropic"):
        grid = pc.deposit("a", averaged=False, gridnums=8, extent=extent,
                          structure=structure, gather_to_root=True)
        if rank == 0:
            inside = np.all((pos >= 0.0) & (pos <= 0.2), axis=1)
            assert grid.shape == (1, 8, 8) and np.isfinite(grid).all()
            assert inside.any()
    print(f"OK {rank}")


def test_divergent_rank_inputs_do_not_deadlock():
    result = subprocess.run(
        [MPIEXEC, "-n", "4", sys.executable, __file__],
        capture_output=True,
        env=mpi_env(),
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    ok = sorted(line for line in result.stdout.splitlines() if line.startswith("OK "))
    assert ok == [f"OK {r}" for r in range(4)]


if __name__ == "__main__":
    _run_under_mpi()
