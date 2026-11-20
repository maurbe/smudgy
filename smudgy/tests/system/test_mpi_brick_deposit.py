"""Brick deposition (``deposit(brick=True)``) against full-grid deposition.

Under ``mpiexec -n 1/3/4``: every deposit method is run with ``brick=True``,
``brick=False`` and the automatic ``brick=None`` on the same rank count (the two must agree up to the
order of float32 additions), and the brick results are compared across rank
counts. The particles include a blob straddling a periodic corner (bricks
extending past the grid edges) and an ``extent`` that leaves some ranks
without particles.

Run directly under MPI:
    mpiexec -n 3 python test_mpi_brick_deposit.py <out.npz>
"""

import subprocess
import sys

import numpy as np
import pytest
from _mpi_launch import MPIEXEC, mpi_env

RTOL, ATOL = 1e-5, 1e-6


def _cases():
    """(name, deposit kwargs, backend) of every deposit checked."""
    base = dict(fields="sf", averaged=False, gridnums=24, return_weights=True)
    cases = [
        ("cic", dict(base, adaptive=False, kernel_name="cic"), "quadrants"),
        ("bspline6", dict(base, adaptive=False, kernel_name="bspline6"), "quadrants"),
        ("numpy_tsc", dict(base, adaptive=False, kernel_name="tsc"), "numpy"),
        ("separable", dict(base, structure="separable", kernel_name="tsc"), "quadrants"),
        ("isotropic", dict(base, structure="isotropic"), "quadrants"),
        ("isotropic_simpson", dict(base, structure="isotropic", gridnums=[8, 12, 10],
                                   integration_method="simpson"), "quadrants"),
        ("covariant", dict(base, structure="covariant"), "quadrants"),
        ("covariant_fine", dict(base, structure="covariant", gridnums=64), "quadrants"),
        ("covariant_extent", dict(base, structure="covariant",
                                  extent=[[0.0, 0.3], [0.7, 1.0], [0.3, 0.7]]), "quadrants"),
        ("isotropic_plane", dict(base, structure="isotropic", plane_projection=[0, 1]),
         "quadrants"),
        ("weights_only", dict(fields=None, averaged=False, gridnums=24,
                              structure="isotropic"), "quadrants"),
        ("averaged", dict(base, averaged=True, structure="covariant"), "quadrants"),
    ]
    return cases


def _positions(rng, n):
    """Uniform particles plus a blob around the periodic corner (0, 1, *)."""
    uniform = rng.uniform(0, 1, (n // 2, 3))
    blob = np.array([0.0, 1.0, 0.5]) + 0.05 * rng.standard_normal((n - n // 2, 3))
    return np.mod(np.concatenate([uniform, blob]), 1.0).astype(np.float32)


def _run_under_mpi():
    from mpi4py import MPI

    from smudgy import PointCloud, execution

    out_path = sys.argv[1]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    rng = np.random.default_rng(3)
    n = 2000
    positions = _positions(rng, n)
    weights = rng.uniform(0.5, 1.5, n).astype(np.float32)
    field = rng.uniform(size=n).astype(np.float32)

    clouds = {}
    for backend in ("quadrants", "numpy"):
        pc = PointCloud(positions, weights=weights, boxsize=1.0, verbose=False,
                        backend=backend)
        pc.add_fields("sf", field)
        pc.find_neighbors(num_neighbors=16)
        if backend == "quadrants":
            pc.global_setup(kernel_name="cubic_spline")
            pc.compute_smoothing(num_neighbors=16, structure="isotropic")
            pc.compute_smoothing(num_neighbors=16, structure="covariant")
        clouds[backend] = pc

    # count brick reductions, to check which path brick=None takes
    calls, sum_bricks = [0], execution._sum_bricks

    def counting_sum_bricks(*args, **kw):
        calls[0] += 1
        return sum_bricks(*args, **kw)

    execution._sum_bricks = counting_sum_bricks

    results, failures = {}, []
    for name, kwargs, backend in _cases():
        pc = clouds[backend]
        for gather in (False, True):
            brick = pc.deposit(**kwargs, brick=True, gather_to_root=gather)
            full = pc.deposit(**kwargs, brick=False, gather_to_root=gather)
            calls[0] = 0
            auto = pc.deposit(**kwargs, gather_to_root=gather)  # brick=None
            uses_bricks = gather and backend == "quadrants" and comm.Get_size() > 1
            if calls[0] != uses_bricks:
                failures.append(f"{name} gather={gather}: brick=None took the wrong path")
            brick = brick if isinstance(brick, tuple) else (brick,)
            full = full if isinstance(full, tuple) else (full,)
            auto = auto if isinstance(auto, tuple) else (auto,)
            brick = brick + auto
            full = full + full
            if gather and rank != 0:
                if any(g is not None for g in brick):
                    failures.append(f"{name}: non-root rank got a grid")
                continue
            for i, (b, f) in enumerate(zip(brick, full)):
                if not np.allclose(b, f, rtol=RTOL, atol=ATOL):
                    failures.append(
                        f"{name} gather={gather} grid {i}: max diff "
                        f"{np.max(np.abs(b - f))}"
                    )
                if gather and rank == 0:
                    results[f"{name}_{i}"] = b

    all_failures = comm.gather(failures, root=0)
    if rank == 0:
        flat = [f for r in all_failures for f in r]
        np.savez(out_path, failures=np.array(flat, dtype=str), **results)
    print(f"RANK {rank} DONE")


def _run(tmp_path, n_ranks):
    out_path = tmp_path / f"brick_r{n_ranks}.npz"
    result = subprocess.run(
        [MPIEXEC, "-n", str(n_ranks), sys.executable, __file__, str(out_path)],
        capture_output=True,
        env=mpi_env(),
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    seen = {int(line.split()[1]) for line in result.stdout.splitlines()
            if line.startswith("RANK")}
    assert seen == set(range(n_ranks)), (seen, result.stdout)
    return np.load(out_path)


@pytest.fixture(scope="module")
def single_rank(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("brick"), 1)


@pytest.mark.parametrize("n_ranks", [1, 3, 4])
def test_brick_matches_full_grid(tmp_path, single_rank, n_ranks):
    result = single_rank if n_ranks == 1 else _run(tmp_path, n_ranks)
    assert list(result["failures"]) == []
    for key in single_rank.files:
        if key != "failures":
            np.testing.assert_allclose(result[key], single_rank[key], rtol=RTOL,
                                       atol=ATOL, err_msg=key)


if __name__ == "__main__":
    _run_under_mpi()
