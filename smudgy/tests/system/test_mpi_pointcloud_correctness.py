"""Regression test: MPI rank count must not change PointCloud results.

Guards `pointcloud.py`'s local-index-then-gather refactor (each rank now
slices to its own `execution._local_slice` range *before* doing per-particle
fancy indexing, instead of building the full array and letting
`execution._scatter` chunk it afterward -- see the MPI scaling investigation
that motivated this). A subtle off-by-one or misaligned index in that
refactor would silently corrupt results only under multi-rank execution,
which nothing else in the test suite exercises: `test_mpi_ranks.py` only
checks rank visibility and `test_bcast_array_large.py` only checks the
`_bcast_array` helper.

Runs a full compute_smoothing -> compute_density -> interpolate -> deposit
pipeline (isotropic and covariant structures, plus a plane_projection
deposit to exercise the `project_2d` / `reduce=False` path) under
`mpiexec -n 1` and `mpiexec -n 3`. 3 ranks (not 2 or 4) so the uneven-
remainder branch of `_local_slice` is exercised for a particle count that
isn't a multiple of the rank count.

Run directly under MPI:
    mpiexec -n 3 python test_mpi_pointcloud_correctness.py <out.npz>

Run via pytest (the test spawns mpiexec itself):
    pytest test_mpi_pointcloud_correctness.py
"""

import subprocess
import sys


def _run_under_mpi():
    import numpy as np
    from mpi4py import MPI

    from smudgy import PointCloud

    out_path = sys.argv[1]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    rng = np.random.default_rng(0)
    n = 37  # not a multiple of 2 or 3: exercises the remainder branch
    dim = 3
    positions = rng.uniform(0, 1, size=(n, dim)).astype(np.float32)
    weights = rng.uniform(0.5, 1.5, size=n).astype(np.float32)
    boxsize = np.ones(dim, dtype=np.float32)
    scalar_field = rng.uniform(size=n).astype(np.float32)
    query_positions = rng.uniform(0, 1, size=(11, dim)).astype(np.float32)

    results = {}
    for structure in ("isotropic", "covariant"):
        pc = PointCloud(
            positions=positions,
            weights=weights,
            boxsize=boxsize,
            verbose=False,
            backend="taichi",
            arch="cpu",
        ).global_setup(kernel_name="cubic_spline", num_neighbors=8, structure=structure)
        pc.compute_smoothing()
        pc.compute_density()
        pc.add_fields("sf", scalar_field)

        density = (
            pc.smoothing.density_covariant
            if structure == "covariant"
            else pc.smoothing.density_isotropic
        )
        results[f"{structure}_density"] = density
        results[f"{structure}_interp_self"] = pc.interpolate(
            "sf", structure=structure
        )
        results[f"{structure}_interp_query"] = pc.interpolate(
            "sf", query_positions=query_positions, structure=structure
        )

        fgrid, wgrid = pc.deposit(
            "sf",
            averaged=True,
            gridnums=6,
            adaptive=True,
            structure=structure,
            return_weights=True,
        )
        results[f"{structure}_deposit_fields"] = fgrid
        results[f"{structure}_deposit_weights"] = wgrid

        if structure == "covariant":
            fgrid2d, wgrid2d = pc.deposit(
                "sf",
                averaged=True,
                gridnums=6,
                adaptive=True,
                structure="covariant",
                plane_projection=[0, 1],
                return_weights=True,
            )
            results["covariant_deposit_2d_fields"] = fgrid2d
            results["covariant_deposit_2d_weights"] = wgrid2d

    if rank == 0:
        np.savez(out_path, **results)
    print(f"RANK {rank} DONE")


def _run_pipeline(tmp_path, n_ranks):
    out_path = tmp_path / f"result_r{n_ranks}.npz"
    result = subprocess.run(
        ["mpiexec", "-n", str(n_ranks), sys.executable, __file__, str(out_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    seen_ranks = {
        int(line.split()[1])
        for line in result.stdout.splitlines()
        if line.startswith("RANK")
    }
    assert seen_ranks == set(range(n_ranks)), (seen_ranks, result.stdout)
    return out_path


def test_results_match_across_rank_counts(tmp_path):
    import numpy as np

    single_rank = np.load(_run_pipeline(tmp_path, 1))
    multi_rank = np.load(_run_pipeline(tmp_path, 3))

    assert set(single_rank.files) == set(multi_rank.files)
    for key in single_rank.files:
        np.testing.assert_allclose(
            single_rank[key],
            multi_rank[key],
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"mismatch for {key!r} between 1 and 3 MPI ranks",
        )


if __name__ == "__main__":
    _run_under_mpi()
