"""Multi-rank correctness tests for the full local+ghost compute pipeline
(`compute_smoothing`/`compute_density`/`interpolate`/`deposit`, all using
`self.decomposition`/`self.ghosts` unconditionally -- there is no other path
since Problem 1 made decomposition-at-construction mandatory).

The strongest check: run the full pipeline (find_neighbors ->
compute_smoothing -> compute_density -> add_fields -> interpolate ->
deposit) at a single rank (trivially "the full dataset", and in original
input order -- decomposition skips the Hilbert sort entirely at size==1)
and again at several rank counts on identical input, gather each
back into original-particle order, and confirm they agree numerically (not
exactly -- different rank counts partition particles differently, reordering
floating-point summation; `deposit`'s grids, whose `allreduce`-based sum is
already global regardless of rank count, tend to agree much more tightly).
Also covers: a `num_neighbors` mismatch between `find_neighbors()` and a
later `compute_smoothing()` call now raises a clear error (there is no
fallback path left to silently reuse).

Run directly under MPI:
    mpiexec -n 3 python test_mpi_local_pipeline.py pipeline <out.npz> isotropic 1

Run via pytest (the test spawns mpiexec itself):
    pytest test_mpi_local_pipeline.py
"""

import subprocess
import sys

import numpy as np
import pytest


def _make_dataset(mode, rng, dim=3):
    if mode == "uniform":
        # Comfortably clear of the periodic ghost-exchange half-box guard at
        # k=8 across the rank counts used below (300 was too sparse in an
        # earlier iteration of this suite).
        n = 900
        positions = rng.uniform(0.0, 1.0, size=(n, dim)).astype(np.float32)
    elif mode == "anisotropic":
        n_dense, n_sparse = 250, 50
        dense = rng.uniform(0.0, 0.02, size=(n_dense, dim))
        sparse = rng.uniform(0.0, 1.0, size=(n_sparse, dim))
        positions = np.concatenate([dense, sparse], axis=0).astype(np.float32)
    elif mode == "wraparound":
        n_a, n_b = 40, 40
        a = np.stack(
            [rng.uniform(0.0, 0.02, n_a), rng.uniform(0.0, 1.0, n_a), rng.uniform(0.0, 1.0, n_a)],
            axis=1,
        )
        b = np.stack(
            [rng.uniform(0.98, 1.0, n_b), rng.uniform(0.0, 1.0, n_b), rng.uniform(0.0, 1.0, n_b)],
            axis=1,
        )
        filler = rng.uniform(0.0, 1.0, size=(220, dim))
        positions = np.concatenate([a, b, filler], axis=0).astype(np.float32)
    else:
        raise ValueError(mode)
    weights = rng.uniform(0.5, 1.5, size=positions.shape[0]).astype(np.float32)
    return positions, weights


def _run_pipeline_under_mpi(out_path, dist_mode, structure, periodic):
    from mpi4py import MPI

    from smudgy import PointCloud

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    periodic = periodic == "1"

    rng = np.random.default_rng(5)
    positions, weights = _make_dataset(dist_mode, rng)
    dim = positions.shape[1]
    scalar_field = np.random.default_rng(6).uniform(size=positions.shape[0]).astype(np.float32)
    boxsize = 1.0 if periodic else None
    # deposit() requires boxsize or an explicit extent -- non-periodic clouds
    # here have no boxsize, so give it the same [0,1]^dim domain explicitly.
    deposit_extent = None if periodic else [[0.0, 1.0]] * dim

    pc = PointCloud(
        positions=positions.copy(), weights=weights.copy(), boxsize=boxsize,
        verbose=False, backend="taichi", arch="cpu",
    ).global_setup(kernel_name="cubic_spline", num_neighbors=8, structure=structure)
    pc.find_neighbors()
    pc.compute_smoothing()
    pc.compute_density()
    pc.add_fields("sf", scalar_field)
    interp = pc.interpolate("sf", structure=structure)
    fgrid, wgrid = pc.deposit(
        "sf", averaged=True, gridnums=6, adaptive=True, structure=structure,
        extent=deposit_extent, return_weights=True,
    )

    density_local = (
        pc.smoothing.density_covariant if structure == "covariant"
        else pc.smoothing.density_isotropic
    )
    density_full = pc.gather_particles(density_local)
    interp_full = pc.gather_particles(interp)

    if rank == 0:
        np.savez(
            out_path,
            density=density_full,
            interp=interp_full,
            fgrid=fgrid.copy(),
            wgrid=wgrid.copy(),
        )
    print(f"RANK {rank} DONE")


def _run_num_neighbors_mismatch_under_mpi(out_path):
    """find_neighbors(num_neighbors=8) then compute_smoothing(num_neighbors=16)
    -- the ghost data's k doesn't match the request. There is no fallback
    path to silently reuse anymore, so this must raise a clear, actionable
    error instead."""
    from mpi4py import MPI

    from smudgy import PointCloud

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    rng = np.random.default_rng(9)
    positions, weights = _make_dataset("uniform", rng)

    pc = PointCloud(
        positions=positions.copy(), weights=weights.copy(), boxsize=1.0,
        verbose=False, backend="taichi", arch="cpu",
    ).global_setup(kernel_name="cubic_spline", structure="isotropic")
    pc.find_neighbors(num_neighbors=8)
    raised = False
    try:
        pc.compute_smoothing(num_neighbors=16, structure="isotropic")
    except ValueError:
        raised = True
    all_raised = comm.allreduce(raised, op=MPI.LAND)

    if rank == 0:
        np.savez(out_path, all_raised=np.asarray(all_raised))
    print(f"RANK {rank} DONE")


def _run_under_mpi():
    mode = sys.argv[1]
    if mode == "pipeline":
        _run_pipeline_under_mpi(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif mode == "num_neighbors_mismatch":
        _run_num_neighbors_mismatch_under_mpi(sys.argv[2])
    else:
        raise ValueError(f"unknown mode {mode!r}")


def _mpiexec(n_ranks, args, timeout=90):
    try:
        result = subprocess.run(
            ["mpiexec", "-n", str(n_ranks), sys.executable, __file__, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"mpiexec -n {n_ranks} {' '.join(args)} did not finish within {timeout}s"
        ) from exc
    assert result.returncode == 0, result.stderr
    seen_ranks = {
        int(line.split()[1])
        for line in result.stdout.splitlines()
        if line.startswith("RANK")
    }
    assert seen_ranks == set(range(n_ranks)), (seen_ranks, result.stdout, result.stderr)


@pytest.mark.parametrize(
    "dist_mode,structure,periodic,n_ranks",
    [
        ("uniform", "isotropic", False, 3),
        ("uniform", "covariant", False, 3),
        ("uniform", "isotropic", True, 3),
        ("uniform", "covariant", True, 3),
        ("anisotropic", "isotropic", False, 4),
        ("wraparound", "isotropic", True, 3),
    ],
)
def test_local_path_matches_single_rank_reference(
    tmp_path, dist_mode, structure, periodic, n_ranks
):
    ref_path = tmp_path / "ref.npz"
    out_path = tmp_path / "result.npz"
    args_tail = [dist_mode, structure, "1" if periodic else "0"]
    _mpiexec(1, ["pipeline", str(ref_path), *args_tail])
    _mpiexec(n_ranks, ["pipeline", str(out_path), *args_tail])

    ref = np.load(ref_path)
    result = np.load(out_path)

    for key in ("density", "interp", "fgrid", "wgrid"):
        assert np.allclose(ref[key], result[key], rtol=1e-3, atol=1e-6), key


def test_num_neighbors_mismatch_raises_clear_error(tmp_path):
    out_path = tmp_path / "result.npz"
    _mpiexec(3, ["num_neighbors_mismatch", str(out_path)])
    result = np.load(out_path)

    assert int(result["all_raised"]) == 1


if __name__ == "__main__":
    _run_under_mpi()
