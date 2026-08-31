"""Multi-rank correctness tests for Step 4a of the domain-decomposition
roadmap: `compute_smoothing`/`compute_density`/`interpolate`/`deposit`
actually using local+ghost data (via `self.decomposition`/`self.ghosts`)
instead of the full-replication path.

The strongest check: run the full pipeline
(decompose -> find_neighbors -> compute_smoothing -> compute_density ->
add_fields -> interpolate -> deposit) both with and without
decompose()/find_neighbors() on identical input, and confirm the two paths
agree numerically (not exactly -- the two paths sum neighbor contributions
in a different order, so float32 rounding differs at the ~1e-4 level, same
as any reordered floating-point summation; `deposit`'s grids, whose
`allreduce`-based sum is already global regardless of path, tend to agree
much more tightly). Also covers the two path-consistency safety nets Step 4a
added: a `num_neighbors` mismatch between `find_neighbors()` and a later
`compute_smoothing()` call falls back correctly, and an adaptive `deposit()`
call falls back correctly when `compute_smoothing()` didn't use the ghost
path.

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
        n = 300
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


def _gather_to_full(comm, rank, n, local_global, local_arr):
    """Gather a rank-local, local_global-indexed array into a full-N array
    in original particle order, on rank 0 only (None elsewhere)."""
    all_global = comm.gather(local_global, root=0)
    all_local = comm.gather(local_arr, root=0)
    if rank != 0:
        return None
    trailing_shape = local_arr.shape[1:]
    full = np.empty((n, *trailing_shape), dtype=local_arr.dtype)
    for g, a in zip(all_global, all_local):
        full[g] = a
    return full


def _run_pipeline_under_mpi(out_path, dist_mode, structure, periodic):
    from mpi4py import MPI

    from smudgy import PointCloud

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    periodic = periodic == "1"

    rng = np.random.default_rng(5)
    positions, weights = _make_dataset(dist_mode, rng)
    n = positions.shape[0]
    scalar_field = np.random.default_rng(6).uniform(size=n).astype(np.float32)
    boxsize = 1.0 if periodic else None
    # deposit() requires boxsize or an explicit extent -- non-periodic clouds
    # here have no boxsize, so give it the same [0,1]^dim domain explicitly.
    deposit_extent = None if periodic else [[0.0, 1.0]] * positions.shape[1]

    def run(use_local):
        pc = PointCloud(
            positions=positions.copy(), weights=weights.copy(), boxsize=boxsize,
            verbose=False, backend="taichi", arch="cpu",
        ).global_setup(kernel_name="cubic_spline", num_neighbors=8, structure=structure)
        if use_local:
            pc.decompose()
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
        if use_local:
            local_global = pc.decomposition.local_global_indices
            density = _gather_to_full(comm, rank, n, local_global, density_local)
            interp = _gather_to_full(comm, rank, n, local_global, interp)
        else:
            density, interp = density_local.copy(), interp.copy()
        return density, interp, fgrid.copy(), wgrid.copy()

    density_a, interp_a, fgrid_a, wgrid_a = run(use_local=False)
    density_b, interp_b, fgrid_b, wgrid_b = run(use_local=True)

    if rank == 0:
        result = {
            "density_matches": np.allclose(density_a, density_b, rtol=1e-3, atol=1e-6),
            "interp_matches": np.allclose(interp_a, interp_b, rtol=1e-3, atol=1e-6),
            "fgrid_matches": np.allclose(fgrid_a, fgrid_b, rtol=1e-3, atol=1e-6),
            "wgrid_matches": np.allclose(wgrid_a, wgrid_b, rtol=1e-3, atol=1e-6),
        }
        np.savez(out_path, **{k: np.asarray(v) for k, v in result.items()})
    print(f"RANK {rank} DONE")


def _run_num_neighbors_mismatch_under_mpi(out_path):
    """find_neighbors(num_neighbors=8) then compute_smoothing(num_neighbors=16)
    -- the ghost path's k doesn't match, must fall back to the full path and
    still give the right answer (checked against a from-scratch, no-ghosts
    run at k=16)."""
    from mpi4py import MPI

    from smudgy import PointCloud

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    rng = np.random.default_rng(9)
    positions, weights = _make_dataset("uniform", rng)
    n = positions.shape[0]

    pc = PointCloud(
        positions=positions.copy(), weights=weights.copy(), boxsize=1.0,
        verbose=False, backend="taichi", arch="cpu",
    ).global_setup(kernel_name="cubic_spline", structure="isotropic")
    pc.decompose()
    pc.find_neighbors(num_neighbors=8)
    pc.compute_smoothing(num_neighbors=16, structure="isotropic")
    used_ghosts_after_mismatch = pc.smoothing.used_ghosts
    pc.compute_density(structure="isotropic")
    density_mismatch_path = pc.smoothing.density_isotropic.copy()

    pc_ref = PointCloud(
        positions=positions.copy(), weights=weights.copy(), boxsize=1.0,
        verbose=False, backend="taichi", arch="cpu",
    ).global_setup(kernel_name="cubic_spline", structure="isotropic")
    pc_ref.compute_smoothing(num_neighbors=16, structure="isotropic")
    pc_ref.compute_density(structure="isotropic")
    density_ref = pc_ref.smoothing.density_isotropic.copy()

    fell_back_correctly = used_ghosts_after_mismatch is False
    values_match = bool(np.allclose(density_mismatch_path, density_ref, rtol=1e-5))

    if rank == 0:
        np.savez(
            out_path,
            fell_back_correctly=np.asarray(fell_back_correctly),
            values_match=np.asarray(values_match),
        )
    print(f"RANK {rank} DONE")


def _run_deposit_fallback_under_mpi(out_path):
    """compute_smoothing() called BEFORE decompose() (so used_ghosts=False),
    then decompose()+find_neighbors() called, then adaptive deposit(). Must
    fall back to the full path for deposit rather than indexing
    smoothing_lengths (full-N) with a local idx."""
    from mpi4py import MPI

    from smudgy import PointCloud

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    rng = np.random.default_rng(17)
    positions, weights = _make_dataset("uniform", rng)
    scalar_field = np.random.default_rng(18).uniform(size=positions.shape[0]).astype(np.float32)

    pc = PointCloud(
        positions=positions.copy(), weights=weights.copy(), boxsize=1.0,
        verbose=False, backend="taichi", arch="cpu",
    ).global_setup(kernel_name="cubic_spline", num_neighbors=8, structure="isotropic")
    pc.compute_smoothing()  # before decompose() -- used_ghosts False
    used_ghosts_before = pc.smoothing.used_ghosts
    pc.decompose()
    pc.find_neighbors()
    pc.compute_density()
    pc.add_fields("sf", scalar_field)
    # must not raise / must not silently use the wrong index convention
    fgrid, wgrid = pc.deposit(
        "sf", averaged=True, gridnums=6, adaptive=True, structure="isotropic",
        return_weights=True,
    )
    finite = bool(np.all(np.isfinite(fgrid)) and np.all(np.isfinite(wgrid)))

    ok = comm.allreduce((used_ghosts_before is False) and finite, op=MPI.LAND)
    if rank == 0:
        np.savez(out_path, ok=np.asarray(ok))
    print(f"RANK {rank} DONE")


def _run_under_mpi():
    mode = sys.argv[1]
    if mode == "pipeline":
        _run_pipeline_under_mpi(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif mode == "num_neighbors_mismatch":
        _run_num_neighbors_mismatch_under_mpi(sys.argv[2])
    elif mode == "deposit_fallback":
        _run_deposit_fallback_under_mpi(sys.argv[2])
    else:
        raise ValueError(f"unknown mode {mode!r}")


def _mpiexec(n_ranks, args, timeout=60):
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
        ("uniform", "isotropic", False, 1),
        ("uniform", "isotropic", False, 3),
        ("uniform", "covariant", False, 3),
        ("uniform", "isotropic", True, 3),
        ("uniform", "covariant", True, 3),
        ("anisotropic", "isotropic", False, 4),
        ("wraparound", "isotropic", True, 3),
    ],
)
def test_local_path_matches_full_replication_path(
    tmp_path, dist_mode, structure, periodic, n_ranks
):
    out_path = tmp_path / "result.npz"
    _mpiexec(
        n_ranks,
        ["pipeline", str(out_path), dist_mode, structure, "1" if periodic else "0"],
        timeout=90,
    )
    result = np.load(out_path)

    assert int(result["density_matches"]) == 1
    assert int(result["interp_matches"]) == 1
    assert int(result["fgrid_matches"]) == 1
    assert int(result["wgrid_matches"]) == 1


def test_num_neighbors_mismatch_falls_back_correctly(tmp_path):
    out_path = tmp_path / "result.npz"
    _mpiexec(3, ["num_neighbors_mismatch", str(out_path)])
    result = np.load(out_path)

    assert int(result["fell_back_correctly"]) == 1
    assert int(result["values_match"]) == 1


def test_deposit_falls_back_when_smoothing_used_old_path(tmp_path):
    out_path = tmp_path / "result.npz"
    _mpiexec(3, ["deposit_fallback", str(out_path)])
    result = np.load(out_path)

    assert int(result["ok"]) == 1


if __name__ == "__main__":
    _run_under_mpi()
