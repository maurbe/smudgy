"""Multi-rank correctness tests for Step 4b of the domain-decomposition
roadmap: `interpolate(query_positions=...)` (arbitrary, non-particle query
positions) actually using local+ghost data (routed via
`decomposition.route_query_positions` + a generalized
`ghosts.exchange_ghosts(..., target_positions=...)`) instead of the
full-replication path.

The strongest check, mirroring `test_mpi_local_pipeline.py`'s own pattern for
Step 4a: run the full pipeline (decompose -> find_neighbors ->
compute_smoothing -> compute_density -> add_fields ->
interpolate(query_positions=grid)) both with and without decompose()/
find_neighbors() on identical input and identical query positions, gather +
reassemble the new path's local, routing-ordered result via
`query_routing.local_global_indices`, and confirm the two paths agree
numerically (not exactly -- reordered floating-point summation, same as
Step 4a). Also covers the deadlock-safety edge case (a query batch small/
skewed enough that at least one rank is routed zero query points) and an
out-of-domain (extrapolation) query point.

Run directly under MPI:
    mpiexec -n 3 python test_mpi_interpolate_query_positions.py pipeline <out.npz> isotropic 1

Run via pytest (the test spawns mpiexec itself):
    pytest test_mpi_interpolate_query_positions.py
"""

import subprocess
import sys

import numpy as np
import pytest


def _make_dataset(mode, rng, dim=3):
    if mode == "uniform":
        # Larger than test_mpi_local_pipeline.py's own n=300 for the same
        # dataset shape: routing query points onto a *subset* of a rank's
        # own particle region (fewer points than local particles) needs a
        # bit more density headroom to stay clear of the periodic half-box
        # guard than the particle-only case does at this k/rank count.
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


def _gather_to_full(comm, rank, m, local_global, local_arr):
    """Gather a rank-local, local_global-indexed array into a full-M array
    in original query order, on rank 0 only (None elsewhere)."""
    all_global = comm.gather(local_global, root=0)
    all_local = comm.gather(local_arr, root=0)
    if rank != 0:
        return None
    trailing_shape = local_arr.shape[1:]
    full = np.empty((m, *trailing_shape), dtype=local_arr.dtype)
    for g, a in zip(all_global, all_local):
        full[g] = a
    return full


def _run_pipeline_under_mpi(out_path, dist_mode, structure, periodic, n_query, extrapolate):
    from mpi4py import MPI

    from smudgy import PointCloud

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    periodic = periodic == "1"
    extrapolate = extrapolate == "1"

    rng = np.random.default_rng(41)
    positions, weights = _make_dataset(dist_mode, rng)
    n, dim = positions.shape
    scalar_field = np.random.default_rng(42).uniform(size=n).astype(np.float32)
    boxsize = 1.0 if periodic else None

    qrng = np.random.default_rng(43)
    lo, hi = (0.0, 1.0) if periodic else (-0.05 if extrapolate else 0.0, 1.05 if extrapolate else 1.0)
    query = qrng.uniform(lo, hi, size=(n_query, dim)).astype(np.float32)

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
        interp = pc.interpolate("sf", query_positions=query.copy(), structure=structure)

        if use_local:
            local_global = pc.query_routing.local_global_indices
            interp = _gather_to_full(comm, rank, n_query, local_global, interp)
        else:
            interp = interp.copy()
        return interp

    interp_a = run(use_local=False)
    interp_b = run(use_local=True)

    if rank == 0:
        result = {
            "interp_matches": np.allclose(interp_a, interp_b, rtol=1e-3, atol=1e-6),
        }
        np.savez(out_path, **{k: np.asarray(v) for k, v in result.items()})
    print(f"RANK {rank} DONE")


def _run_zero_query_ranks_under_mpi(out_path):
    """A tiny query batch (fewer points than ranks) must route some ranks to
    zero query points without deadlocking, and still give correct results
    for the ranks that DO get points."""
    from mpi4py import MPI

    from smudgy import PointCloud

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    rng = np.random.default_rng(51)
    positions, weights = _make_dataset("uniform", rng)
    n = positions.shape[0]
    scalar_field = np.random.default_rng(52).uniform(size=n).astype(np.float32)
    query = np.array([[0.2, 0.3, 0.4], [0.6, 0.5, 0.5]], dtype=np.float32)

    def run(use_local):
        pc = PointCloud(
            positions=positions.copy(), weights=weights.copy(), boxsize=1.0,
            verbose=False, backend="taichi", arch="cpu",
        ).global_setup(kernel_name="cubic_spline", num_neighbors=8, structure="isotropic")
        if use_local:
            pc.decompose()
            pc.find_neighbors()
        pc.compute_smoothing()
        pc.compute_density()
        pc.add_fields("sf", scalar_field)
        interp = pc.interpolate("sf", query_positions=query.copy(), structure="isotropic")
        if use_local:
            local_global = pc.query_routing.local_global_indices
            interp = _gather_to_full(comm, rank, query.shape[0], local_global, interp)
        else:
            interp = interp.copy()
        return interp

    interp_a = run(use_local=False)
    interp_b = run(use_local=True)

    if rank == 0:
        result = {"interp_matches": np.allclose(interp_a, interp_b, rtol=1e-3, atol=1e-6)}
        np.savez(out_path, **{k: np.asarray(v) for k, v in result.items()})
    print(f"RANK {rank} DONE")


def _run_under_mpi():
    mode = sys.argv[1]
    if mode == "pipeline":
        _run_pipeline_under_mpi(
            sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), sys.argv[7]
        )
    elif mode == "zero_query_ranks":
        _run_zero_query_ranks_under_mpi(sys.argv[2])
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
            f"mpiexec -n {n_ranks} {' '.join(args)} did not finish within "
            f"{timeout}s -- likely a deadlock."
        ) from exc
    assert result.returncode == 0, result.stderr
    seen_ranks = {
        int(line.split()[1])
        for line in result.stdout.splitlines()
        if line.startswith("RANK")
    }
    assert seen_ranks == set(range(n_ranks)), (seen_ranks, result.stdout, result.stderr)


@pytest.mark.parametrize(
    "dist_mode,structure,periodic,n_query,extrapolate,n_ranks",
    [
        ("uniform", "isotropic", False, 200, False, 1),
        ("uniform", "isotropic", False, 200, False, 3),
        ("uniform", "covariant", False, 200, False, 3),
        ("uniform", "isotropic", True, 200, False, 3),
        ("uniform", "covariant", True, 200, False, 3),
        ("anisotropic", "isotropic", False, 200, False, 4),
        ("wraparound", "isotropic", True, 200, False, 3),
        ("uniform", "isotropic", False, 200, True, 3),  # extrapolation points
    ],
)
def test_query_positions_local_path_matches_full_replication_path(
    tmp_path, dist_mode, structure, periodic, n_query, extrapolate, n_ranks
):
    out_path = tmp_path / "result.npz"
    _mpiexec(
        n_ranks,
        [
            "pipeline", str(out_path), dist_mode, structure,
            "1" if periodic else "0", str(n_query), "1" if extrapolate else "0",
        ],
    )
    result = np.load(out_path)
    assert int(result["interp_matches"]) == 1


def test_zero_query_point_ranks_no_deadlock(tmp_path):
    out_path = tmp_path / "result.npz"
    _mpiexec(8, ["zero_query_ranks", str(out_path)], timeout=60)
    result = np.load(out_path)
    assert int(result["interp_matches"]) == 1


if __name__ == "__main__":
    _run_under_mpi()
