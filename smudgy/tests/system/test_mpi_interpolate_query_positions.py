"""Multi-rank correctness tests for `interpolate(query_positions=...)`
(arbitrary, non-particle query positions), routed via
`decomposition.route_query_positions` + a generalized
`ghosts.exchange_ghosts(..., target_positions=...)` -- the only path, since
Problem 1 made decomposition-at-construction mandatory.

The strongest check, mirroring `test_mpi_local_pipeline.py`'s own pattern:
run the full pipeline (find_neighbors -> compute_smoothing ->
compute_density -> add_fields -> interpolate(query_positions=grid)) at a
single rank (reference) and again at several rank counts on identical input
and identical query positions, gather + reassemble each via
`query_routing.local_global_indices`, and confirm they agree numerically.
Also covers the deadlock-safety edge case (a query batch small/skewed enough
that at least one rank is routed zero query points) and an out-of-domain
(extrapolation) query point.

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
        # Comfortably clear of the periodic ghost-exchange half-box guard:
        # routing query points onto a *subset* of a rank's own particle
        # region (fewer points than local particles) needs more density
        # headroom than the particle-only case does at this k/rank count.
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

    pc = PointCloud(
        positions=positions.copy(), weights=weights.copy(), boxsize=boxsize,
        verbose=False, backend="taichi", arch="cpu",
    ).global_setup(kernel_name="cubic_spline", num_neighbors=8, structure=structure)
    pc.find_neighbors()
    pc.compute_smoothing()
    pc.compute_density()
    pc.add_fields("sf", scalar_field)
    interp_local = pc.interpolate("sf", query_positions=query.copy(), structure=structure)
    interp_full = pc.gather_queries(interp_local)

    if rank == 0:
        np.savez(out_path, interp=interp_full)
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

    pc = PointCloud(
        positions=positions.copy(), weights=weights.copy(), boxsize=1.0,
        verbose=False, backend="taichi", arch="cpu",
    ).global_setup(kernel_name="cubic_spline", num_neighbors=8, structure="isotropic")
    pc.find_neighbors()
    pc.compute_smoothing()
    pc.compute_density()
    pc.add_fields("sf", scalar_field)
    interp_local = pc.interpolate("sf", query_positions=query.copy(), structure="isotropic")
    interp_full = pc.gather_queries(interp_local)

    if rank == 0:
        np.savez(out_path, interp=interp_full)
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
        ("uniform", "isotropic", False, 200, False, 3),
        ("uniform", "covariant", False, 200, False, 3),
        ("uniform", "isotropic", True, 200, False, 3),
        ("uniform", "covariant", True, 200, False, 3),
        ("anisotropic", "isotropic", False, 200, False, 4),
        ("wraparound", "isotropic", True, 200, False, 3),
        ("uniform", "isotropic", False, 200, True, 3),  # extrapolation points
    ],
)
def test_query_positions_local_path_matches_single_rank_reference(
    tmp_path, dist_mode, structure, periodic, n_query, extrapolate, n_ranks
):
    ref_path = tmp_path / "ref.npz"
    out_path = tmp_path / "result.npz"
    args_tail = [
        dist_mode, structure, "1" if periodic else "0", str(n_query),
        "1" if extrapolate else "0",
    ]
    _mpiexec(1, ["pipeline", str(ref_path), *args_tail])
    _mpiexec(n_ranks, ["pipeline", str(out_path), *args_tail])

    ref = np.load(ref_path)
    result = np.load(out_path)
    assert np.allclose(ref["interp"], result["interp"], rtol=1e-3, atol=1e-6)


def test_zero_query_point_ranks_no_deadlock(tmp_path):
    ref_path = tmp_path / "ref.npz"
    out_path = tmp_path / "result.npz"
    _mpiexec(1, ["zero_query_ranks", str(ref_path)], timeout=60)
    _mpiexec(8, ["zero_query_ranks", str(out_path)], timeout=60)

    ref = np.load(ref_path)
    result = np.load(out_path)
    assert np.allclose(ref["interp"], result["interp"], rtol=1e-3, atol=1e-6)


if __name__ == "__main__":
    _run_under_mpi()
