"""Multi-rank correctness tests for `PointCloud.decompose()` (Step 1 of the
domain-decomposition roadmap: Hilbert-curve partitioning + Scatterv
redistribution).

Guards the properties a decomposition must have regardless of rank count or
particle distribution: every particle assigned to exactly one rank and none
dropped (bijection), counts balanced to within 1, the original array is
exactly reconstructible from local chunks + provenance, and -- most
importantly -- that `decompose()` is truly opt-in and doesn't perturb
`self.positions`/`self.weights` or the existing compute pipeline's results
at all (the property that makes Step 1 safely landable on its own).

Run directly under MPI:
    mpiexec -n 3 python test_mpi_decompose.py props <out.npz> 37 uniform 0
    mpiexec -n 3 python test_mpi_decompose.py pipeline <out.npz>

Run via pytest (the test spawns mpiexec itself):
    pytest test_mpi_decompose.py
"""

import subprocess
import sys

import numpy as np
import pytest


def _run_props_under_mpi(out_path, n, dist_mode, periodic):
    from mpi4py import MPI

    from smudgy import PointCloud

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rng = np.random.default_rng(0)
    dim = 3
    if dist_mode == "anisotropic":
        n_dense = int(n * 0.9)
        n_sparse = n - n_dense
        dense = rng.uniform(0.0, 0.01, size=(n_dense, dim))
        sparse = rng.uniform(0.0, 1.0, size=(n_sparse, dim))
        positions = np.concatenate([dense, sparse], axis=0).astype(np.float32)
    else:
        positions = rng.uniform(0.0, 1.0, size=(n, dim)).astype(np.float32)
    weights = rng.uniform(0.5, 1.5, size=n).astype(np.float32)
    boxsize = 1.0 if periodic else None

    pc = PointCloud(
        positions=positions, weights=weights, boxsize=boxsize, verbose=False,
        backend="numpy",
    )
    positions_before = pc.positions.copy()
    weights_before = pc.weights.copy()

    pc.decompose()

    positions_unchanged = np.array_equal(pc.positions, positions_before)
    weights_unchanged = np.array_equal(pc.weights, weights_before)

    local_pos = pc.decomposition.local_positions
    local_w = pc.decomposition.local_weights
    local_idx = pc.decomposition.local_global_indices
    counts = pc.decomposition.counts

    local_shape_ok = (
        local_pos.shape[0] == counts[rank]
        and local_w.shape[0] == counts[rank]
        and local_idx.shape[0] == counts[rank]
    )

    all_pos = comm.gather(local_pos, root=0)
    all_w = comm.gather(local_w, root=0)
    all_idx = comm.gather(local_idx, root=0)
    all_shape_ok = comm.gather(local_shape_ok, root=0)
    all_unchanged = comm.gather((positions_unchanged, weights_unchanged), root=0)

    if rank == 0:
        concatenated_idx = np.concatenate(all_idx)
        bijection_ok = np.array_equal(np.sort(concatenated_idx), np.arange(n))

        reconstructed_pos = np.empty_like(positions_before)
        reconstructed_w = np.empty_like(weights_before)
        for p, w, idx in zip(all_pos, all_w, all_idx):
            if idx.shape[0] == 0:
                continue
            reconstructed_pos[idx] = p
            reconstructed_w[idx] = w

        result = {
            "n": n,
            "size": size,
            "counts": counts,
            "count_balance_ok": (counts.max() - counts.min()) <= 1,
            "count_sum_ok": counts.sum() == n,
            "bijection_ok": bijection_ok,
            "reconstruct_positions_exact": np.array_equal(
                reconstructed_pos, positions_before
            ),
            "reconstruct_weights_exact": np.array_equal(reconstructed_w, weights_before),
            "local_shapes_ok": all(all_shape_ok),
            "positions_unchanged": all(u[0] for u in all_unchanged),
            "weights_unchanged": all(u[1] for u in all_unchanged),
        }
        np.savez(out_path, **{k: np.asarray(v) for k, v in result.items()})
    print(f"RANK {rank} DONE")


def _run_pipeline_regression_under_mpi(out_path):
    """Same pipeline, with vs. without an inserted `.decompose()` call --
    results must be byte-identical, proving decompose() has zero effect on
    the existing (still full-replication-based) compute path."""
    from mpi4py import MPI

    from smudgy import PointCloud

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    rng = np.random.default_rng(1)
    n, dim = 41, 3
    positions = rng.uniform(0.0, 1.0, size=(n, dim)).astype(np.float32)
    weights = rng.uniform(0.5, 1.5, size=n).astype(np.float32)
    scalar_field = rng.uniform(size=n).astype(np.float32)

    def run_pipeline(call_decompose):
        pc = PointCloud(
            positions=positions.copy(), weights=weights.copy(), boxsize=1.0,
            verbose=False, backend="numpy",
        ).global_setup(kernel_name="cubic_spline", num_neighbors=8, structure="isotropic")
        if call_decompose:
            pc.decompose()
        pc.compute_smoothing()
        pc.compute_density()
        pc.add_fields("sf", scalar_field)
        density = pc.smoothing.density_isotropic.copy()
        interp = pc.interpolate("sf", structure="isotropic").copy()
        fgrid, wgrid = pc.deposit(
            "sf", averaged=True, gridnums=6, adaptive=False, kernel_name="cic",
            return_weights=True,
        )
        return density, interp, fgrid.copy(), wgrid.copy()

    density_a, interp_a, fgrid_a, wgrid_a = run_pipeline(call_decompose=False)
    density_b, interp_b, fgrid_b, wgrid_b = run_pipeline(call_decompose=True)

    if rank == 0:
        result = {
            "density_matches": np.array_equal(density_a, density_b),
            "interp_matches": np.array_equal(interp_a, interp_b),
            "fgrid_matches": np.array_equal(fgrid_a, fgrid_b),
            "wgrid_matches": np.array_equal(wgrid_a, wgrid_b),
        }
        np.savez(out_path, **{k: np.asarray(v) for k, v in result.items()})
    print(f"RANK {rank} DONE")


def _run_route_query_positions_under_mpi(out_path, n_particles, n_query, periodic):
    """Route an arbitrary (M, D) query array via `route_query_positions` and
    check the properties that make it a valid partition: bijection (every
    query point routed to exactly one rank, none dropped/duplicated),
    reconstruction via `local_global_indices` is exact, and -- the actual
    routing correctness property -- each routed point's own Hilbert code
    truly falls inside the *owning* rank's `[boundary_codes[r],
    boundary_codes[r+1])` interval (not just "some rank claimed it")."""
    from mpi4py import MPI

    from smudgy import PointCloud
    from smudgy.decomposition import hilbert_encode, route_query_positions

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    rng = np.random.default_rng(4)
    dim = 3
    positions = rng.uniform(0.0, 1.0, size=(n_particles, dim)).astype(np.float32)
    weights = np.ones(n_particles, dtype=np.float32)
    boxsize = 1.0 if periodic else None

    pc = PointCloud(
        positions=positions, weights=weights, boxsize=boxsize, verbose=False,
        backend="numpy",
    )
    pc.decompose()

    qrng = np.random.default_rng(5)
    # deliberately includes some out-of-[0,1] points to exercise the
    # non-periodic clip-routing edge case too
    lo, hi = (0.0, 1.0) if periodic else (-0.05, 1.05)
    query = qrng.uniform(lo, hi, size=(n_query, dim)).astype(np.float32)

    routing = route_query_positions(
        comm, pc.decomposition, query if rank == 0 else None, periodic
    )

    local_shape_ok = (
        routing.local_positions.shape[0] == routing.counts[rank]
        and routing.local_global_indices.shape[0] == routing.counts[rank]
    )

    # routing correctness: this rank's own routed points' Hilbert codes must
    # fall inside ITS OWN boundary_codes interval.
    routing_correct = True
    if routing.local_positions.shape[0] > 0:
        codes = hilbert_encode(
            routing.local_positions,
            pc.decomposition.domain_min,
            pc.decomposition.domain_max,
            periodic=periodic,
        )
        lo_code = pc.decomposition.boundary_codes[rank]
        hi_code = pc.decomposition.boundary_codes[rank + 1]
        routing_correct = bool(np.all((codes >= lo_code) & (codes < hi_code)))

    all_global = comm.gather(routing.local_global_indices, root=0)
    all_local = comm.gather(routing.local_positions, root=0)
    all_shape_ok = comm.gather(local_shape_ok, root=0)
    all_routing_correct = comm.gather(routing_correct, root=0)

    if rank == 0:
        concatenated = np.concatenate(all_global)
        bijection_ok = np.array_equal(np.sort(concatenated), np.arange(n_query))

        reconstructed = np.empty_like(query)
        for g, p in zip(all_global, all_local):
            if g.shape[0] == 0:
                continue
            reconstructed[g] = p
        reconstruct_exact = np.array_equal(reconstructed, query)

        result = {
            "bijection_ok": bijection_ok,
            "reconstruct_exact": reconstruct_exact,
            "local_shapes_ok": all(all_shape_ok),
            "routing_correct": all(all_routing_correct),
            "counts_sum_ok": routing.counts.sum() == n_query,
        }
        np.savez(out_path, **{k: np.asarray(v) for k, v in result.items()})
    print(f"RANK {rank} DONE")


def _run_under_mpi():
    mode = sys.argv[1]
    if mode == "props":
        out_path, n, dist_mode, periodic = sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5] == "1"
        _run_props_under_mpi(out_path, n, dist_mode, periodic)
    elif mode == "pipeline":
        _run_pipeline_regression_under_mpi(sys.argv[2])
    elif mode == "route_query":
        out_path, n_particles, n_query, periodic = (
            sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] == "1"
        )
        _run_route_query_positions_under_mpi(out_path, n_particles, n_query, periodic)
    else:
        raise ValueError(f"unknown mode {mode!r}")


def _mpiexec(n_ranks, args):
    result = subprocess.run(
        ["mpiexec", "-n", str(n_ranks), sys.executable, __file__, *args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    seen_ranks = {
        int(line.split()[1])
        for line in result.stdout.splitlines()
        if line.startswith("RANK")
    }
    assert seen_ranks == set(range(n_ranks)), (seen_ranks, result.stdout, result.stderr)


@pytest.mark.parametrize(
    "n_particles,n_ranks,dist_mode,periodic",
    [
        (37, 3, "uniform", False),  # N not divisible by P
        (37, 3, "uniform", True),  # same, periodic domain
        (100, 1, "uniform", False),  # trivial single-rank path
        (2, 5, "uniform", False),  # N < P: some ranks get zero particles
        (2000, 4, "anisotropic", False),  # highly clustered: the real motivator
    ],
)
def test_decomposition_properties(tmp_path, n_particles, n_ranks, dist_mode, periodic):
    out_path = tmp_path / "result.npz"
    _mpiexec(
        n_ranks,
        ["props", str(out_path), str(n_particles), dist_mode, "1" if periodic else "0"],
    )
    result = np.load(out_path)

    assert int(result["count_sum_ok"]) == 1
    assert int(result["bijection_ok"]) == 1
    assert int(result["count_balance_ok"]) == 1
    assert int(result["reconstruct_positions_exact"]) == 1
    assert int(result["reconstruct_weights_exact"]) == 1
    assert int(result["local_shapes_ok"]) == 1
    assert int(result["positions_unchanged"]) == 1
    assert int(result["weights_unchanged"]) == 1


def test_decompose_does_not_change_pipeline_results(tmp_path):
    out_path = tmp_path / "result.npz"
    _mpiexec(3, ["pipeline", str(out_path)])
    result = np.load(out_path)

    assert int(result["density_matches"]) == 1
    assert int(result["interp_matches"]) == 1
    assert int(result["fgrid_matches"]) == 1
    assert int(result["wgrid_matches"]) == 1


@pytest.mark.parametrize(
    "n_particles,n_query,n_ranks,periodic",
    [
        (300, 200, 3, False),  # includes deliberate out-of-domain (clip-routed) points
        (300, 200, 3, True),
        (100, 1, 1, False),  # trivial single-rank path
        (2000, 500, 4, False),
        (5, 3, 5, False),  # N < P AND M < P: some ranks get zero of both
    ],
)
def test_route_query_positions_properties(tmp_path, n_particles, n_query, n_ranks, periodic):
    out_path = tmp_path / "result.npz"
    _mpiexec(
        n_ranks,
        ["route_query", str(out_path), str(n_particles), str(n_query), "1" if periodic else "0"],
    )
    result = np.load(out_path)

    assert int(result["bijection_ok"]) == 1
    assert int(result["reconstruct_exact"]) == 1
    assert int(result["local_shapes_ok"]) == 1
    assert int(result["routing_correct"]) == 1
    assert int(result["counts_sum_ok"]) == 1


if __name__ == "__main__":
    _run_under_mpi()
