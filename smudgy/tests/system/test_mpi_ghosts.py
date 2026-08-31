"""Multi-rank correctness tests for `PointCloud.find_neighbors()` (Step 2 of
the domain-decomposition roadmap: ghost-particle exchange + iterative
true-KNN solver).

The strongest check: for every local particle, the converged local+ghost
K-NN result (translated from GhostInfo's local-combined-array convention to
global particle identity) must exactly match a single-rank, full-dataset
`cKDTree` reference query -- proving the iterative radius-growth/ghost-fetch
loop finds the TRUE global nearest neighbors, not just "whatever's nearby".
Includes a deliberately constructed periodic case (two tight clusters near
opposite box edges) verified to actually split across ranks and require
wraparound ghosts to answer correctly -- proving the periodic-image
bounding-box overlap logic isn't a no-op.

Run directly under MPI:
    mpiexec -n 3 python test_mpi_ghosts.py knn <out.npz> uniform_periodic 8

Run via pytest (the test spawns mpiexec itself):
    pytest test_mpi_ghosts.py
"""

import subprocess
import sys

import numpy as np
import pytest


def _make_dataset(mode, rng):
    dim = 3
    if mode in ("uniform_nonperiodic", "uniform_periodic"):
        n = 300
        positions = rng.uniform(0.0, 1.0, size=(n, dim)).astype(np.float32)
        boxsize = None if mode == "uniform_nonperiodic" else 1.0
    elif mode == "anisotropic":
        n_dense, n_sparse = 270, 30
        dense = rng.uniform(0.0, 0.01, size=(n_dense, dim))
        sparse = rng.uniform(0.0, 1.0, size=(n_sparse, dim))
        positions = np.concatenate([dense, sparse], axis=0).astype(np.float32)
        boxsize = None
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
        filler = rng.uniform(0.0, 1.0, size=(200, dim))
        positions = np.concatenate([a, b, filler], axis=0).astype(np.float32)
        boxsize = 1.0
    else:
        raise ValueError(mode)
    weights = np.ones(positions.shape[0], dtype=np.float32)
    return positions, weights, boxsize


def _run_knn_under_mpi(out_path, mode, k):
    from mpi4py import MPI
    from scipy.spatial import cKDTree

    from smudgy import PointCloud

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    rng = np.random.default_rng(7)
    positions, weights, boxsize = _make_dataset(mode, rng)

    pc = PointCloud(
        positions=positions, weights=weights, boxsize=boxsize, verbose=False,
        backend="numpy",
    )
    pc.decompose()
    pc.find_neighbors(num_neighbors=k)

    n_local = pc.ghosts.n_local
    local_global = pc.decomposition.local_global_indices
    ghost_global = pc.ghosts.ghost_global_index
    ghost_source = pc.ghosts.ghost_source_rank

    global_sets = []
    for row in pc.ghosts.nn_inds:
        s = set()
        for idx in row:
            s.add(int(local_global[idx]) if idx < n_local else int(ghost_global[idx - n_local]))
        global_sets.append(s)

    # reference: full-dataset tree (pc.positions is still the full replicated
    # array today -- decompose()/find_neighbors() don't touch it)
    ref_tree = cKDTree(pc.positions, boxsize=boxsize)
    ref_dists, ref_inds = ref_tree.query(pc.positions, k=k)
    if k == 1:
        ref_dists, ref_inds = ref_dists.reshape(-1, 1), ref_inds.reshape(-1, 1)

    all_match = True
    max_dist_err = 0.0
    cross_rank_wraparound_hits = 0
    for local_i, gid in enumerate(local_global.tolist()):
        mine = global_sets[local_i]
        ref = set(int(x) for x in ref_inds[gid])
        if mine != ref:
            all_match = False
        mine_d = np.sort(pc.ghosts.nn_dists[local_i])
        ref_d = np.sort(ref_dists[gid])
        max_dist_err = max(max_dist_err, float(np.max(np.abs(mine_d - ref_d))))
        for idx in pc.ghosts.nn_inds[local_i]:
            if idx >= n_local and int(ghost_source[idx - n_local]) != rank:
                cross_rank_wraparound_hits += 1

    n_local_shape_ok = pc.ghosts.nn_inds.shape == (n_local, k)

    result = {
        "all_match": all_match,
        "max_dist_err": max_dist_err,
        "n_local_shape_ok": n_local_shape_ok,
        "cross_rank_hits": cross_rank_wraparound_hits,
    }
    all_results = comm.gather(result, root=0)
    if rank == 0:
        agg = {
            "all_match": all(r["all_match"] for r in all_results),
            "max_dist_err": max(r["max_dist_err"] for r in all_results),
            "n_local_shape_ok": all(r["n_local_shape_ok"] for r in all_results),
            "total_cross_rank_hits": sum(r["cross_rank_hits"] for r in all_results),
        }
        np.savez(out_path, **{k2: np.asarray(v) for k2, v in agg.items()})
    print(f"RANK {rank} DONE")


def _run_pipeline_regression_under_mpi(out_path):
    """Since Step 4a, `find_neighbors()` is no longer a no-op for the
    pipeline -- compute_smoothing/compute_density/interpolate now actually
    take the local+ghost path when it's available, producing LOCAL-sized (not
    full-N) results. So this compares the OLD path's full-N results against
    the NEW path's results gathered + reassembled into original-particle
    order, with a numerical tolerance (not exact equality -- the two paths
    sum contributions in a different order, so float32 rounding differs at
    the ~1e-4 level, same as any reordered floating-point summation).
    `deposit`'s grids need no such reassembly: its `allreduce`-based grid-sum
    is already global regardless of path.
    """
    from mpi4py import MPI

    from smudgy import PointCloud

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    rng = np.random.default_rng(3)
    # N=53 (an earlier choice) was too sparse for k=8 neighbors in a
    # periodic unit cube split across 3 ranks -- it legitimately tripped
    # the half-box safety guard (needed radius ~0.75 > boxsize/2). 300
    # keeps this fast while staying well clear of that.
    n, dim = 300, 3
    positions = rng.uniform(0.0, 1.0, size=(n, dim)).astype(np.float32)
    weights = rng.uniform(0.5, 1.5, size=n).astype(np.float32)
    scalar_field = rng.uniform(size=n).astype(np.float32)

    def run_pipeline(call_find_neighbors):
        pc = PointCloud(
            positions=positions.copy(), weights=weights.copy(), boxsize=1.0,
            verbose=False, backend="numpy",
        ).global_setup(kernel_name="cubic_spline", num_neighbors=8, structure="isotropic")
        if call_find_neighbors:
            pc.decompose()
            pc.find_neighbors()
        pc.compute_smoothing()
        pc.compute_density()
        pc.add_fields("sf", scalar_field)
        density = pc.smoothing.density_isotropic.copy()
        interp = pc.interpolate("sf", structure="isotropic").copy()
        fgrid, wgrid = pc.deposit(
            "sf", averaged=True, gridnums=6, adaptive=False, kernel_name="cic",
            return_weights=True,
        )

        if call_find_neighbors:
            # local-sized -> gather + reassemble into original particle order
            local_global = pc.decomposition.local_global_indices
            all_global = comm.gather(local_global, root=0)
            all_density = comm.gather(density, root=0)
            all_interp = comm.gather(interp, root=0)
            if rank == 0:
                density_full = np.empty(n, dtype=density.dtype)
                interp_full = np.empty((n, interp.shape[1]), dtype=interp.dtype)
                for g, d, itp in zip(all_global, all_density, all_interp):
                    density_full[g] = d
                    interp_full[g] = itp
                density, interp = density_full, interp_full

        return density, interp, fgrid.copy(), wgrid.copy()

    density_a, interp_a, fgrid_a, wgrid_a = run_pipeline(call_find_neighbors=False)
    density_b, interp_b, fgrid_b, wgrid_b = run_pipeline(call_find_neighbors=True)

    if rank == 0:
        result = {
            "density_matches": np.allclose(density_a, density_b, rtol=1e-3, atol=1e-6),
            "interp_matches": np.allclose(interp_a, interp_b, rtol=1e-3, atol=1e-6),
            "fgrid_matches": np.allclose(fgrid_a, fgrid_b, rtol=1e-3, atol=1e-6),
            "wgrid_matches": np.allclose(wgrid_a, wgrid_b, rtol=1e-3, atol=1e-6),
        }
        np.savez(out_path, **{k: np.asarray(v) for k, v in result.items()})
    print(f"RANK {rank} DONE")


def _run_deadlock_check_under_mpi(out_path):
    """N < P: some ranks get zero local particles. Must not deadlock, and
    empty ranks must produce correctly-shaped (0, k) results. Non-periodic
    deliberately, to isolate this from the half-box guard (see
    `_run_half_box_guard_under_mpi` for that, separately)."""
    from mpi4py import MPI

    from smudgy import PointCloud

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    rng = np.random.default_rng(11)
    n, dim, k = 2, 3, 2
    positions = rng.uniform(0.0, 1.0, size=(n, dim)).astype(np.float32)
    weights = np.ones(n, dtype=np.float32)

    pc = PointCloud(positions=positions, weights=weights, boxsize=None, verbose=False, backend="numpy")
    pc.decompose()
    pc.find_neighbors(num_neighbors=k)

    shape_ok = pc.ghosts.nn_inds.shape == (pc.ghosts.n_local, k)
    all_shape_ok = comm.allreduce(shape_ok, op=MPI.LAND)

    if rank == 0:
        np.savez(out_path, all_shape_ok=np.asarray(all_shape_ok))
    print(f"RANK {rank} DONE")


def _run_half_box_guard_under_mpi():
    """Deliberately trigger the periodic half-box safety guard (very few
    particles, k close to N, in a periodic domain -- the radius needed to
    find k neighbors among so few particles legitimately approaches/exceeds
    half the box). Every rank must raise ValueError -- if the raise
    decision weren't synchronized across ranks, a rank that doesn't trip
    the guard would hang forever waiting on a peer that already exited, so
    this test (run under a subprocess timeout by the pytest driver) also
    guards against that deadlock regressing."""
    from mpi4py import MPI

    from smudgy import PointCloud

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    rng = np.random.default_rng(11)
    n, dim, k = 2, 3, 2
    positions = rng.uniform(0.0, 1.0, size=(n, dim)).astype(np.float32)
    weights = np.ones(n, dtype=np.float32)

    pc = PointCloud(positions=positions, weights=weights, boxsize=1.0, verbose=False, backend="numpy")
    pc.decompose()
    try:
        pc.find_neighbors(num_neighbors=k)
        raised = False
    except ValueError:
        raised = True
    all_raised = comm.allreduce(raised, op=MPI.LAND)
    if rank == 0 and not all_raised:
        sys.exit(1)
    print(f"RANK {rank} DONE")


def _run_negative_test_under_mpi():
    from mpi4py import MPI

    from smudgy import PointCloud

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    rng = np.random.default_rng(13)
    n, dim = 5, 3
    positions = rng.uniform(0.0, 1.0, size=(n, dim)).astype(np.float32)
    weights = np.ones(n, dtype=np.float32)

    pc = PointCloud(positions=positions, weights=weights, boxsize=1.0, verbose=False, backend="numpy")
    pc.decompose()
    try:
        pc.find_neighbors(num_neighbors=n + 5)
        raised = False
    except ValueError:
        raised = True
    all_raised = comm.allreduce(raised, op=MPI.LAND)
    if rank == 0 and not all_raised:
        sys.exit(1)
    print(f"RANK {rank} DONE")


def _run_push_to_ghosts_under_mpi(out_path):
    """Isolation test for Step 4a's `push_to_ghosts`: a value pushed from
    rank A's local index i must arrive as rank B's ghost value at exactly
    the row where ghost_source_rank == A and ghost_source_local_index == i
    -- i.e. it must correctly reuse exchange_ghosts's routing, not just
    produce a plausible-shaped result."""
    from mpi4py import MPI

    from smudgy import PointCloud
    from smudgy.ghosts import push_to_ghosts

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    rng = np.random.default_rng(21)
    n, dim = 400, 3
    positions = rng.uniform(0.0, 1.0, size=(n, dim)).astype(np.float32)
    weights = np.ones(n, dtype=np.float32)

    pc = PointCloud(positions=positions, weights=weights, boxsize=1.0, verbose=False, backend="numpy")
    pc.decompose()
    pc.find_neighbors(num_neighbors=8)

    # a synthetic local array that unambiguously encodes both which particle
    # and which rank sent it, so a routing mistake shows up as a value
    # mismatch rather than accidentally looking right
    local_global = pc.decomposition.local_global_indices
    local_array = local_global.astype(np.float64) * 1000.0 + rank

    ghost_values = push_to_ghosts(comm, pc.ghosts, local_array)
    expected = (
        pc.ghosts.ghost_global_index.astype(np.float64) * 1000.0
        + pc.ghosts.ghost_source_rank
    )
    values_match = bool(np.allclose(ghost_values, expected))
    shape_ok = ghost_values.shape[0] == pc.ghosts.ghost_positions.shape[0]

    all_ok = comm.allreduce(values_match and shape_ok, op=MPI.LAND)
    if rank == 0:
        np.savez(out_path, all_ok=np.asarray(all_ok))
    print(f"RANK {rank} DONE")


def _run_target_positions_under_mpi(out_path, mode):
    """Step 4b: `exchange_ghosts(..., target_positions=...)` generalization.

    Two checks in one run: (1) `target_positions=decomposition.local_positions`
    reproduces `target_positions=None` exactly -- the generalization is
    behavior-preserving in the default-equivalent case, not just "close
    enough". (2) `target_positions=<a separate synthetic point set>` (a
    random grid, unrelated to any particle) converges to the TRUE global K-NN
    at those points, checked against a brute-force single-tree reference --
    proving the radius-growth/ghost-fetch machinery works correctly when the
    thing needing answers isn't this rank's own particles."""
    from mpi4py import MPI
    from scipy.spatial import cKDTree

    from smudgy import PointCloud
    from smudgy.ghosts import exchange_ghosts

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    rng = np.random.default_rng(31)
    n, dim, k = 400, 3, 8
    positions = rng.uniform(0.0, 1.0, size=(n, dim)).astype(np.float32)
    weights = rng.uniform(0.5, 1.5, size=n).astype(np.float32)
    boxsize = 1.0 if mode == "periodic" else None
    periodic = mode == "periodic"

    pc = PointCloud(
        positions=positions, weights=weights, boxsize=boxsize, verbose=False,
        backend="numpy",
    )
    pc.decompose()

    default_ghosts = exchange_ghosts(
        comm, pc.decomposition, k, dim, periodic, boxsize,
    )
    explicit_self_ghosts = exchange_ghosts(
        comm, pc.decomposition, k, dim, periodic, boxsize,
        target_positions=pc.decomposition.local_positions,
    )
    equivalence_ok = (
        np.array_equal(default_ghosts.nn_inds, explicit_self_ghosts.nn_inds)
        and np.array_equal(default_ghosts.nn_dists, explicit_self_ghosts.nn_dists)
    )

    # a separate, unrelated query-point set (not particles at all) -- a
    # different draw per rank (via the rank-offset seed), all still owned/
    # searched locally -- this test isn't exercising cross-rank query
    # routing (that's Step 4b's PointCloud-level test), just
    # exchange_ghosts's target_positions mechanics in isolation. Deliberately
    # confined to a small sub-cube (not a full-domain-spanning sample): a
    # spatially compact query batch is what real Hilbert-routed query points
    # actually look like (the whole reason 4b routes by Hilbert code rather
    # than arbitrarily); a batch scattered across the whole periodic domain
    # would legitimately need a huge padding radius and isn't representative.
    qrng = np.random.default_rng(32 + rank)
    n_query_local = 15
    query = qrng.uniform(0.3, 0.4, size=(n_query_local, dim)).astype(np.float32)

    query_ghosts = exchange_ghosts(
        comm, pc.decomposition, k, dim, periodic, boxsize, target_positions=query,
    )
    shape_ok = query_ghosts.nn_inds.shape == (n_query_local, k)

    n_local = query_ghosts.n_local
    local_global = pc.decomposition.local_global_indices
    ghost_global = query_ghosts.ghost_global_index

    ref_tree = cKDTree(positions, boxsize=boxsize)
    ref_dists, ref_inds = ref_tree.query(query, k=k)

    all_match = True
    max_dist_err = 0.0
    for i in range(n_query_local):
        mine = set()
        for idx in query_ghosts.nn_inds[i]:
            mine.add(
                int(local_global[idx]) if idx < n_local else int(ghost_global[idx - n_local])
            )
        ref = set(int(x) for x in ref_inds[i])
        if mine != ref:
            all_match = False
        max_dist_err = max(
            max_dist_err,
            float(np.max(np.abs(np.sort(query_ghosts.nn_dists[i]) - np.sort(ref_dists[i])))),
        )

    result = {
        "equivalence_ok": equivalence_ok,
        "shape_ok": shape_ok,
        "all_match": all_match,
        "max_dist_err": max_dist_err,
    }
    all_results = comm.gather(result, root=0)
    if rank == 0:
        agg = {
            "equivalence_ok": all(r["equivalence_ok"] for r in all_results),
            "shape_ok": all(r["shape_ok"] for r in all_results),
            "all_match": all(r["all_match"] for r in all_results),
            "max_dist_err": max(r["max_dist_err"] for r in all_results),
        }
        np.savez(out_path, **{k: np.asarray(v) for k, v in agg.items()})
    print(f"RANK {rank} DONE")


def _run_sparse_target_density_under_mpi(out_path):
    """Regression test for a real bug found while building Step 4b/5: an
    earlier version of `exchange_ghosts`'s `target_positions` generalization
    estimated the INITIAL radius from `target_positions`'s own density (its
    count over its own bounding box) rather than the local particles' -- for
    a `target_positions` batch spread over roughly the same region as the
    local particles but with far fewer points (exactly what routed
    interpolation query positions look like relative to particles: e.g. 200
    query points vs 900 particles), that formula's initial guess came out
    ~2x too large, occasionally exceeding the periodic half-box guard on the
    very first iteration -- before any real ghost-fetching had happened at
    all. Fixed by estimating the initial radius from the local PARTICLE
    density unconditionally (the radius needed to reach `num_neighbors`
    particles depends on how densely packed particles are, not on how many
    target points happen to be nearby). This test reproduces that exact
    shape (many particles, few target points, similar bounding box,
    periodic) and confirms it converges without raising."""
    from mpi4py import MPI
    from scipy.spatial import cKDTree

    from smudgy import PointCloud
    from smudgy.ghosts import exchange_ghosts

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    rng = np.random.default_rng(41)
    n, dim, k = 700, 3, 10
    positions = rng.uniform(0.0, 1.0, size=(n, dim)).astype(np.float32)
    weights = rng.uniform(0.5, 1.5, size=n).astype(np.float32)
    boxsize = 1.0

    pc = PointCloud(
        positions=positions, weights=weights, boxsize=boxsize, verbose=False,
        backend="numpy",
    )
    pc.decompose()

    # ~10x fewer target points than particles, spread over the SAME
    # near-full-domain extent as the particles (not confined to a small
    # sub-cube) -- the specific shape that tripped the bug.
    qrng = np.random.default_rng(42 + rank)
    n_query_local = max(1, pc.decomposition.local_positions.shape[0] // 10)
    query = qrng.uniform(0.0, 1.0, size=(n_query_local, dim)).astype(np.float32)

    raised = False
    query_ghosts = None
    try:
        query_ghosts = exchange_ghosts(
            comm, pc.decomposition, k, dim, True, boxsize, target_positions=query,
        )
    except ValueError:
        raised = True
    no_raise_ok = comm.allreduce(not raised, op=MPI.LAND)

    all_match = True
    if not raised:
        n_local = query_ghosts.n_local
        local_global = pc.decomposition.local_global_indices
        ghost_global = query_ghosts.ghost_global_index
        ref_tree = cKDTree(positions, boxsize=boxsize)
        ref_dists, ref_inds = ref_tree.query(query, k=k)
        for i in range(n_query_local):
            mine = set()
            for idx in query_ghosts.nn_inds[i]:
                mine.add(
                    int(local_global[idx]) if idx < n_local else int(ghost_global[idx - n_local])
                )
            ref = set(int(x) for x in ref_inds[i])
            if mine != ref:
                all_match = False
    all_match_ok = comm.allreduce(all_match, op=MPI.LAND)

    if rank == 0:
        np.savez(
            out_path,
            no_raise_ok=np.asarray(no_raise_ok),
            all_match_ok=np.asarray(all_match_ok),
        )
    print(f"RANK {rank} DONE")


def _run_under_mpi():
    mode = sys.argv[1]
    if mode == "target_positions":
        _run_target_positions_under_mpi(sys.argv[2], sys.argv[3])
    elif mode == "sparse_target_density":
        _run_sparse_target_density_under_mpi(sys.argv[2])
    elif mode == "knn":
        _run_knn_under_mpi(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif mode == "pipeline":
        _run_pipeline_regression_under_mpi(sys.argv[2])
    elif mode == "deadlock":
        _run_deadlock_check_under_mpi(sys.argv[2])
    elif mode == "negative":
        _run_negative_test_under_mpi()
    elif mode == "push_to_ghosts":
        _run_push_to_ghosts_under_mpi(sys.argv[2])
    elif mode == "half_box_guard":
        _run_half_box_guard_under_mpi()
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
            f"mpiexec -n {n_ranks} {' '.join(args)} did not finish within "
            f"{timeout}s -- likely a deadlock (one or more ranks raised/exited "
            "while others were still waiting on a collective call)."
        ) from exc
    assert result.returncode == 0, result.stderr
    seen_ranks = {
        int(line.split()[1])
        for line in result.stdout.splitlines()
        if line.startswith("RANK")
    }
    assert seen_ranks == set(range(n_ranks)), (seen_ranks, result.stdout, result.stderr)


@pytest.mark.parametrize(
    "mode,n_ranks,k",
    [
        ("uniform_nonperiodic", 1, 8),
        ("uniform_nonperiodic", 3, 8),
        ("uniform_periodic", 1, 8),
        ("uniform_periodic", 3, 8),
        ("anisotropic", 4, 8),
        ("wraparound", 3, 8),
    ],
)
def test_knn_matches_full_dataset_reference(tmp_path, mode, n_ranks, k):
    out_path = tmp_path / "result.npz"
    _mpiexec(n_ranks, ["knn", str(out_path), mode, str(k)])
    result = np.load(out_path)

    assert int(result["all_match"]) == 1
    assert int(result["n_local_shape_ok"]) == 1
    assert float(result["max_dist_err"]) < 1e-4


def test_wraparound_case_actually_uses_cross_rank_ghosts(tmp_path):
    """Guards against the wraparound scenario passing vacuously (e.g. if the
    two edge clusters happened to land on the same rank, or if convergence
    were reached without ever using a periodic-shifted ghost) -- confirmed
    at implementation time this seed splits the clusters across ranks."""
    out_path = tmp_path / "result.npz"
    _mpiexec(3, ["knn", str(out_path), "wraparound", "8"])
    result = np.load(out_path)
    assert int(result["total_cross_rank_hits"]) > 0


def test_deadlock_free_with_empty_ranks(tmp_path):
    out_path = tmp_path / "result.npz"
    _mpiexec(5, ["deadlock", str(out_path)])
    result = np.load(out_path)
    assert int(result["all_shape_ok"]) == 1


def test_num_neighbors_exceeding_total_raises():
    _mpiexec(3, ["negative"])


def test_half_box_guard_raises_on_every_rank_without_deadlock():
    # timeout is the actual assertion here: if the raise decision weren't
    # synchronized (see the fix in ghosts.py), a non-raising rank would hang
    # forever waiting on a peer that already exited, and this would time out.
    _mpiexec(5, ["half_box_guard"], timeout=30)


def test_find_neighbors_gives_equivalent_pipeline_results(tmp_path):
    """Since Step 4a, find_neighbors() changes WHICH code path the pipeline
    takes (local+ghost vs. full-replication) -- this checks the two paths
    still agree numerically, not that find_neighbors() is a no-op (it isn't,
    by design; see `test_mpi_local_pipeline.py` for the fuller version of
    this same check across structures/periodicity/rank counts)."""
    out_path = tmp_path / "result.npz"
    _mpiexec(3, ["pipeline", str(out_path)])
    result = np.load(out_path)

    assert int(result["density_matches"]) == 1
    assert int(result["interp_matches"]) == 1
    assert int(result["fgrid_matches"]) == 1
    assert int(result["wgrid_matches"]) == 1


def test_push_to_ghosts_routes_values_correctly(tmp_path):
    out_path = tmp_path / "result.npz"
    _mpiexec(4, ["push_to_ghosts", str(out_path)])
    result = np.load(out_path)
    assert int(result["all_ok"]) == 1


@pytest.mark.parametrize("mode", ["nonperiodic", "periodic"])
def test_exchange_ghosts_target_positions_generalization(tmp_path, mode):
    out_path = tmp_path / "result.npz"
    _mpiexec(3, ["target_positions", str(out_path), mode])
    result = np.load(out_path)

    assert int(result["equivalence_ok"]) == 1
    assert int(result["shape_ok"]) == 1
    assert int(result["all_match"]) == 1
    assert float(result["max_dist_err"]) < 1e-4


def test_sparse_target_density_does_not_inflate_initial_radius(tmp_path):
    out_path = tmp_path / "result.npz"
    _mpiexec(3, ["sparse_target_density", str(out_path)])
    result = np.load(out_path)

    assert int(result["no_raise_ok"]) == 1
    assert int(result["all_match_ok"]) == 1


if __name__ == "__main__":
    _run_under_mpi()
