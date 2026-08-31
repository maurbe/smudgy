"""Multi-rank correctness tests for Step 5 of the domain-decomposition
roadmap: `PointCloud.gather_particles`/`gather_queries` and
`deposit(gather_to_root=True)` -- collecting a local-sized 4a/4b result (or
deposit's grid) back onto rank 0 only, in original order, instead of paying
to replicate it onto every rank.

The strongest check: run the full pipeline (decompose -> find_neighbors ->
compute_smoothing -> compute_density -> add_fields -> interpolate ->
deposit) via the local+ghost path, gather every result via the new Step 5
utilities, and compare against a reference `PointCloud` that never calls
`decompose()`/`find_neighbors()` at all (the old, full-replication path) --
mirroring `test_mpi_local_pipeline.py`/`test_mpi_interpolate_query_positions.py`'s
own old-vs-new comparison pattern, just using the new utilities to do the
reassembly instead of a hand-rolled `comm.gather` + loop. Also covers: `None`
on non-root ranks, negative/guard-rail tests, and the deadlock-safety edge
case (some ranks with zero local rows).

Run directly under MPI:
    mpiexec -n 3 python test_mpi_gather_to_root.py pipeline <out.npz> 1

Run via pytest (the test spawns mpiexec itself):
    pytest test_mpi_gather_to_root.py
"""

import subprocess
import sys

import numpy as np
import pytest


def _run_pipeline_under_mpi(out_path, periodic):
    from mpi4py import MPI

    from smudgy import PointCloud

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    periodic = periodic == "1"

    # Same seeds/sizes as Step 4b's own manually-verified dataset
    # (n=900 particles, k=10, 200 query points).
    rng = np.random.default_rng(61)
    n, dim = 900, 3
    positions = rng.uniform(0.0, 1.0, size=(n, dim)).astype(np.float32)
    weights = rng.uniform(0.5, 1.5, size=n).astype(np.float32)
    scalar_field = np.random.default_rng(62).uniform(size=n).astype(np.float32)
    boxsize = 1.0 if periodic else None
    deposit_extent = None if periodic else [[0.0, 1.0]] * dim

    qrng = np.random.default_rng(63)
    n_query = 200
    query = qrng.uniform(0.0, 1.0, size=(n_query, dim)).astype(np.float32)

    # reference: old (non-decomposed) full-replication path
    pc_ref = PointCloud(
        positions=positions.copy(), weights=weights.copy(), boxsize=boxsize,
        verbose=False, backend="taichi", arch="cpu",
    ).global_setup(kernel_name="cubic_spline", num_neighbors=10, structure="isotropic")
    pc_ref.compute_smoothing()
    pc_ref.compute_density()
    pc_ref.add_fields("sf", scalar_field)
    density_ref = pc_ref.smoothing.density_isotropic.copy()
    interp_particles_ref = pc_ref.interpolate("sf", structure="isotropic").copy()
    interp_query_ref = pc_ref.interpolate(
        "sf", query_positions=query.copy(), structure="isotropic"
    ).copy()
    fgrid_ref, wgrid_ref = pc_ref.deposit(
        "sf", averaged=True, gridnums=6, adaptive=True, structure="isotropic",
        return_weights=True, extent=deposit_extent,
    )

    pc = PointCloud(
        positions=positions.copy(), weights=weights.copy(), boxsize=boxsize,
        verbose=False, backend="taichi", arch="cpu",
    ).global_setup(kernel_name="cubic_spline", num_neighbors=10, structure="isotropic")
    pc.decompose()
    pc.find_neighbors()
    pc.compute_smoothing()
    pc.compute_density()
    pc.add_fields("sf", scalar_field)

    density_gathered = pc.gather_particles(pc.smoothing.density_isotropic)
    interp_particles_gathered = pc.gather_particles(
        pc.interpolate("sf", structure="isotropic")
    )
    interp_query_gathered = pc.gather_queries(
        pc.interpolate("sf", query_positions=query.copy(), structure="isotropic")
    )
    fgrid_root, wgrid_root = pc.deposit(
        "sf", averaged=True, gridnums=6, adaptive=True, structure="isotropic",
        return_weights=True, gather_to_root=True, extent=deposit_extent,
    )

    # Every rank must call the same collective the same number of times, so
    # the "gathered on root only, None elsewhere" check is done as a single
    # synchronized allreduce rather than branching on rank before deciding
    # whether to participate.
    is_root = rank == 0
    local_none_ness_ok = (
        (density_gathered is not None) == is_root
        and (interp_particles_gathered is not None) == is_root
        and (interp_query_gathered is not None) == is_root
        and (fgrid_root is not None) == is_root
        and (wgrid_root is not None) == is_root
    )
    none_ness_ok = comm.allreduce(local_none_ness_ok, op=MPI.LAND)

    if rank == 0:
        result = {
            "none_ness_ok": none_ness_ok,
            "density_matches": np.allclose(
                density_ref, density_gathered, rtol=1e-3, atol=1e-6
            ),
            "interp_particles_matches": np.allclose(
                interp_particles_ref, interp_particles_gathered, rtol=1e-3, atol=1e-6
            ),
            "interp_query_matches": np.allclose(
                interp_query_ref, interp_query_gathered, rtol=1e-3, atol=1e-6
            ),
            "fgrid_matches": np.allclose(fgrid_ref, fgrid_root, rtol=1e-3, atol=1e-5),
            "wgrid_matches": np.allclose(wgrid_ref, wgrid_root, rtol=1e-3, atol=1e-5),
        }
        np.savez(out_path, **{k: np.asarray(v) for k, v in result.items()})
    print(f"RANK {rank} DONE")


def _run_negative_tests_under_mpi(out_path):
    """gather_particles before decompose(); gather_queries before any 4b
    interpolate() call; a shape-mismatched local_array passed to
    gather_particles -- each must raise a clear error, synchronized so no
    rank hangs waiting on a peer that already raised."""
    from mpi4py import MPI

    from smudgy import PointCloud

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    rng = np.random.default_rng(74)
    n, dim = 50, 3
    positions = rng.uniform(0.0, 1.0, size=(n, dim)).astype(np.float32)
    weights = np.ones(n, dtype=np.float32)

    pc = PointCloud(
        positions=positions, weights=weights, boxsize=1.0, verbose=False,
        backend="numpy",
    )

    raised_without_decompose = False
    try:
        pc.gather_particles(np.zeros(5, dtype=np.float32))
    except AttributeError:
        raised_without_decompose = True

    raised_without_query_routing = False
    try:
        pc.gather_queries(np.zeros(5, dtype=np.float32))
    except AttributeError:
        raised_without_query_routing = True

    pc.decompose()
    raised_on_shape_mismatch = False
    try:
        wrong_shape = np.zeros(pc.decomposition.local_global_indices.shape[0] + 7, dtype=np.float32)
        pc.gather_particles(wrong_shape)
    except ValueError:
        raised_on_shape_mismatch = True

    all_ok = comm.allreduce(
        raised_without_decompose and raised_without_query_routing and raised_on_shape_mismatch,
        op=MPI.LAND,
    )
    if rank == 0:
        np.savez(out_path, all_ok=np.asarray(all_ok))
    print(f"RANK {rank} DONE")


def _run_zero_local_rows_under_mpi(out_path):
    """N < P (gather_particles) and a tiny query batch (gather_queries): some
    ranks have zero local rows to contribute. Must not deadlock, and the
    reassembled result on root must still be correct."""
    from mpi4py import MPI

    from smudgy import PointCloud

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    rng = np.random.default_rng(75)
    n, dim = 5, 3
    positions = rng.uniform(0.0, 1.0, size=(n, dim)).astype(np.float32)
    weights = np.ones(n, dtype=np.float32)
    scalar_field = np.arange(n, dtype=np.float32)
    query = np.array([[0.2, 0.3, 0.4], [0.6, 0.5, 0.5]], dtype=np.float32)

    # Non-periodic deliberately: this dataset is tiny enough (N=5, k=2) that
    # a periodic domain would legitimately trip the ghost-exchange half-box
    # guard (see test_mpi_ghosts.py's own dedicated guard test) -- that's a
    # different, already-covered property, not what this test is checking.
    pc_ref = PointCloud(
        positions=positions.copy(), weights=weights.copy(), boxsize=None,
        verbose=False, backend="taichi", arch="cpu",
    ).global_setup(kernel_name="cubic_spline", num_neighbors=2, structure="isotropic")
    pc_ref.compute_smoothing()
    pc_ref.compute_density()
    pc_ref.add_fields("sf", scalar_field)
    density_ref = pc_ref.smoothing.density_isotropic.copy()
    interp_query_ref = pc_ref.interpolate(
        "sf", query_positions=query.copy(), structure="isotropic"
    ).copy()

    pc = PointCloud(
        positions=positions.copy(), weights=weights.copy(), boxsize=None,
        verbose=False, backend="taichi", arch="cpu",
    ).global_setup(kernel_name="cubic_spline", num_neighbors=2, structure="isotropic")
    pc.decompose()
    pc.find_neighbors()
    pc.compute_smoothing()
    pc.compute_density()
    pc.add_fields("sf", scalar_field)

    density_gathered = pc.gather_particles(pc.smoothing.density_isotropic)
    interp_query_gathered = pc.gather_queries(
        pc.interpolate("sf", query_positions=query.copy(), structure="isotropic")
    )

    if rank == 0:
        result = {
            "density_matches": np.allclose(density_ref, density_gathered, rtol=1e-3, atol=1e-6),
            "interp_query_matches": np.allclose(
                interp_query_ref, interp_query_gathered, rtol=1e-3, atol=1e-6
            ),
        }
        np.savez(out_path, **{k: np.asarray(v) for k, v in result.items()})
    print(f"RANK {rank} DONE")


def _run_under_mpi():
    mode = sys.argv[1]
    if mode == "pipeline":
        _run_pipeline_under_mpi(sys.argv[2], sys.argv[3])
    elif mode == "negative":
        _run_negative_tests_under_mpi(sys.argv[2])
    elif mode == "zero_local_rows":
        _run_zero_local_rows_under_mpi(sys.argv[2])
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


@pytest.mark.parametrize("periodic,n_ranks", [(False, 1), (False, 3), (True, 3), (True, 4)])
def test_gather_to_root_matches_full_replication_reference(tmp_path, periodic, n_ranks):
    out_path = tmp_path / "result.npz"
    _mpiexec(n_ranks, ["pipeline", str(out_path), "1" if periodic else "0"])
    result = np.load(out_path)

    assert int(result["none_ness_ok"]) == 1
    assert int(result["density_matches"]) == 1
    assert int(result["interp_particles_matches"]) == 1
    assert int(result["interp_query_matches"]) == 1
    assert int(result["fgrid_matches"]) == 1
    assert int(result["wgrid_matches"]) == 1


def test_negative_cases_raise_clear_errors(tmp_path):
    out_path = tmp_path / "result.npz"
    _mpiexec(3, ["negative", str(out_path)])
    result = np.load(out_path)
    assert int(result["all_ok"]) == 1


def test_zero_local_rows_no_deadlock(tmp_path):
    out_path = tmp_path / "result.npz"
    _mpiexec(5, ["zero_local_rows", str(out_path)], timeout=60)
    result = np.load(out_path)
    assert int(result["density_matches"]) == 1
    assert int(result["interp_query_matches"]) == 1


if __name__ == "__main__":
    _run_under_mpi()
