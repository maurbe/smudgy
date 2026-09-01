"""Multi-rank correctness tests for `PointCloud.gather_particles`/
`gather_queries` and `deposit(gather_to_root=True)` -- collecting a
local-sized pipeline result (or deposit's grid) back onto rank 0 only, in
original order, instead of paying to replicate it onto every rank.

The strongest check, mirroring `test_mpi_local_pipeline.py`'s own pattern:
run the full pipeline (find_neighbors -> compute_smoothing ->
compute_density -> add_fields -> interpolate -> deposit) at a single rank
(reference) and again at several rank counts on identical input, gather
every result via the Step 5 utilities, and compare. Also covers: `None` on
non-root ranks, negative/guard-rail tests, and the deadlock-safety edge case
(some ranks with zero local rows).

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

    pc = PointCloud(
        positions=positions.copy(), weights=weights.copy(), boxsize=boxsize,
        verbose=False, backend="taichi", arch="cpu",
    ).global_setup(kernel_name="cubic_spline", num_neighbors=10, structure="isotropic")
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
        np.savez(
            out_path,
            none_ness_ok=np.asarray(none_ness_ok),
            density=density_gathered,
            interp_particles=interp_particles_gathered,
            interp_query=interp_query_gathered,
            fgrid=fgrid_root,
            wgrid=wgrid_root,
        )
    print(f"RANK {rank} DONE")


def _run_negative_tests_under_mpi(out_path):
    """gather_queries before any query-position interpolate() call, and a
    shape-mismatched local_array passed to gather_particles -- each must
    raise a clear error, synchronized so no rank hangs waiting on a peer
    that already raised. (There is no "before decompose()" case anymore --
    decomposition always exists by the time gather_particles could be
    called at all.)"""
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

    raised_without_query_routing = False
    try:
        pc.gather_queries(np.zeros(5, dtype=np.float32))
    except AttributeError:
        raised_without_query_routing = True

    raised_on_shape_mismatch = False
    try:
        wrong_shape = np.zeros(pc.decomposition.local_global_indices.shape[0] + 7, dtype=np.float32)
        pc.gather_particles(wrong_shape)
    except ValueError:
        raised_on_shape_mismatch = True

    all_ok = comm.allreduce(
        raised_without_query_routing and raised_on_shape_mismatch, op=MPI.LAND,
    )
    if rank == 0:
        np.savez(out_path, all_ok=np.asarray(all_ok))
    print(f"RANK {rank} DONE")


def _run_zero_local_rows_under_mpi(out_path):
    """N < P: some ranks have zero local rows to contribute. Must not
    deadlock, and the reassembled result on root must still be correct."""
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

    pc = PointCloud(
        positions=positions.copy(), weights=weights.copy(), boxsize=None,
        verbose=False, backend="taichi", arch="cpu",
    ).global_setup(kernel_name="cubic_spline", num_neighbors=2, structure="isotropic")
    pc.find_neighbors()
    pc.compute_smoothing()
    pc.compute_density()
    pc.add_fields("sf", scalar_field)

    density_gathered = pc.gather_particles(pc.smoothing.density_isotropic)
    interp_query_gathered = pc.gather_queries(
        pc.interpolate("sf", query_positions=query.copy(), structure="isotropic")
    )

    if rank == 0:
        np.savez(out_path, density=density_gathered, interp_query=interp_query_gathered)
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


@pytest.mark.parametrize("periodic,n_ranks", [(False, 3), (True, 3), (True, 4)])
def test_gather_to_root_matches_single_rank_reference(tmp_path, periodic, n_ranks):
    ref_path = tmp_path / "ref.npz"
    out_path = tmp_path / "result.npz"
    args_tail = ["1" if periodic else "0"]
    _mpiexec(1, ["pipeline", str(ref_path), *args_tail])
    _mpiexec(n_ranks, ["pipeline", str(out_path), *args_tail])

    ref = np.load(ref_path)
    result = np.load(out_path)

    assert int(result["none_ness_ok"]) == 1
    for key in ("density", "interp_particles", "interp_query", "fgrid", "wgrid"):
        atol = 1e-5 if key in ("fgrid", "wgrid") else 1e-6
        assert np.allclose(ref[key], result[key], rtol=1e-3, atol=atol), key


def test_negative_cases_raise_clear_errors(tmp_path):
    out_path = tmp_path / "result.npz"
    _mpiexec(3, ["negative", str(out_path)])
    result = np.load(out_path)
    assert int(result["all_ok"]) == 1


def test_zero_local_rows_no_deadlock(tmp_path):
    ref_path = tmp_path / "ref.npz"
    out_path = tmp_path / "result.npz"
    _mpiexec(1, ["zero_local_rows", str(ref_path)], timeout=60)
    _mpiexec(5, ["zero_local_rows", str(out_path)], timeout=60)

    ref = np.load(ref_path)
    result = np.load(out_path)
    assert np.allclose(ref["density"], result["density"], rtol=1e-3, atol=1e-6)
    assert np.allclose(ref["interp_query"], result["interp_query"], rtol=1e-3, atol=1e-6)


if __name__ == "__main__":
    _run_under_mpi()
