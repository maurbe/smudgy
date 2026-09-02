"""Multi-rank correctness tests for `PointCloud.get` (the name-based,
results-out-of-the-pipeline counterpart to `add_fields`, wrapping
`gather_particles`) and for `PointCloud.__setattr__`'s field-assignment
warning (fires only at `size > 1`, for an ndarray whose length matches the
global particle count -- see `pointcloud.py`'s `__setattr__` docstring).

The `size == 1` side of both (`get()` on a single rank, and confirming the
warning does NOT fire at `size == 1`, since local count == global count
there) is covered by `tests/unit/test_pointcloud.py`, which runs
single-process; this file covers the genuinely multi-rank behaviors that
need a real communicator with `size > 1`.

Run directly under MPI:
    mpiexec -n 3 python test_mpi_get_and_setattr_guard.py get <out.npz>
    mpiexec -n 3 python test_mpi_get_and_setattr_guard.py setattr_warns <out.npz>

Run via pytest (the test spawns mpiexec itself):
    pytest test_mpi_get_and_setattr_guard.py
"""

import subprocess
import sys
import warnings

import numpy as np
import pytest


def _run_get_under_mpi(out_path):
    from mpi4py import MPI

    from smudgy import PointCloud

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    rng = np.random.default_rng(71)
    n, dim = 400, 3
    positions = rng.uniform(0.0, 1.0, size=(n, dim)).astype(np.float32)
    weights = rng.uniform(0.5, 1.5, size=n).astype(np.float32)
    scalar_field = np.random.default_rng(72).uniform(size=n).astype(np.float32)

    pc = PointCloud(
        positions=positions.copy(), weights=weights.copy(), boxsize=None,
        verbose=False, backend="numpy",
    ).global_setup(kernel_name="cubic_spline", num_neighbors=8, structure="isotropic")
    pc.find_neighbors()
    pc.compute_smoothing()
    pc.compute_density()
    pc.add_fields("sf", scalar_field)

    # `get()` must agree with the equivalent explicit gather_particles call,
    # for both a smoothing.* field and a custom add_fields field.
    density_via_get = pc.get("density_isotropic")
    density_via_gather = pc.gather_particles(pc.smoothing.density_isotropic)
    sf_via_get = pc.get("sf")
    sf_via_gather = pc.gather_particles(pc.sf)
    positions_via_get = pc.get("positions")

    is_root = rank == 0
    local_ok = (
        ((density_via_get is not None) == is_root)
        and ((sf_via_get is not None) == is_root)
        and ((positions_via_get is not None) == is_root)
        and (
            not is_root
            or (
                np.array_equal(density_via_get, density_via_gather)
                and np.array_equal(sf_via_get, sf_via_gather)
                and np.array_equal(positions_via_get, positions)
            )
        )
    )

    # unknown-name error path
    unknown_raised = False
    try:
        pc.get("does_not_exist")
    except AttributeError:
        unknown_raised = True

    # not-yet-computed error path: a fresh PointCloud, density never computed
    pc2 = PointCloud(
        positions=positions.copy(), weights=weights.copy(), boxsize=None,
        verbose=False, backend="numpy",
    )
    not_computed_raised = False
    try:
        pc2.get("density_isotropic")
    except AttributeError:
        not_computed_raised = True

    ok = comm.allreduce(
        local_ok and unknown_raised and not_computed_raised, op=MPI.LAND
    )

    if rank == 0:
        np.savez(out_path, ok=np.asarray(ok))
    print(f"RANK {rank} DONE")


def _run_setattr_warns_under_mpi(out_path):
    from mpi4py import MPI

    from smudgy import PointCloud

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    rng = np.random.default_rng(73)
    n, dim = 40, 3
    positions = rng.uniform(0.0, 1.0, size=(n, dim)).astype(np.float32)

    pc = PointCloud(positions=positions.copy(), verbose=False, backend="numpy")

    # Case 1: a full-global-length ndarray -- the likely-a-mistake case this
    # guard exists for -- must warn (size > 1 is guaranteed by this test's
    # n_ranks, see the pytest wrapper below).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pc.new_field = np.ones(n, dtype=np.float32)
    full_length_warns = any(
        issubclass(w.category, UserWarning) and "add_fields" in str(w.message)
        for w in caught
    )
    # the assignment itself must still have gone through, unblocked
    full_length_set_ok = np.array_equal(pc.new_field, np.ones(n, dtype=np.float32))

    # Case 2: an ordinary per-rank scalar/metadata assignment must never warn.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pc.step = 3
        pc.label = "run1"
    no_warn_on_metadata = len(caught) == 0

    # Case 3: a legitimately local-sized array (this rank's own local particle
    # count) must never warn -- mirrors what `add_fields`'s own internal
    # `setattr` does, and is generically != n (global) at size > 1.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pc.local_sized_array = np.ones(pc.positions.shape[0], dtype=np.float32)
    no_warn_on_local_sized = len(caught) == 0 or pc.positions.shape[0] == n

    ok = comm.allreduce(
        full_length_warns
        and full_length_set_ok
        and no_warn_on_metadata
        and no_warn_on_local_sized,
        op=MPI.LAND,
    )

    if rank == 0:
        np.savez(out_path, ok=np.asarray(ok))
    print(f"RANK {rank} DONE")


def _run_under_mpi():
    mode = sys.argv[1]
    out_path = sys.argv[2]
    if mode == "get":
        _run_get_under_mpi(out_path)
    elif mode == "setattr_warns":
        _run_setattr_warns_under_mpi(out_path)
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


def test_get_matches_gather_particles_and_error_paths(tmp_path):
    out_path = tmp_path / "result.npz"
    _mpiexec(3, ["get", str(out_path)])
    result = np.load(out_path)
    assert int(result["ok"]) == 1


def test_setattr_warns_only_for_global_length_array(tmp_path):
    out_path = tmp_path / "result.npz"
    _mpiexec(3, ["setattr_warns", str(out_path)])
    result = np.load(out_path)
    assert int(result["ok"]) == 1


if __name__ == "__main__":
    _run_under_mpi()
