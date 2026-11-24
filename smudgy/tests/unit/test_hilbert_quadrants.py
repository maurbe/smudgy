"""The Quadrants Hilbert keys and sort reproduce the numpy reference exactly.

The decomposition must not depend on where the keys are computed: the same
input gives the same keys, the same (stable) order and so the same partition.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from smudgy import decomposition
from smudgy.backend.quadrants import hilbert


def _points(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.random((n, dim), dtype=np.float32)
    pos[: n // 10] = pos[n // 10 : 2 * (n // 10)]  # duplicates: equal keys
    # corners and edges of the domain
    pos[-4:] = np.array([[0.0] * dim, [1.0] * dim, [0.5] * dim, [np.nextafter(1, 0)] * dim])
    return pos


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("periodic", [True, False])
def test_keys_match_numpy(dim, periodic):
    pos = _points(20_000, dim)
    lo, hi = np.zeros(dim, np.float32), np.ones(dim, np.float32)
    ref = decomposition.hilbert_encode(pos, lo, hi, periodic, backend="numpy")
    got = decomposition.hilbert_encode(pos, lo, hi, periodic, backend="quadrants")
    assert got.dtype == np.uint64
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("dim,bits", [(2, 5), (3, 4), (3, 1), (2, 32)])
def test_keys_match_numpy_all_coordinates(dim, bits):
    """Every coordinate tuple of a small grid (and the 2D 64-bit key layout)."""
    if bits <= 5:
        axes = [np.arange(2**bits, dtype=np.uint64)] * dim
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dim)
    else:
        rng = np.random.default_rng(1)
        grid = rng.integers(0, 2**bits, size=(5000, dim), dtype=np.uint64)
        grid[0] = 2**bits - 1
    transpose = decomposition._axes_to_hilbert_transpose(list(grid.T), bits)
    ref = decomposition._transpose_to_index(transpose, bits)
    got = hilbert.hilbert_keys(grid.astype(np.uint32), bits)
    np.testing.assert_array_equal(got, ref)
    if bits <= 5:  # a Hilbert curve visits every cell exactly once
        assert np.unique(got).size == got.size == 2 ** (dim * bits)


def test_keys_chunked(monkeypatch):
    monkeypatch.setattr(hilbert, "_ENCODE_CHUNK", 1000)
    pos = _points(4321, 3)
    lo, hi = np.zeros(3, np.float32), np.ones(3, np.float32)
    ref = decomposition.hilbert_encode(pos, lo, hi, True, backend="numpy")
    got = decomposition.hilbert_encode(pos, lo, hi, True, backend="quadrants")
    np.testing.assert_array_equal(got, ref)


def test_argsort_matches_stable_numpy():
    codes = decomposition.hilbert_encode(
        _points(10_000, 3), np.zeros(3), np.ones(3), True, backend="numpy"
    )
    np.testing.assert_array_equal(
        decomposition.hilbert_argsort(codes, backend="quadrants"),
        np.argsort(codes, kind="stable"),
    )


def test_backend_auto_and_invalid():
    # the test session runs a Quadrants runtime (conftest), so None -> quadrants
    assert decomposition._use_quadrants(None)
    assert not decomposition._use_quadrants("numpy")
    with pytest.raises(ValueError):
        decomposition._use_quadrants("taichi")


def test_partition_identical_across_backends():
    """Single-process stand-in for the root's work: same order and boundaries."""
    pos = _points(50_000, 3)
    lo, hi = np.zeros(3, np.float32), np.ones(3, np.float32)
    counts = np.array([12_500] * 4, dtype=np.int64)
    out = {}
    for backend in ("numpy", "quadrants"):
        codes = decomposition.hilbert_encode(pos, lo, hi, True, backend=backend)
        order = decomposition.hilbert_argsort(codes, backend=backend)
        out[backend] = (
            order,
            decomposition._partition_boundary_codes(codes[order], counts),
        )
    np.testing.assert_array_equal(out["numpy"][0], out["quadrants"][0])
    np.testing.assert_array_equal(out["numpy"][1], out["quadrants"][1])


_GPU_SORT_SCRIPT = textwrap.dedent(
    """
    import numpy as np, quadrants as qd
    from quadrants.lang import impl
    from smudgy.backend.quadrants import init
    init(arch="gpu")
    if impl.current_cfg().arch in (qd.x64, qd.arm64):
        print("NO_GPU"); raise SystemExit
    from smudgy.backend.quadrants import hilbert
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 2**63, 300_000, dtype=np.uint64)
    keys[:5000] = keys[5000:10000]
    order = hilbert.argsort_keys(keys)
    assert hilbert._sort_available.get(impl.current_cfg().arch) is True
    assert np.array_equal(order, np.argsort(keys, kind="stable"))
    print("OK")
    """
)


def test_radix_sort_on_gpu():
    """The radix sort itself only runs on GPU archs: test it in a fresh process."""
    res = subprocess.run(
        [sys.executable, "-c", _GPU_SORT_SCRIPT], capture_output=True, text=True
    )
    if "NO_GPU" in res.stdout:
        pytest.skip("no GPU arch available")
    assert res.returncode == 0 and "OK" in res.stdout, res.stdout + res.stderr


@pytest.mark.parametrize("dup_fraction", [0.0, 0.3, 1.0])
def test_stable_argsort_matches_numpy_stable(dup_fraction):
    rng = np.random.default_rng(2)
    codes = rng.integers(0, 2**63, 20_000, dtype=np.uint64)
    n_dup = int(dup_fraction * codes.size)
    codes[:n_dup] = rng.choice(codes[:50], n_dup)  # long runs of equal keys
    np.testing.assert_array_equal(
        decomposition.stable_argsort(codes), np.argsort(codes, kind="stable")
    )
    assert decomposition.stable_argsort(codes[:1]).tolist() == [0]
    assert decomposition.stable_argsort(codes[:0]).size == 0
