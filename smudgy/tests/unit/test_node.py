"""Per-node CPU/GPU sharing defaults (`smudgy.backend._node`), MPI-free."""

import pytest

from smudgy.backend import _node


@pytest.fixture
def two_ranks(monkeypatch):
    """Pretend to be local rank 1 of 2 in a 2-rank job."""
    monkeypatch.setattr(_node, "node_local", lambda: (1, 2))
    for var in ("CUDA_VISIBLE_DEVICES", "QD_VISIBLE_DEVICE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(_node.sys, "platform", "linux")


def test_single_rank_uses_all_cores():
    assert _node.node_local() == (0, 1)  # pytest itself is a 1-rank job
    if _node.cpu_threads() is None:
        assert _node.kdtree_workers() == -1


def test_cores_split_between_ranks(two_ranks, monkeypatch):
    monkeypatch.setattr(_node.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(
        _node.os, "sched_getaffinity", lambda pid: range(16), raising=False
    )
    assert _node.cpu_threads() == 8
    assert _node.kdtree_workers() == 8
    # launcher already pinned the rank to 3 cores: use those
    monkeypatch.setattr(
        _node.os, "sched_getaffinity", lambda pid: range(3), raising=False
    )
    assert _node.cpu_threads() == 3


def test_gpu_env_cuda(two_ranks, monkeypatch):
    monkeypatch.setattr(_node, "_cuda_device_count", lambda: 4)
    assert _node.gpu_env("cuda") == {"CUDA_VISIBLE_DEVICES": "1"}
    assert _node.gpu_env("gpu", device=6) == {"CUDA_VISIBLE_DEVICES": "2"}
    assert _node.gpu_env("cuda", device=None) == {}
    assert _node.gpu_env("cpu") == {}
    # an explicit device list is indexed, a single device is left alone
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6")
    assert _node.gpu_env("cuda") == {"CUDA_VISIBLE_DEVICES": "5"}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    assert _node.gpu_env("cuda") == {}


def test_gpu_env_unknown_count_warns(two_ranks, monkeypatch):
    monkeypatch.setattr(_node, "_cuda_device_count", lambda: None)
    with pytest.warns(UserWarning, match="CUDA_VISIBLE_DEVICES"):
        assert _node.gpu_env("cuda") == {}


def test_gpu_env_vulkan_and_macos(two_ranks, monkeypatch):
    assert _node.gpu_env("vulkan") == {"QD_VISIBLE_DEVICE": "1"}
    monkeypatch.setattr(_node.sys, "platform", "darwin")
    assert _node.gpu_env("gpu") == {}
