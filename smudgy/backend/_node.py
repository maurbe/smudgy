"""Per-node resource sharing between the MPI ranks of one machine.

Every rank runs its own Quadrants runtime and k-d tree queries. By default
each of them would use *all* cores of the node, and on GPU all of them would
bind CUDA device 0. The helpers below split cores and GPUs between the ranks
of a node. They only read the launcher's environment (no MPI communication),
so they are safe to call from a single rank.
"""

import os
import shutil
import subprocess
import sys
import warnings

# (local rank, local size) variables set by common MPI launchers
_LOCAL_ENV = (
    ("OMPI_COMM_WORLD_LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_SIZE"),  # Open MPI
    ("MPI_LOCALRANKID", "MPI_LOCALNRANKS"),  # MPICH / Intel MPI (Hydra)
    ("MV2_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_SIZE"),  # MVAPICH2
    ("PALS_LOCAL_RANKID", "PALS_LOCAL_SIZE"),  # HPE Cray PALS
    ("SLURM_LOCALID", "SLURM_NTASKS_PER_NODE"),  # srun (if set per node)
)


def node_local():
    """``(local_rank, local_size)`` of this process among the ranks on its node.

    ``(0, 1)`` if the launcher does not export them, or for a single-rank run
    (e.g. a serial script inside a multi-task Slurm allocation).
    """
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() == 1:
        return 0, 1
    for rank_var, size_var in _LOCAL_ENV:
        rank, size = os.environ.get(rank_var), os.environ.get(size_var)
        if rank is not None and size is not None:
            try:
                return int(rank), max(1, int(size))
            except ValueError:
                continue
    return 0, 1


def cpu_threads():
    """CPU threads this rank should use, or None for "all cores" (no sharing).

    If the launcher pinned the rank to a subset of cores (affinity), that
    subset is used; otherwise the node's cores are split evenly between its
    ranks.
    """
    total = os.cpu_count() or 1
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS: no affinity API
        available = total
    if available < total:
        return available
    _, local_size = node_local()
    if local_size == 1:
        return None
    return max(1, total // local_size)


def kdtree_workers():
    """``workers`` argument for cKDTree queries (-1 = all cores)."""
    return cpu_threads() or -1


def _cuda_device_count():
    """Count CUDA GPUs via ``nvidia-smi -L``; None if unknown."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    n = sum(line.startswith("GPU ") for line in out.stdout.splitlines())
    return n or None


def gpu_env(arch, device="auto"):
    """Environment updates that bind this rank to one GPU, before ``qd.init``.

    Quadrants always opens CUDA device 0 (and the best Vulkan device), so the
    device is chosen through what the driver exposes: ``CUDA_VISIBLE_DEVICES``
    (CUDA), or ``QD_VISIBLE_DEVICE`` (Vulkan).

    Parameters
    ----------
    arch : str
        Quadrants arch name. Only ``"gpu"``, ``"cuda"`` and ``"vulkan"`` on
        non-macOS systems select a device.
    device : "auto", int or None
        ``"auto"``: the node-local rank, if several ranks share the node.
        An int: that device (modulo the visible devices). None: no change.

    Returns
    -------
    dict
        Variables to set (empty if nothing to do).
    """
    if device is None or arch not in ("gpu", "cuda", "vulkan") or sys.platform == "darwin":
        return {}
    if device == "auto":
        local_rank, local_size = node_local()
        if local_size == 1:
            return {}
        device = local_rank
    device = int(device)

    if arch == "vulkan":
        if "QD_VISIBLE_DEVICE" in os.environ:
            return {}
        return {"QD_VISIBLE_DEVICE": str(device)}

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        ids = [v for v in visible.split(",") if v.strip()]
        if len(ids) <= 1:  # launcher / user already assigned one device
            return {}
        return {"CUDA_VISIBLE_DEVICES": ids[device % len(ids)].strip()}
    count = _cuda_device_count()
    if count is None:
        warnings.warn(
            "Could not count the CUDA devices (no nvidia-smi); every rank on "
            "this node uses GPU 0. Set CUDA_VISIBLE_DEVICES per rank, or pass "
            "device=<int> to smudgy's quadrants init.",
            stacklevel=3,
        )
        return {}
    return {"CUDA_VISIBLE_DEVICES": str(device % count)}
