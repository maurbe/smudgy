"""SmoothingInfo dataclass to store smoothing-related information."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SmoothingInfo:
    """Smoothing information of a point cloud (``PointCloud.smoothing``).

    All arrays are rank-local, i.e. indexed like ``PointCloud.positions``.

    Parameters
    ----------
    num_neighbors : int
        Number of nearest neighbours used for smoothing.
    nn_inds : np.ndarray
        Nearest-neighbour indices, into the local particles followed by the
        ghost particles.
    nn_dists : np.ndarray
        Distances to the nearest neighbours.
    smoothing_lengths : np.ndarray
        Smoothing lengths, shape (N,).
    smoothing_tensors : np.ndarray
        Smoothing tensors, shape (N, D, D).
    smoothing_tensors_eigvals : np.ndarray
        Principal smoothing lengths (eigenvalues), shape (N, D).
    smoothing_tensors_eigvecs : np.ndarray
        Principal axes (eigenvectors, as columns), shape (N, D, D).
    tensor_options : dict
        Options used for the smoothing tensors (``ratio_max``, ``min_axis``,
        ``volume_preserving``).
    kernel_name : str
        Kernel used for the density estimates.
    density_isotropic : np.ndarray
        Isotropic (and separable) density estimates, shape (N,).
    density_covariant : np.ndarray
        Covariant density estimates, shape (N,).
    ghost_positions : np.ndarray
        Positions of the ghost particles referenced by ``nn_inds``.
    ghost_weights : np.ndarray
        Weights of the ghost particles referenced by ``nn_inds``.
    """

    num_neighbors: int = None
    nn_inds: np.ndarray = None
    nn_dists: np.ndarray = None

    smoothing_lengths: np.ndarray = None
    smoothing_tensors: np.ndarray = None
    smoothing_tensors_eigvals: np.ndarray = None
    smoothing_tensors_eigvecs: np.ndarray = None
    tensor_options: dict = None

    kernel_name: str = None
    density_isotropic: np.ndarray = None
    density_covariant: np.ndarray = None
    ghost_positions: np.ndarray = None
    ghost_weights: np.ndarray = None
