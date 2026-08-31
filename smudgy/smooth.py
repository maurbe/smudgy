"""SmoothingInfo dataclass to store smoothing-related information."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SmoothingInfo:
    """Dataclass to store smoothing-related information.

    Parameters
    ----------
    tree : object
        Neighbor search tree (e.g., KDTree) for efficient neighbor queries.
    num_neighbors : int
        Number of nearest neighbors used for smoothing.
    nn_inds : np.ndarray
        Indices of nearest neighbors for each particle.
    nn_dists : np.ndarray
        Distances to nearest neighbors for each particle.
    nn_dists_vec : np.ndarray
        Vector distances to nearest neighbors for each particle.
    smoothing_lengths : np.ndarray
        Smoothing lengths for each particle.
    smoothing_tensors : np.ndarray
        Smoothing tensors for each particle.
    smoothing_tensors_eigvals : np.ndarray
        Eigenvalues of the smoothing tensors.
    smoothing_tensors_eigvecs : np.ndarray
        Eigenvectors of the smoothing tensors.
    kernel_name : str
        Name of the smoothing kernel used.
    density_isotropic : np.ndarray
        Isotropic density estimates for each particle.
    density_covariant : np.ndarray
        Anisotropic density estimates for each particle.
    used_ghosts : bool
        Whether the most recent `compute_smoothing()` call used the
        local+ghost path (see `PointCloud._using_decomposition`) rather than
        the full-replication path. `compute_density`/`interpolate`/`deposit`
        key off this (not off `_using_decomposition()` independently) so
        every stage's path decision is provably consistent with what
        `compute_smoothing()` actually produced -- e.g. `nn_inds` and
        `smoothing_lengths`/`smoothing_tensors` are local-sized when this is
        True, full-N-sized when False, and indexing one with the wrong
        convention would silently read the wrong particle's data rather than
        raise.

    """

    tree: object = None
    num_neighbors: int = None
    nn_inds: np.ndarray = None
    nn_dists: np.ndarray = None
    nn_dists_vec: np.ndarray = None

    smoothing_lengths: np.ndarray = None
    smoothing_tensors: np.ndarray = None
    smoothing_tensors_eigvals: np.ndarray = None
    smoothing_tensors_eigvecs: np.ndarray = None

    kernel_name: str = None
    density_isotropic: np.ndarray = None
    density_covariant: np.ndarray = None
    used_ghosts: bool = False
