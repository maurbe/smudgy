"""SmoothingInfo dataclass to store smoothing-related information."""

import numpy as np

from dataclasses import dataclass

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
