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
    smoLens : np.ndarray
        Smoothing lengths for each particle.
    smoTens : np.ndarray
        Smoothing tensors for each particle.
    smoTens_eigvals : np.ndarray
        Eigenvalues of the smoothing tensors.
    smoTens_eigvecs : np.ndarray
        Eigenvectors of the smoothing tensors.
    smoTens_projected : np.ndarray
        Projected smoothing tensors for each particle.
    smoTens_projected_eigvals : np.ndarray
        Eigenvalues of the projected smoothing tensors.
    smoTens_projected_eigvecs : np.ndarray
        Eigenvectors of the projected smoothing tensors.
    kernel_name : str
        Name of the smoothing kernel used.
    density_iso : np.ndarray
        Isotropic density estimates for each particle.
    density_aniso : np.ndarray
        Anisotropic density estimates for each particle.

    """

    tree: object = None
    num_neighbors: int = None
    nn_inds: np.ndarray = None
    nn_dists: np.ndarray = None
    nn_dists_vec: np.ndarray = None

    smoLens: np.ndarray = None

    smoTens: np.ndarray = None
    smoTens_eigvals: np.ndarray = None
    smoTens_eigvecs: np.ndarray = None

    smoTens_projected: np.ndarray = None
    smoTens_projected_eigvals: np.ndarray = None
    smoTens_projected_eigvecs: np.ndarray = None

    kernel_name: str = None
    density_iso: np.ndarray = None
    density_aniso: np.ndarray = None
