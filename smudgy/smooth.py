"""SmoothingInfo dataclass to store smoothing-related information."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SmoothingInfo:
    """Dataclass to store smoothing-related information.

    Parameters
    ----------
    num_neighbors : int
        Number of nearest neighbors used for smoothing.
    nn_inds : np.ndarray
        Indices of nearest neighbors for each particle, into the combined
        [local particles, ghost particles] array -- see `ghost_positions`/
        `ghost_weights` below for the ghost half of that array.
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
    ghost_positions : np.ndarray
        The ghost positions `nn_inds` values `>= n_local` index into
        (`n_local` = `decomposition.local_positions.shape[0]`), aliased
        (not copied) from whichever `GhostInfo` the last `compute_smoothing()`
        call actually used -- `self.ghosts` for the particle-position case,
        or a freshly-routed query-position ghost exchange otherwise.
        `compute_density`/`interpolate` reuse this (not `self.ghosts`
        directly) so they stay consistent with whichever ghost set actually
        produced the currently-stored smoothing data, rather than assuming
        it was always `self.ghosts`.
    ghost_weights : np.ndarray
        Companion to `ghost_positions`, same provenance.

    """

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
    ghost_positions: np.ndarray = None
    ghost_weights: np.ndarray = None
