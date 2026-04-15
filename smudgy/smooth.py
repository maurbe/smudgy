"""SmoothingInfo dataclass to store smoothing-related information"""

import numpy as np
from dataclasses import dataclass

@dataclass
class SmoothingInfo:
    smoLens: np.ndarray = None

    smoTens: np.ndarray = None
    smoTens_eigvals: np.ndarray = None
    smoTens_eigvecs: np.ndarray = None

    smoTens_projected: np.ndarray = None
    smoTens_projected_eigvals: np.ndarray = None
    smoTens_projected_eigvecs: np.ndarray = None

    nn_inds: np.ndarray = None
    nn_dists: np.ndarray = None
    rel_coords: np.ndarray = None
    
    num_neighbors: int = None
    density: np.ndarray = None
    kernel_name: str = None