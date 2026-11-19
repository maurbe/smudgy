"""Covariant deposition conserves weight on non-cubic grid cells.

Regression: the support box of an ellipsoid was converted to cell units with
the cell size of the eigen index instead of the grid axis, so on non-cubic
cells the box was too small and samples outside it were dropped (up to ~6% of
the weight).
"""

import numpy as np
import pytest

from smudgy.backend.quadrants import deposition as dep


@pytest.mark.parametrize("gridnums", [(8, 64), (64, 8), (8, 32, 64)])
@pytest.mark.parametrize("eta_crit", [4.0, 1e-3])
def test_covariant_noncubic_cells_conserve_weight(gridnums, eta_crit):
    dim = len(gridnums)
    rng = np.random.default_rng(1)
    n = 1000
    pos = rng.random((n, dim)).astype(np.float32)
    vecs = np.linalg.qr(rng.normal(size=(n, dim, dim)))[0].astype(np.float32)
    vals = np.sort(rng.uniform(0.005, 0.06, (n, dim)), axis=1)[:, ::-1]
    _, grid = dep.deposit(
        pos, np.ones((n, 1), np.float32), np.ones(n, np.float32),
        np.ones(dim, np.float32), gridnums, adaptive=True, structure="covariant",
        kernel_name="cubic_spline", periodic=True, particle_hmat_eigvecs=vecs,
        particle_hmat_eigvals=vals.astype(np.float32), eta_crit=eta_crit,
    )
    assert grid.sum() == pytest.approx(n, rel=1e-5)
