"""Contract of the public PointCloud API: every supported input combination
runs, every unsupported one raises the documented error type.

Small clouds only; numerical correctness is covered elsewhere.
"""

import numpy as np
import pytest

from smudgy import PointCloud

N = 120
FIXED = ("ngp", "cic", "tsc", "bspline4", "bspline5", "bspline6")
SEPARABLE_KERNELS = ("tophat", "tsc", "gaussian")
BACKENDS = ("numpy", "quadrants")


def _cloud(dim=2, periodic=True, backend="quadrants", seed=0):
    rng = np.random.default_rng(seed)
    pc = PointCloud(
        rng.uniform(0, 1, (N, dim)),
        weights=rng.uniform(0.5, 1.5, N),
        boxsize=1.0 if periodic else None,
        backend=backend,
        verbose=False,
    )
    pc.add_fields(["s", "v"], [rng.standard_normal(N), rng.standard_normal((N, dim))])
    return pc


def _ready(dim=2, backend="quadrants", periodic=True):
    """Cloud with isotropic and covariant smoothing and densities."""
    pc = _cloud(dim, periodic, backend).global_setup(
        num_neighbors=12, kernel_name="cubic_spline")
    for structure in ("isotropic", "covariant"):
        pc.compute_smoothing(structure=structure)
    for structure in ("isotropic", "covariant"):
        pc.compute_density(structure=structure)
    return pc


# =============================================================================
# Construction
# =============================================================================
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("weights", [None, "array"])
@pytest.mark.parametrize("boxsize", [None, 1.0, "per-axis"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_init_combinations(dim, weights, boxsize, backend):
    rng = np.random.default_rng(0)
    w = rng.uniform(size=N) if weights else None
    box = [1.0 + d for d in range(dim)] if boxsize == "per-axis" else boxsize
    pc = PointCloud(rng.uniform(0, 1, (N, dim)), weights=w, boxsize=box,
                    backend=backend, verbose=False)
    assert pc.positions.shape == (N, dim) and pc.weights.shape == (N,)
    assert pc.periodic == (box is not None)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(positions=np.zeros((N, 4))),
        dict(positions=np.zeros(N)),
        dict(positions=np.zeros((N, 2)), weights=np.ones(N + 1)),
        dict(positions=np.zeros((N, 2)), weights=np.ones((N, 2))),
        dict(positions=np.zeros((N, 2)), boxsize=[1.0, 1.0, 1.0]),
        dict(positions=np.zeros((N, 2)), boxsize=0.0),
        dict(positions=np.zeros((N, 2)), backend="torch"),
    ],
)
def test_init_invalid(kwargs):
    with pytest.raises(ValueError):
        PointCloud(verbose=False, **kwargs)


# =============================================================================
# Smoothing and density
# =============================================================================
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("structure", ["separable", "isotropic", "covariant"])
@pytest.mark.parametrize("query", [False, True])
@pytest.mark.parametrize("backend", BACKENDS)
def test_compute_smoothing(dim, structure, query, backend):
    pc = _cloud(dim, backend=backend).global_setup(num_neighbors=8)
    q = np.random.default_rng(1).uniform(0, 1, (17, dim)) if query else None
    pc.compute_smoothing(query_positions=q, structure=structure)
    n = 17 if query else N
    if structure == "covariant":
        assert pc.smoothing.smoothing_tensors.shape == (n, dim, dim)
    else:
        assert pc.smoothing.smoothing_lengths.shape == (n,)


def test_compute_smoothing_invalid():
    pc = _cloud().global_setup(num_neighbors=8)
    with pytest.raises(AttributeError):  # no structure anywhere
        pc.compute_smoothing()
    with pytest.raises(ValueError):
        pc.compute_smoothing(structure="spherical")
    with pytest.raises(ValueError):
        pc.compute_smoothing(structure="covariant", ratio_max=0.5)
    with pytest.raises(ValueError):
        pc.compute_smoothing(num_neighbors=N + 1, structure="isotropic")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("structure", ["isotropic", "covariant"])
def test_compute_density(backend, structure):
    pc = _cloud(backend=backend).global_setup(num_neighbors=8, kernel_name="lucy")
    pc.compute_smoothing(structure=structure)
    pc.compute_density(structure=structure)
    assert pc.get(f"density_{structure}").shape == (N,)


def test_compute_density_invalid():
    pc = _cloud().global_setup(num_neighbors=8, kernel_name="cubic_spline")
    with pytest.raises(AttributeError):  # smoothing missing
        pc.compute_density(structure="isotropic")
    pc.compute_smoothing(structure="isotropic")
    with pytest.raises(ValueError):
        pc.compute_density(structure="separable")
    with pytest.raises(ValueError):
        pc.compute_density(structure="isotropic", kernel_name="no_such_kernel")


# =============================================================================
# Interpolation
# =============================================================================
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("structure", ["isotropic", "covariant"])
@pytest.mark.parametrize("query", [False, True])
def test_interpolate_modes(backend, dim, structure, query):
    pc = _ready(dim, backend)
    q = np.random.default_rng(1).uniform(0, 1, (9, dim)) if query else None
    m = 9 if query else N
    kw = dict(query_positions=q, structure=structure)
    assert pc.interpolate("s", **kw).shape == (m, 1)
    assert pc.interpolate(["s", "v"], **kw).shape == (m, 1 + dim)
    assert pc.interpolate("v", mode="gradient", **kw).shape == (m, dim, dim)
    assert pc.interpolate(["v", "v"], mode="divergence", **kw).shape == (m, 2)
    if dim == 2:
        assert pc.interpolate("v", mode="curl", **kw).shape == (m, 1)
    elif dim == 3:
        assert pc.interpolate("v", mode="curl", **kw).shape == (m, 1, 3)


def test_interpolate_field_forms():
    pc = _ready()
    ref = pc.interpolate(["s", "v"], structure="isotropic")
    np.testing.assert_array_equal(
        pc.interpolate([pc.s, "v"], structure="isotropic"), ref)
    np.testing.assert_array_equal(
        pc.interpolate(np.column_stack([pc.s, pc.v]), structure="isotropic"), ref)


def test_interpolate_invalid():
    pc = _ready(dim=2)
    with pytest.raises(ValueError):
        pc.interpolate("s", mode="laplacian", structure="isotropic")
    with pytest.raises(ValueError):
        pc.interpolate("s", mode="divergence", structure="isotropic")
    with pytest.raises(ValueError):
        pc.interpolate("s", structure="separable")
    with pytest.raises(ValueError):
        pc.interpolate(np.ones(N + 1), structure="isotropic")
    with pytest.raises(AttributeError):
        pc.interpolate("no_such_field", structure="isotropic")
    with pytest.raises(ValueError):
        _ready(dim=1).interpolate("v", mode="curl", structure="isotropic")
    fresh = _cloud().global_setup(num_neighbors=8, structure="isotropic")
    with pytest.raises(AttributeError):  # no density yet
        fresh.interpolate("s")


# =============================================================================
# Deposition
# =============================================================================
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("kernel", FIXED)
@pytest.mark.parametrize("periodic", [True, False])
def test_deposit_fixed(backend, dim, kernel, periodic):
    pc = _cloud(dim, periodic, backend)
    ext = None if periodic else [[0.0, 1.0]] * dim
    f, w = pc.deposit("s", averaged=False, gridnums=8, extent=ext, adaptive=False,
                      kernel_name=kernel, return_weights=True)
    assert f.shape == (1,) + (8,) * dim and w.shape == (8,) * dim


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize(
    "structure,kernel",
    [("separable", k) for k in SEPARABLE_KERNELS]
    + [(s, k) for s in ("isotropic", "covariant") for k in ("tophat", "cubic_spline", "wendland_c6")],
)
@pytest.mark.parametrize("periodic", [True, False])
def test_deposit_adaptive(dim, structure, kernel, periodic):
    pc = _ready(dim, periodic=periodic)
    ext = None if periodic else [[0.0, 1.0]] * dim
    f, w = pc.deposit(["s", "v"], averaged=[True, False], gridnums=8, extent=ext,
                      structure=structure, kernel_name=kernel, return_weights=True)
    assert f.shape == (1 + dim,) + (8,) * dim and w.shape == (8,) * dim
    assert np.isfinite(f).all()


@pytest.mark.parametrize(
    "kw",
    [
        dict(adaptive=False, kernel_name="cic"),
        dict(structure="separable", kernel_name="tsc"),
        dict(structure="isotropic"),
        dict(structure="covariant"),
    ],
)
def test_deposit_plane_projection(kw):
    pc = _ready(dim=3)
    grid = pc.deposit("s", averaged=False, gridnums=(8, 6), plane_projection=[2, 0], **kw)
    assert grid.shape == (1, 8, 6)


@pytest.mark.parametrize("method", ["midpoint", "trapezoidal", "simpson"])
def test_deposit_integration_methods(method):
    pc = _ready()
    pc.deposit("s", averaged=False, gridnums=8, structure="isotropic",
               integration_method=method)


def test_deposit_output_forms():
    pc = _ready()
    kw = dict(gridnums=8, structure="isotropic")
    mass = pc.deposit(None, averaged=False, **kw)
    f, w = pc.deposit("v", averaged=False, return_weights=True, **kw)
    np.testing.assert_allclose(mass, w)
    per_component = pc.deposit("v", averaged=[True, False], **kw)
    per_field = pc.deposit(["v"], averaged=[True], **kw)
    np.testing.assert_allclose(per_component[0], per_field[0])
    root = pc.deposit("v", averaged=False, gather_to_root=True, **kw)
    np.testing.assert_allclose(root, f)


def test_deposit_numpy_backend_is_fixed_grid_only():
    pc = _ready(backend="numpy")
    pc.deposit("s", averaged=False, gridnums=8, adaptive=False, kernel_name="tsc")
    with pytest.raises(ValueError):
        pc.deposit("s", averaged=False, gridnums=8, structure="isotropic")


@pytest.mark.parametrize(
    "kw",
    [
        dict(adaptive=False, kernel_name="cic", structure="isotropic"),
        dict(adaptive=False, kernel_name="cubic_spline"),
        dict(structure="separable", kernel_name="cubic_spline"),
        dict(structure="isotropic", kernel_name="cic"),
        dict(structure="spherical"),
        dict(structure="isotropic", integration_method="gauss"),
        dict(structure="isotropic", eta_crit=-1.0),
        dict(structure="isotropic", gridnums=(8, 8, 8)),
        dict(structure="isotropic", gridnums=0),
        dict(structure="isotropic", extent=[[0, 1]]),
        dict(structure="isotropic", extent=[[0, 1], [1, 0]]),
        dict(structure="isotropic", averaged=[True, False, True]),
        dict(fields=None, structure="isotropic", averaged=True),
    ],
)
def test_deposit_invalid(kw):
    pc = _ready()
    args = dict(fields="s", averaged=False, gridnums=8)
    args.update(kw)
    with pytest.raises(ValueError):
        pc.deposit(**args)


def test_deposit_requirements():
    pc = _cloud(periodic=False).global_setup(kernel_name="cubic_spline")
    with pytest.raises(ValueError):  # neither boxsize nor extent
        pc.deposit("s", averaged=False, gridnums=8, adaptive=False, kernel_name="cic")
    with pytest.raises(AttributeError):  # no smoothing yet
        pc.deposit("s", averaged=False, gridnums=8, extent=[[0, 1]] * 2,
                   structure="isotropic")


# =============================================================================
# Collecting results
# =============================================================================
def test_get_and_gather():
    pc = _cloud()
    with pytest.raises(AttributeError):
        pc.get("density_isotropic")
    with pytest.raises(AttributeError):
        pc.get("no_such_field")
    with pytest.raises(AttributeError):
        pc.gather_queries(np.ones(3))
    with pytest.raises(ValueError):
        pc.gather_particles(np.ones(N + 1))
    np.testing.assert_array_equal(pc.get("s"), pc.s)
