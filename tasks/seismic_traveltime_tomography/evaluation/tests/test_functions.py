"""Unit tests for seismic traveltime tomography src/ modules."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import numpy as np
from numpy.testing import assert_allclose

from src.physics_model import (
    solve_eikonal,
    compute_traveltime_at,
    compute_traveltime_gradient,
    compute_all_traveltimes,
)
from src.preprocessing import compute_residuals, compute_misfit
from src.generate_data import (
    make_background_velocity,
    make_checkerboard_perturbation,
    make_marmousi_velocity,
    make_true_velocity,
    make_receivers,
    make_sources,
)
from src.visualization import compute_ncc, compute_nrmse


# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture
def small_grid():
    """Small grid for fast tests: 21×11 nodes, dx=dz=2 km."""
    return dict(Nx=21, Nz=11, dx=2.0, dz=2.0)


@pytest.fixture
def constant_slowness(small_grid):
    return np.full((small_grid['Nz'], small_grid['Nx']),
                   1.0 / 5.0, dtype=np.float64)  # 5 km/s everywhere


@pytest.fixture
def data_dir():
    return os.path.join(os.path.dirname(__file__), '..', '..', 'data')


# --------------------------------------------------------------------------- #
#  physics_model tests                                                         #
# --------------------------------------------------------------------------- #

class TestEikonalSolver:

    def test_output_shape(self, constant_slowness, small_grid):
        T = solve_eikonal(constant_slowness,
                          small_grid['dx'], small_grid['dz'],
                          20.0, 10.0)
        assert T.shape == (small_grid['Nz'], small_grid['Nx'])

    def test_output_dtype(self, constant_slowness, small_grid):
        T = solve_eikonal(constant_slowness,
                          small_grid['dx'], small_grid['dz'], 20.0, 10.0)
        assert T.dtype == np.float32

    def test_traveltime_monotone_from_source(self, constant_slowness, small_grid):
        """Traveltime increases with distance from source."""
        dx, dz = small_grid['dx'], small_grid['dz']
        src_x, src_z = 20.0, 10.0
        T = solve_eikonal(constant_slowness, dx, dz, src_x, src_z)
        Nz, Nx = T.shape
        xs = np.arange(Nx) * dx
        zs = np.arange(Nz) * dz
        xx, zz = np.meshgrid(xs, zs)
        dist = np.sqrt((xx - src_x) ** 2 + (zz - src_z) ** 2)
        # Sort by distance and check monotone increase on average
        sort_idx = np.argsort(dist.ravel())
        T_sorted = T.ravel()[sort_idx]
        dist_sorted = dist.ravel()[sort_idx]
        # T should be positively correlated with distance
        cc = np.corrcoef(T_sorted, dist_sorted)[0, 1]
        assert cc > 0.99, f"Expected strong correlation, got {cc:.4f}"

    def test_analytical_accuracy(self, constant_slowness, small_grid):
        """FMM traveltime matches analytical solution to within 5% at far field."""
        dx, dz = small_grid['dx'], small_grid['dz']
        src_x, src_z = 20.0, 10.0
        T = solve_eikonal(constant_slowness, dx, dz, src_x, src_z)
        # Test at a point far from source
        rec_x, rec_z = 0.0, 0.0
        T_calc = compute_traveltime_at(T, rec_x, rec_z, dx, dz)
        dist = np.sqrt((rec_x - src_x) ** 2 + (rec_z - src_z) ** 2)
        T_true = dist / 5.0  # v = 5 km/s
        assert abs(T_calc - T_true) / T_true < 0.05

    def test_nonnegative(self, constant_slowness, small_grid):
        T = solve_eikonal(constant_slowness,
                          small_grid['dx'], small_grid['dz'], 10.0, 6.0)
        assert np.all(T >= 0)

    def test_source_clamp(self, constant_slowness, small_grid):
        """Source outside grid is clamped to boundary without error."""
        T = solve_eikonal(constant_slowness,
                          small_grid['dx'], small_grid['dz'],
                          -5.0, -5.0)  # outside grid
        assert T.shape == (small_grid['Nz'], small_grid['Nx'])


class TestTraveltimeGradient:

    def test_gradient_shape(self, constant_slowness, small_grid):
        dx, dz = small_grid['dx'], small_grid['dz']
        T = solve_eikonal(constant_slowness, dx, dz, 20.0, 10.0)
        dTdz, dTdx = compute_traveltime_gradient(T, dx, dz)
        assert dTdz.shape == T.shape
        assert dTdx.shape == T.shape

    def test_gradient_magnitude_near_slowness(self, constant_slowness, small_grid):
        """|∇T| ≈ slowness (Eikonal condition) away from source."""
        dx, dz = small_grid['dx'], small_grid['dz']
        s = constant_slowness[0, 0]
        T = solve_eikonal(constant_slowness, dx, dz, 20.0, 10.0)
        dTdz, dTdx = compute_traveltime_gradient(T, dx, dz)
        grad_mag = np.sqrt(dTdz ** 2 + dTdx ** 2)
        # Exclude source neighbourhood and boundary
        interior = grad_mag[2:-2, 2:-2]
        assert_allclose(interior.mean(), s, rtol=0.1,
                        err_msg="|∇T| should be close to slowness")


class TestComputeAllTraveltimes:

    def test_shape(self, constant_slowness, small_grid):
        dx, dz = small_grid['dx'], small_grid['dz']
        sources   = np.array([[10.0, 8.0], [20.0, 6.0]])
        receivers = np.array([[0.0, 0.0], [40.0, 0.0], [20.0, 0.0]])
        T_syn = compute_all_traveltimes(constant_slowness, dx, dz,
                                        sources, receivers)
        assert T_syn.shape == (2, 3)

    def test_symmetry(self, constant_slowness, small_grid):
        """Traveltime is symmetric: T(A→B) ≈ T(B→A) for isotropic media."""
        dx, dz = small_grid['dx'], small_grid['dz']
        src = np.array([[5.0, 8.0]])
        rec = np.array([[35.0, 2.0]])
        T1 = compute_all_traveltimes(constant_slowness, dx, dz, src, rec)
        T2 = compute_all_traveltimes(constant_slowness, dx, dz, rec, src)
        assert_allclose(T1[0, 0], T2[0, 0], rtol=0.02,
                        err_msg="Reciprocity violated")


# --------------------------------------------------------------------------- #
#  preprocessing tests                                                         #
# --------------------------------------------------------------------------- #

class TestPreprocessing:

    def test_residual_shape(self):
        T_syn = np.random.rand(5, 3).astype(np.float32)
        T_obs = np.random.rand(5, 3).astype(np.float32)
        R = compute_residuals(T_syn, T_obs)
        assert R.shape == (5, 3)

    def test_residual_values(self):
        T_syn = np.array([[2.0, 3.0]], dtype=np.float32)
        T_obs = np.array([[1.5, 3.5]], dtype=np.float32)
        R = compute_residuals(T_syn, T_obs)
        assert_allclose(R, [[0.5, -0.5]], atol=1e-6)

    def test_misfit_formula(self):
        R = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        expected = 0.5 * (1 + 4 + 9 + 16)
        assert_allclose(compute_misfit(R), expected, rtol=1e-5)

    def test_zero_residual(self):
        R = np.zeros((4, 8), dtype=np.float32)
        assert compute_misfit(R) == 0.0


# --------------------------------------------------------------------------- #
#  generate_data tests                                                         #
# --------------------------------------------------------------------------- #

class TestModelBuilders:

    def test_background_shape(self, small_grid):
        v = make_background_velocity(
            small_grid['Nx'], small_grid['Nz'],
            small_grid['dx'], small_grid['dz'], 4.0, 8.0)
        assert v.shape == (small_grid['Nz'], small_grid['Nx'])

    def test_background_range(self, small_grid):
        v = make_background_velocity(
            small_grid['Nx'], small_grid['Nz'],
            small_grid['dx'], small_grid['dz'], 4.0, 8.0)
        assert_allclose(v[0, :].mean(), 4.0, atol=1e-4)
        assert_allclose(v[-1, :].mean(), 8.0, atol=0.1)

    def test_background_x_constant(self, small_grid):
        """Background velocity should be constant along x (1D gradient in z)."""
        v = make_background_velocity(
            small_grid['Nx'], small_grid['Nz'],
            small_grid['dx'], small_grid['dz'], 4.0, 8.0)
        assert_allclose(v[:, 0], v[:, -1], rtol=1e-5)

    def test_checkerboard_zero_mean(self, small_grid):
        """Checkerboard perturbation should have approximately zero mean."""
        dv = make_checkerboard_perturbation(
            small_grid['Nx'], small_grid['Nz'],
            small_grid['dx'], small_grid['dz'])
        assert abs(dv.mean()) < 0.05, f"Mean too large: {dv.mean()}"

    def test_checkerboard_amplitude(self, small_grid):
        """Checkerboard max should equal pert parameter."""
        pert = 0.05
        dv = make_checkerboard_perturbation(
            small_grid['Nx'], small_grid['Nz'],
            small_grid['dx'], small_grid['dz'], pert=pert)
        assert_allclose(np.abs(dv).max(), pert, rtol=0.01)

    def test_true_velocity_range(self, small_grid):
        """True model = background × (1 + perturbation)."""
        v = make_true_velocity(
            small_grid['Nx'], small_grid['Nz'],
            small_grid['dx'], small_grid['dz'],
            v0=4.0, v1=8.0, pert=0.05)
        v_bg = make_background_velocity(
            small_grid['Nx'], small_grid['Nz'],
            small_grid['dx'], small_grid['dz'], 4.0, 8.0)
        # True model should be ±5% around background
        ratio = v / v_bg
        assert_allclose(ratio.min(), 0.95, atol=0.01)
        assert_allclose(ratio.max(), 1.05, atol=0.01)

    def test_receivers_on_surface(self):
        rec = make_receivers(8, 220.0)
        assert rec.shape == (8, 2)
        assert np.all(rec[:, 1] == 0.0)  # z = 0

    def test_receivers_perimeter(self):
        """Perimeter layout: 8 surf + 8 left + 8 right = 24 total."""
        rec = make_receivers(24, 220.0, z_max=60.0, n_surf=8, n_side=8)
        assert rec.shape == (24, 2)
        assert np.sum(rec[:, 1] == 0.0) == 8    # 8 surface
        assert np.sum(rec[:, 0] == 0.0) == 8    # 8 left
        assert np.sum(rec[:, 0] == 220.0) == 8  # 8 right

    def test_sources_count(self):
        src = make_sources(100, 220.0)
        assert src.shape == (100, 2)

    def test_sources_2d_random_in_domain(self):
        """2-D random sources should lie within the domain."""
        src = make_sources(100, 220.0, z_max=50.0)
        assert src.shape == (100, 2)
        assert np.all(src[:, 0] >= 0) and np.all(src[:, 0] <= 220.0)
        assert np.all(src[:, 1] >= 0) and np.all(src[:, 1] <= 50.0)

    def test_sources_in_domain(self):
        """Legacy fixed-depth sources should lie within the domain."""
        src = make_sources(100, 220.0)
        assert np.all(src[:, 0] >= 0) and np.all(src[:, 0] <= 220.0)
        assert np.all(src[:, 1] >= 0)


# --------------------------------------------------------------------------- #
#  visualization / metrics tests                                               #
# --------------------------------------------------------------------------- #

class TestMetrics:

    def test_ncc_identical(self):
        a = np.random.rand(20)
        assert_allclose(compute_ncc(a, a), 1.0, atol=1e-5)

    def test_ncc_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert_allclose(compute_ncc(a, b), 0.0, atol=1e-10)

    def test_ncc_range(self):
        a = np.random.rand(50)
        b = np.random.rand(50)
        ncc = compute_ncc(a, b)
        assert -1.0 <= ncc <= 1.0

    def test_nrmse_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert_allclose(compute_nrmse(a, a), 0.0, atol=1e-10)

    def test_nrmse_formula(self):
        pred = np.array([1.0, 2.0, 3.0])
        ref  = np.array([1.0, 3.0, 5.0])
        rms  = np.sqrt(np.mean((pred - ref) ** 2))
        dr   = ref.max() - ref.min()
        assert_allclose(compute_nrmse(pred, ref), rms / dr, rtol=1e-5)


# --------------------------------------------------------------------------- #
#  Marmousi model tests                                                        #
# --------------------------------------------------------------------------- #

class TestMarmousiModel:

    def test_shape(self, data_dir):
        v = make_marmousi_velocity(data_dir)
        assert v.shape == (31, 93)

    def test_dtype(self, data_dir):
        v = make_marmousi_velocity(data_dir)
        assert v.dtype == np.float64

    def test_velocity_range(self, data_dir):
        v = make_marmousi_velocity(data_dir)
        assert v.min() >= 1.4, f"min velocity too low: {v.min():.3f}"
        assert v.max() <= 5.6, f"max velocity too high: {v.max():.3f}"

    def test_depth_gradient(self, data_dir):
        """Marmousi velocity should increase on average with depth."""
        v = make_marmousi_velocity(data_dir)
        assert v[0, :].mean() < v[-1, :].mean()
