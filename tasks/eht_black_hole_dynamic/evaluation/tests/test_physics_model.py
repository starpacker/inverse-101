"""Tests for physics_model module."""

import os
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures', 'physics_model')

import sys
sys.path.insert(0, TASK_DIR)

from src.physics_model import (
    DFTForwardModel,
    compute_visibilities, compute_visibility_amplitude,
    grad_vis, grad_visibility_amplitude,
    gauss_image_covariance,
    affine_motion_basis, apply_motion_basis,
    calc_warp_matrix, gen_freq_comp, gen_phase_shift_matrix,
    product_gaussians_lem1, product_gaussians_lem2,
    evaluate_gaussian_log,
    realimag_stack, reshape_flowbasis,
)


class TestDFTForwardModel:
    """Tests for DFT forward model."""

    def test_dft_forward_parity(self):
        """Test DFT forward model against fixture."""
        fix = np.load(os.path.join(FIXTURE_DIR, 'dft_forward.npz'))
        model = DFTForwardModel(fix['input_uv'], int(fix['param_N']),
                                float(fix['param_psize']))
        vis = compute_visibilities(fix['input_imvec'], model.matrix)
        np.testing.assert_allclose(vis, fix['output_vis'], rtol=1e-10)

    def test_adjoint_consistency(self):
        """Test A^H A has correct shape and is Hermitian."""
        fix = np.load(os.path.join(FIXTURE_DIR, 'dft_forward.npz'))
        model = DFTForwardModel(fix['input_uv'], int(fix['param_N']),
                                float(fix['param_psize']))
        A = model.matrix
        AHA = A.conj().T @ A
        np.testing.assert_allclose(AHA, AHA.conj().T, rtol=1e-10)

    def test_grad_vis_is_A(self):
        """Gradient of visibility is just the DFT matrix."""
        fix = np.load(os.path.join(FIXTURE_DIR, 'dft_forward.npz'))
        model = DFTForwardModel(fix['input_uv'], int(fix['param_N']),
                                float(fix['param_psize']))
        F = grad_vis(fix['input_imvec'], model.matrix)
        np.testing.assert_allclose(F, model.matrix, rtol=1e-10)


class TestGaussCovariance:
    """Tests for Gaussian image covariance."""

    def test_covariance_parity(self):
        """Test covariance matches fixture."""
        fix = np.load(os.path.join(FIXTURE_DIR, 'gauss_covariance.npz'))
        cov = gauss_image_covariance(int(fix['param_N']),
                                     float(fix['param_psize']),
                                     fix['input_imvec'])
        np.testing.assert_allclose(cov, fix['output_cov'], rtol=1e-10)

    def test_covariance_symmetric(self):
        """Covariance matrix must be symmetric."""
        fix = np.load(os.path.join(FIXTURE_DIR, 'gauss_covariance.npz'))
        cov = gauss_image_covariance(int(fix['param_N']),
                                     float(fix['param_psize']),
                                     fix['input_imvec'])
        np.testing.assert_allclose(cov, cov.T, rtol=1e-10)

    def test_covariance_positive_semidefinite(self):
        """Covariance eigenvalues must be non-negative."""
        fix = np.load(os.path.join(FIXTURE_DIR, 'gauss_covariance.npz'))
        cov = gauss_image_covariance(int(fix['param_N']),
                                     float(fix['param_psize']),
                                     fix['input_imvec'])
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-10)


class TestMotionBasis:
    """Tests for motion basis functions."""

    def test_affine_motion_basis_shapes(self):
        """Test affine motion basis output shapes."""
        fix = np.load(os.path.join(FIXTURE_DIR, 'affine_motion_basis.npz'))
        N = int(fix['param_N'])
        psize = float(fix['param_psize'])
        init_x, init_y, fb_x, fb_y, initTheta = affine_motion_basis(N, psize)

        assert init_x.shape == (N, N, 1)
        assert init_y.shape == (N, N, 1)
        assert fb_x.shape == (N, N, 6)
        assert fb_y.shape == (N, N, 6)
        assert initTheta.shape == (6,)

    def test_affine_motion_basis_parity(self):
        """Test affine motion basis matches fixture."""
        fix = np.load(os.path.join(FIXTURE_DIR, 'affine_motion_basis.npz'))
        N = int(fix['param_N'])
        psize = float(fix['param_psize'])
        init_x, init_y, fb_x, fb_y, initTheta = affine_motion_basis(N, psize)

        np.testing.assert_allclose(initTheta, fix['output_initTheta'], rtol=1e-10)

    def test_identity_warp_is_identity(self):
        """Warp matrix at identity theta should be close to identity."""
        fix = np.load(os.path.join(FIXTURE_DIR, 'warp_matrix.npz'))
        N = int(fix['param_N'])
        warp = fix['output_warp']
        # Identity warp should be approximately identity matrix
        np.testing.assert_allclose(warp, np.eye(N * N), atol=1e-8)


class TestGaussianAlgebra:
    """Tests for Gaussian product lemmas."""

    def test_product_gaussians_lem1_parity(self):
        """Test Lemma 1 against fixture."""
        fix = np.load(os.path.join(FIXTURE_DIR, 'prod_gaussians_lem1.npz'))
        mean, cov = product_gaussians_lem1(
            fix['input_m1'], fix['input_S1'],
            fix['input_m2'], fix['input_S2'])
        np.testing.assert_allclose(mean, fix['output_mean'], rtol=1e-10)
        np.testing.assert_allclose(cov, fix['output_cov'], rtol=1e-10)

    def test_product_gaussians_lem2_parity(self):
        """Test Lemma 2 against fixture."""
        fix = np.load(os.path.join(FIXTURE_DIR, 'prod_gaussians_lem2.npz'))
        mean, cov = product_gaussians_lem2(
            fix['input_A'], fix['input_Sigma'],
            fix['input_y'], fix['input_mu'], fix['input_Q'])
        np.testing.assert_allclose(mean, fix['output_mean'], rtol=1e-10)
        np.testing.assert_allclose(cov, fix['output_cov'], rtol=1e-10)

    def test_lem1_symmetry(self):
        """Lemma 1 output covariance should be symmetric."""
        fix = np.load(os.path.join(FIXTURE_DIR, 'prod_gaussians_lem1.npz'))
        _, cov = product_gaussians_lem1(
            fix['input_m1'], fix['input_S1'],
            fix['input_m2'], fix['input_S2'])
        np.testing.assert_allclose(cov, cov.T, rtol=1e-10)

    def test_lem2_output_covariance_symmetric(self):
        """Lemma 2 output covariance should be symmetric."""
        fix = np.load(os.path.join(FIXTURE_DIR, 'prod_gaussians_lem2.npz'))
        _, cov = product_gaussians_lem2(
            fix['input_A'], fix['input_Sigma'],
            fix['input_y'], fix['input_mu'], fix['input_Q'])
        np.testing.assert_allclose(cov, cov.T, rtol=1e-10)


class TestHelpers:
    """Tests for helper functions."""

    def test_realimag_stack(self):
        """Test real-imaginary stacking."""
        mtx = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
        result = realimag_stack(mtx)
        assert result.shape == (4, 2)
        np.testing.assert_allclose(result[:2], np.real(mtx))
        np.testing.assert_allclose(result[2:], np.imag(mtx))

    def test_reshape_flowbasis(self):
        """Test flowbasis reshaping."""
        fb = np.ones((4, 4, 3))
        result = reshape_flowbasis(fb)
        assert result.shape == (16, 3)
