"""Tests for T2 mapping solvers."""

import os
import numpy as np
import pytest

from src.solvers import (
    mono_exp_model, mono_exp_jacobian,
    levenberg_marquardt_mono_exp,
    fit_t2_loglinear, fit_t2_nonlinear,
)


class TestMonoExpModel:
    """Tests for the mono-exponential signal model and Jacobian."""

    def test_model_at_TE_zero(self):
        """At TE=0, signal should equal M0."""
        TE = np.array([0.0, 10.0, 20.0])
        s = mono_exp_model(TE, M0=1.5, T2=80.0)
        assert abs(s[0] - 1.5) < 1e-12

    def test_model_decay(self):
        """Signal should decrease monotonically with TE."""
        TE = np.arange(10, 110, 10, dtype=np.float64)
        s = mono_exp_model(TE, M0=1.0, T2=50.0)
        assert np.all(np.diff(s) < 0)

    def test_jacobian_shape(self):
        TE = np.arange(10, 110, 10, dtype=np.float64)
        J = mono_exp_jacobian(TE, M0=1.0, T2=80.0)
        assert J.shape == (10, 2)

    def test_jacobian_numerical(self):
        """Jacobian should match finite-difference approximation."""
        TE = np.array([10.0, 30.0, 50.0, 80.0, 100.0])
        M0, T2 = 1.2, 75.0
        J = mono_exp_jacobian(TE, M0, T2)

        eps = 1e-7
        J_fd = np.zeros_like(J)
        J_fd[:, 0] = (mono_exp_model(TE, M0 + eps, T2) -
                       mono_exp_model(TE, M0 - eps, T2)) / (2 * eps)
        J_fd[:, 1] = (mono_exp_model(TE, M0, T2 + eps) -
                       mono_exp_model(TE, M0, T2 - eps)) / (2 * eps)
        np.testing.assert_allclose(J, J_fd, rtol=1e-5)


class TestLevenbergMarquardt:
    """Tests for the LM solver on known mono-exponential data."""

    def test_exact_recovery_clean_data(self):
        """LM should recover exact parameters from clean data."""
        TE = np.arange(10, 110, 10, dtype=np.float64)
        M0_true, T2_true = 1.0, 80.0
        signal = mono_exp_model(TE, M0_true, T2_true)

        M0_fit, T2_fit, converged = levenberg_marquardt_mono_exp(
            TE, signal, M0_init=0.8, T2_init=60.0,
        )
        assert converged
        assert abs(M0_fit - M0_true) < 1e-6
        assert abs(T2_fit - T2_true) < 1e-4

    def test_converges_from_poor_init(self):
        """LM should converge even with poor initialization."""
        TE = np.arange(10, 110, 10, dtype=np.float64)
        signal = mono_exp_model(TE, 1.5, 100.0)

        M0_fit, T2_fit, converged = levenberg_marquardt_mono_exp(
            TE, signal, M0_init=0.1, T2_init=10.0,
        )
        assert converged
        assert abs(T2_fit - 100.0) < 0.1

    def test_different_T2_values(self):
        """Should fit various T2 values correctly."""
        TE = np.arange(10, 110, 10, dtype=np.float64)
        for t2_true in [30.0, 80.0, 150.0, 500.0]:
            signal = mono_exp_model(TE, 1.0, t2_true)
            _, t2_fit, conv = levenberg_marquardt_mono_exp(
                TE, signal, 0.8, t2_true * 0.5,
            )
            assert conv
            assert abs(t2_fit - t2_true) / t2_true < 1e-3


class TestLogLinearFit:
    """Tests for log-linear T2 fitting."""

    def test_recovers_exact_T2_clean_data(self, fixtures_dir):
        """On clean data, log-linear fit should recover T2 exactly."""
        fix = np.load(os.path.join(fixtures_dir, 'solver_fixtures.npz'))
        signal = fix['input_signal_clean']
        TE = fix['input_TE']
        T2_true = fix['output_T2']

        T2_est, M0_est = fit_t2_loglinear(signal, TE)
        np.testing.assert_allclose(T2_est, T2_true, rtol=1e-6)

    def test_recovers_exact_M0_clean_data(self, fixtures_dir):
        """On clean data, log-linear fit should recover M0 exactly."""
        fix = np.load(os.path.join(fixtures_dir, 'solver_fixtures.npz'))
        signal = fix['input_signal_clean']
        TE = fix['input_TE']
        M0_true = fix['output_M0']

        T2_est, M0_est = fit_t2_loglinear(signal, TE)
        np.testing.assert_allclose(M0_est, M0_true, rtol=1e-6)

    def test_output_shape(self):
        """Output maps should match spatial dimensions."""
        signal = np.random.rand(16, 16, 10) + 0.1
        TE = np.arange(10, 110, 10, dtype=np.float64)
        T2, M0 = fit_t2_loglinear(signal, TE)
        assert T2.shape == (16, 16)
        assert M0.shape == (16, 16)

    def test_mask_excludes_background(self):
        """Masked-out pixels should remain zero."""
        TE = np.arange(10, 110, 10, dtype=np.float64)
        # Create signal with actual T2 decay
        T2_true = 80.0
        signal = np.exp(-TE[None, None, :] / T2_true) * np.ones((8, 8, 1))
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        T2, M0 = fit_t2_loglinear(signal, TE, mask=mask)
        assert T2[0, 0] == 0.0
        assert M0[0, 0] == 0.0
        assert T2[3, 3] > 0


class TestNonlinearFit:
    """Tests for nonlinear least-squares T2 fitting."""

    def test_recovers_exact_T2_clean_data(self, fixtures_dir):
        """On clean data, NLS should recover T2 exactly."""
        fix = np.load(os.path.join(fixtures_dir, 'solver_fixtures.npz'))
        signal = fix['input_signal_clean']
        TE = fix['input_TE']
        T2_true = fix['output_T2']

        T2_est, M0_est = fit_t2_nonlinear(signal, TE)
        np.testing.assert_allclose(T2_est, T2_true, rtol=1e-4)

    def test_better_than_loglinear_with_noise(self):
        """NLS should be at least as good as log-linear on noisy data."""
        rng = np.random.default_rng(42)
        TE = np.arange(10, 110, 10, dtype=np.float64)
        N = 16
        T2_true = np.full((N, N), 80.0)
        M0_true = np.full((N, N), 1.0)
        signal = M0_true[:,:,None] * np.exp(-TE[None,None,:] / T2_true[:,:,None])
        from src.physics_model import add_rician_noise
        signal_noisy = add_rician_noise(signal, sigma=0.03, rng=rng)

        T2_ll, _ = fit_t2_loglinear(signal_noisy, TE)
        T2_nls, _ = fit_t2_nonlinear(signal_noisy, TE, T2_init=T2_ll)

        rmse_ll = np.sqrt(np.mean((T2_ll - T2_true)**2))
        rmse_nls = np.sqrt(np.mean((T2_nls - T2_true)**2))
        assert rmse_nls <= rmse_ll * 1.1  # NLS should be comparable or better

    def test_different_T2_values(self):
        """Should correctly fit different T2 values across pixels."""
        TE = np.arange(10, 110, 10, dtype=np.float64)
        T2_vals = [40.0, 80.0, 150.0]
        signal = np.zeros((3, 1, 10))
        T2_true = np.zeros((3, 1))
        for i, t2 in enumerate(T2_vals):
            signal[i, 0, :] = np.exp(-TE / t2)
            T2_true[i, 0] = t2

        T2_est, _ = fit_t2_nonlinear(signal, TE)
        np.testing.assert_allclose(T2_est, T2_true, rtol=1e-3)
