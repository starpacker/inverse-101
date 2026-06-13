"""Tests for the multi-echo MRI signal model."""

import os
import numpy as np
import pytest

from src.physics_model import mono_exponential_signal, add_rician_noise, simulate_multi_echo


class TestMonoExponentialSignal:
    """Tests for the deterministic forward model."""

    def test_known_values(self, fixtures_dir):
        """Signal model reproduces precomputed values exactly."""
        fix = np.load(os.path.join(fixtures_dir, 'physics_model_fixtures.npz'))
        M0 = fix['param_M0'][0]
        T2 = fix['param_T2'][0]
        TE = fix['input_TE']
        expected = fix['output_signal_clean']

        result = mono_exponential_signal(np.array(M0), np.array(T2), TE)
        np.testing.assert_allclose(result.ravel(), expected, rtol=1e-10)

    def test_zero_T2_gives_zero_signal(self):
        """Pixels with T2=0 should produce zero signal."""
        TE = np.array([10, 20, 30], dtype=np.float64)
        signal = mono_exponential_signal(
            np.array([1.0, 1.0]),
            np.array([0.0, 80.0]),
            TE,
        )
        assert signal.shape == (2, 3)
        np.testing.assert_array_equal(signal[0], 0.0)
        assert np.all(signal[1] > 0)

    def test_signal_decays_monotonically(self):
        """Signal should decrease with increasing TE."""
        TE = np.array([10, 20, 30, 40, 50], dtype=np.float64)
        signal = mono_exponential_signal(np.array(1.0), np.array(80.0), TE)
        diffs = np.diff(signal.ravel())
        assert np.all(diffs < 0)

    def test_at_TE_zero(self):
        """At TE=0, signal should equal M0."""
        TE = np.array([0.0], dtype=np.float64)
        M0 = np.array(2.5)
        signal = mono_exponential_signal(M0, np.array(80.0), TE)
        np.testing.assert_allclose(signal.ravel()[0], 2.5, rtol=1e-10)

    def test_batch_shape(self):
        """Output shape should be (..., N_echoes)."""
        M0 = np.ones((4, 5))
        T2 = np.full((4, 5), 80.0)
        TE = np.array([10, 20, 30])
        signal = mono_exponential_signal(M0, T2, TE)
        assert signal.shape == (4, 5, 3)


class TestRicianNoise:
    """Tests for Rician noise (statistical properties)."""

    def test_output_shape(self):
        """Output shape matches input."""
        signal = np.ones((10, 10, 5))
        noisy = add_rician_noise(signal, sigma=0.1, rng=np.random.default_rng(0))
        assert noisy.shape == signal.shape

    def test_non_negative(self):
        """Rician noise produces non-negative magnitudes."""
        signal = np.ones((100, 100))
        noisy = add_rician_noise(signal, sigma=0.5, rng=np.random.default_rng(0))
        assert np.all(noisy >= 0)

    def test_noise_increases_variance(self):
        """Noisy signal should have higher variance than clean."""
        signal = np.full((1000,), 1.0)
        noisy = add_rician_noise(signal, sigma=0.1, rng=np.random.default_rng(42))
        assert np.var(noisy) > np.var(signal)

    def test_zero_noise_returns_original(self):
        """With sigma=0, simulate_multi_echo should return clean signal."""
        M0 = np.array(1.0)
        T2 = np.array(80.0)
        TE = np.array([10, 20, 30], dtype=np.float64)
        clean = simulate_multi_echo(M0, T2, TE, sigma=0.0)
        expected = mono_exponential_signal(M0, T2, TE)
        np.testing.assert_allclose(clean, expected, rtol=1e-10)
