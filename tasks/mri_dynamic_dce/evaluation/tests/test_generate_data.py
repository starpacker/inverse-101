"""Tests for src/generate_data.py"""

import os
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
sys.path.insert(0, TASK_DIR)

from src.generate_data import (
    gamma_variate, make_dynamic_phantom,
    generate_variable_density_mask, generate_dce_data,
)


@pytest.fixture
def fixtures():
    return np.load(os.path.join(TASK_DIR, 'evaluation', 'fixtures', 'generate_data.npz'))


class TestGammaVariate:
    def test_values(self, fixtures):
        t = fixtures['input_time']
        expected = fixtures['output_gamma_variate']
        result = gamma_variate(t, A=0.6, t_arrival=5.0, t_peak=12.0, alpha=3.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_zero_before_arrival(self):
        t = np.array([0, 1, 2, 3, 4])
        c = gamma_variate(t, A=1.0, t_arrival=5.0, t_peak=10.0, alpha=2.0)
        np.testing.assert_allclose(c, 0.0, atol=1e-15)

    def test_non_negative(self):
        t = np.linspace(0, 100, 200)
        c = gamma_variate(t, A=1.0, t_arrival=5.0, t_peak=10.0, alpha=2.0)
        assert np.all(c >= 0)

    def test_peak_near_t_peak(self):
        """Peak should occur near t_arrival + t_peak."""
        t = np.linspace(0, 60, 1000)
        c = gamma_variate(t, A=1.0, t_arrival=5.0, t_peak=12.0, alpha=3.0)
        peak_time = t[np.argmax(c)]
        # Peak should be near t_arrival + t_peak = 17
        assert abs(peak_time - 17.0) < 2.0


class TestDynamicPhantom:
    def test_shape(self):
        phantom, time_pts = make_dynamic_phantom(N=32, T=10)
        assert phantom.shape == (10, 32, 32)
        assert time_pts.shape == (10,)

    def test_non_negative(self):
        phantom, _ = make_dynamic_phantom(N=32, T=10)
        assert phantom.min() >= 0.0

    def test_temporal_variation(self):
        """Phantom should vary over time (dynamic regions)."""
        phantom, _ = make_dynamic_phantom(N=64, T=20)
        temporal_std = phantom.std(axis=0)
        assert temporal_std.max() > 0.01


class TestVariableDensityMask:
    def test_shape(self):
        mask = generate_variable_density_mask(64, sampling_rate=0.25, seed=0)
        assert mask.shape == (64, 64)

    def test_binary(self):
        mask = generate_variable_density_mask(64, sampling_rate=0.25, seed=0)
        assert set(np.unique(mask)).issubset({0.0, 1.0})

    def test_approximate_rate(self):
        mask = generate_variable_density_mask(128, sampling_rate=0.20, seed=0)
        actual_rate = mask.mean()
        assert abs(actual_rate - 0.20) < 0.05

    def test_center_sampled(self):
        """Center of k-space should be fully sampled."""
        N = 64
        mask = generate_variable_density_mask(N, sampling_rate=0.25,
                                              center_fraction=0.1, seed=0)
        n_center = max(int(0.1 * N), 2)
        c_start = (N - n_center) // 2
        c_end = c_start + n_center
        assert np.all(mask[c_start:c_end, c_start:c_end] == 1.0)

    def test_reproducible(self):
        m1 = generate_variable_density_mask(64, sampling_rate=0.25, seed=42)
        m2 = generate_variable_density_mask(64, sampling_rate=0.25, seed=42)
        np.testing.assert_array_equal(m1, m2)

    def test_different_seeds(self):
        m1 = generate_variable_density_mask(64, sampling_rate=0.25, seed=0)
        m2 = generate_variable_density_mask(64, sampling_rate=0.25, seed=1)
        assert not np.array_equal(m1, m2)


class TestGenerateData:
    def test_shapes(self):
        phantom, ksp, masks, time_pts = generate_dce_data(
            N=16, T=5, sampling_rate=0.5, noise_level=0.01, seed=0)
        assert phantom.shape == (5, 16, 16)
        assert ksp.shape == (5, 16, 16)
        assert masks.shape == (5, 16, 16)
        assert time_pts.shape == (5,)

    def test_kspace_complex(self):
        _, ksp, _, _ = generate_dce_data(N=16, T=5, seed=0)
        assert np.iscomplexobj(ksp)

    def test_kspace_zero_at_unsampled(self):
        """k-space should be zero where mask is zero."""
        _, ksp, masks, _ = generate_dce_data(N=16, T=5, seed=0)
        assert np.all(ksp[masks == 0] == 0)
