"""Unit tests for the physics model module."""

import os
import sys

import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)

from src.physics_model import (
    radon_forward,
    radon_backproject,
    poisson_pre_log_model,
    simulate_poisson_noise,
    post_log_transform,
    compute_poisson_weights,
)

FIXTURES_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")


@pytest.fixture(scope="module")
def fixtures():
    return dict(np.load(os.path.join(FIXTURES_DIR, "physics_model_fixtures.npz")))


class TestRadonForward:
    """Tests for the forward projection operator."""

    def test_output_shape(self, fixtures):
        phantom = fixtures["param_phantom"]
        angles = fixtures["param_angles"]
        num_channels = int(fixtures["param_num_channels"])
        sino = radon_forward(phantom, angles, num_channels)
        assert sino.shape == (len(angles), num_channels)

    def test_deterministic(self, fixtures):
        """Forward projection is deterministic."""
        phantom = fixtures["param_phantom"]
        angles = fixtures["param_angles"]
        num_channels = int(fixtures["param_num_channels"])
        sino_expected = fixtures["output_forward_proj"]
        sino = radon_forward(phantom, angles, num_channels)
        np.testing.assert_allclose(sino, sino_expected, rtol=1e-10)

    def test_nonnegative(self, fixtures):
        """Sinogram of a nonnegative image should be nonnegative."""
        phantom = fixtures["param_phantom"]
        angles = fixtures["param_angles"]
        num_channels = int(fixtures["param_num_channels"])
        sino = radon_forward(phantom, angles, num_channels)
        assert np.all(sino >= 0)

    def test_zero_image(self, fixtures):
        """Projection of a zero image should be zero."""
        angles = fixtures["param_angles"]
        num_channels = int(fixtures["param_num_channels"])
        zero_img = np.zeros((256, 256))
        sino = radon_forward(zero_img, angles, num_channels)
        np.testing.assert_allclose(sino, 0.0, atol=1e-12)


class TestPoissonModel:
    """Tests for the Poisson noise model functions."""

    def test_transmission_values(self, fixtures):
        """poisson_pre_log_model should give I0 * exp(-sino)."""
        sino = fixtures["input_sinogram"]
        I0 = float(fixtures["param_I0"])
        expected = fixtures["output_transmission"]
        result = poisson_pre_log_model(sino, I0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_poisson_noise_statistical(self):
        """Poisson noise should have mean approximately equal to lambda."""
        rng = np.random.RandomState(123)
        lam = np.full((1000, 100), 500.0)
        counts = simulate_poisson_noise(lam, rng)
        # Mean should be close to 500
        assert abs(counts.mean() - 500.0) < 5.0
        # Variance should be close to 500 (Poisson property)
        assert abs(counts.var() - 500.0) < 30.0

    def test_poisson_noise_minimum(self):
        """Poisson counts should be clamped to >= 1."""
        rng = np.random.RandomState(0)
        lam = np.full((100,), 0.001)  # very low expected count
        counts = simulate_poisson_noise(lam, rng)
        assert np.all(counts >= 1.0)

    def test_post_log_roundtrip(self):
        """post_log of exact transmission should recover clean sinogram."""
        sino_clean = np.array([0.5, 1.0, 2.0, 3.0])
        I0 = 10000.0
        counts = I0 * np.exp(-sino_clean)
        recovered = post_log_transform(counts, I0)
        np.testing.assert_allclose(recovered, sino_clean, rtol=1e-10)

    def test_weights_equal_counts(self):
        """Poisson weights should equal the photon counts."""
        counts = np.array([100.0, 500.0, 1000.0, 50.0])
        weights = compute_poisson_weights(counts)
        np.testing.assert_array_equal(weights, counts)

    def test_weights_are_copy(self):
        """Modifying weights should not change the input counts."""
        counts = np.array([100.0, 200.0])
        weights = compute_poisson_weights(counts)
        weights[0] = 999.0
        assert counts[0] == 100.0


class TestBackproject:
    """Tests for the back-projection operator."""

    def test_output_shape(self, fixtures):
        sino = fixtures["input_sinogram"]
        angles = fixtures["param_angles"]
        bp = radon_backproject(sino, angles, 256, 256)
        assert bp.shape == (256, 256)

    def test_deterministic(self, fixtures):
        sino = fixtures["input_sinogram"]
        angles = fixtures["param_angles"]
        expected = fixtures["output_backproject"]
        bp = radon_backproject(sino, angles, 256, 256)
        np.testing.assert_allclose(bp, expected, rtol=1e-10)
