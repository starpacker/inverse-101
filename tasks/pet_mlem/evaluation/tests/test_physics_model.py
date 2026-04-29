"""Tests for src/physics_model.py."""
import os
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")

import sys
sys.path.insert(0, TASK_DIR)

from src.physics_model import (
    pet_forward_project, pet_back_project, compute_sensitivity_image,
    add_poisson_noise, add_background,
)


@pytest.fixture
def fwd_fix():
    return np.load(os.path.join(FIXTURE_DIR, "physics_model_forward.npz"))


@pytest.fixture
def bp_fix():
    return np.load(os.path.join(FIXTURE_DIR, "physics_model_backproject.npz"))


@pytest.fixture
def sens_fix():
    return np.load(os.path.join(FIXTURE_DIR, "physics_model_sensitivity.npz"))


@pytest.fixture
def bg_fix():
    return np.load(os.path.join(FIXTURE_DIR, "physics_model_background.npz"))


class TestForwardProject:
    def test_fixture_match(self, fwd_fix):
        sino = pet_forward_project(fwd_fix["input_image"], fwd_fix["input_theta"])
        np.testing.assert_allclose(sino, fwd_fix["output_sinogram"], rtol=1e-10)

    def test_zero_image(self):
        theta = np.linspace(0, 180, 20, endpoint=False)
        sino = pet_forward_project(np.zeros((32, 32)), theta)
        np.testing.assert_allclose(sino, 0, atol=1e-10)

    def test_positive_for_positive_image(self, fwd_fix):
        sino = pet_forward_project(fwd_fix["input_image"], fwd_fix["input_theta"])
        assert np.all(sino >= 0)
        assert np.sum(sino) > 0

    def test_deterministic(self, fwd_fix):
        s1 = pet_forward_project(fwd_fix["input_image"], fwd_fix["input_theta"])
        s2 = pet_forward_project(fwd_fix["input_image"], fwd_fix["input_theta"])
        np.testing.assert_array_equal(s1, s2)

    def test_shape(self):
        theta = np.linspace(0, 180, 30, endpoint=False)
        sino = pet_forward_project(np.ones((64, 64)), theta)
        assert sino.ndim == 2
        assert sino.shape[1] == 30


class TestBackProject:
    def test_fixture_match(self, bp_fix):
        bp = pet_back_project(bp_fix["input_sinogram"], bp_fix["input_theta"],
                               int(bp_fix["param_N"]))
        np.testing.assert_allclose(bp, bp_fix["output_image"], rtol=1e-10)

    def test_shape(self):
        theta = np.linspace(0, 180, 20, endpoint=False)
        sino = pet_forward_project(np.ones((32, 32)), theta)
        bp = pet_back_project(sino, theta, 32)
        assert bp.shape == (32, 32)

    def test_positive_for_positive_sinogram(self, bp_fix):
        bp = pet_back_project(bp_fix["input_sinogram"], bp_fix["input_theta"],
                               int(bp_fix["param_N"]))
        assert np.any(bp > 0)


class TestSensitivity:
    def test_fixture_match(self, sens_fix):
        s = compute_sensitivity_image(sens_fix["input_theta"],
                                       int(sens_fix["param_N"]))
        np.testing.assert_allclose(s, sens_fix["output_sensitivity"], rtol=1e-10)

    def test_positive_inside_fov(self):
        theta = np.linspace(0, 180, 30, endpoint=False)
        s = compute_sensitivity_image(theta, 32)
        assert s[16, 16] > 0

    def test_shape(self):
        theta = np.linspace(0, 180, 60, endpoint=False)
        s = compute_sensitivity_image(theta, 64)
        assert s.shape == (64, 64)


class TestPoissonNoise:
    def test_non_negative(self):
        sino = np.ones((20, 30)) * 10
        noisy = add_poisson_noise(sino, scale=100, rng=np.random.default_rng(42))
        assert np.all(noisy >= 0)

    def test_shape(self):
        sino = np.ones((20, 30))
        noisy = add_poisson_noise(sino, scale=100, rng=np.random.default_rng(42))
        assert noisy.shape == sino.shape

    def test_mean_approximately_correct(self):
        """Mean of Poisson samples should be close to expected."""
        sino = np.ones((100, 100)) * 5.0
        noisy = add_poisson_noise(sino, scale=1000, rng=np.random.default_rng(42))
        assert abs(np.mean(noisy) / 1000 - 5.0) < 0.5

    def test_reproducible(self):
        sino = np.ones((20, 30)) * 10
        n1 = add_poisson_noise(sino, scale=100, rng=np.random.default_rng(42))
        n2 = add_poisson_noise(sino, scale=100, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(n1, n2)


class TestBackground:
    def test_fixture_match(self, bg_fix):
        sino_bg, bg = add_background(bg_fix["input_sinogram"],
                                      randoms_fraction=float(bg_fix["param_randoms_fraction"]))
        np.testing.assert_allclose(sino_bg, bg_fix["output_sino_with_bg"], rtol=1e-10)
        np.testing.assert_allclose(bg, bg_fix["output_background"], rtol=1e-10)

    def test_adds_positive(self):
        sino = np.ones((20, 30)) * 10
        sino_bg, bg = add_background(sino, randoms_fraction=0.1)
        assert np.all(sino_bg >= sino)
        assert np.all(bg > 0)

    def test_background_level(self):
        sino = np.ones((20, 30)) * 100
        _, bg = add_background(sino, randoms_fraction=0.2)
        expected_bg = 0.2 * 100
        np.testing.assert_allclose(bg[0, 0], expected_bg, rtol=0.01)
