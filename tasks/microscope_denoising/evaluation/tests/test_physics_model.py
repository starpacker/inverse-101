"""Unit tests for src/physics_model.py"""

import os
import sys
import json
import numpy as np
import unittest

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIX_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures', 'physics_model')
sys.path.insert(0, TASK_DIR)

from src.physics_model import noise_variance, add_noise, psf_convolve, estimate_noise_params


def _cfg():
    with open(os.path.join(FIX_DIR, 'config_noise.json')) as f:
        return json.load(f)


class TestNoiseVariance(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIX_DIR, 'noise_variance.npz'))
        self.cfg = _cfg()

    def test_shape(self):
        x = self.f['input_image']
        cfg = self.cfg
        result = noise_variance(x, cfg['beta1'], cfg['beta2'], cfg['bg'], cfg['filter_size'])
        self.assertEqual(result.shape, x.shape)

    def test_nonneg(self):
        x = self.f['input_image']
        cfg = self.cfg
        result = noise_variance(x, cfg['beta1'], cfg['beta2'], cfg['bg'], cfg['filter_size'])
        self.assertTrue(np.all(result >= 0))

    def test_deterministic(self):
        x = self.f['input_image']
        cfg = self.cfg
        result = noise_variance(x, cfg['beta1'], cfg['beta2'], cfg['bg'], cfg['filter_size'])
        np.testing.assert_allclose(result, self.f['output_noise_variance'], rtol=1e-10)

    def test_increases_with_signal(self):
        """Pixels with higher signal above bg should have higher variance (beta1 > 0)."""
        cfg = self.cfg
        x_low = np.full((16, 16), cfg['bg'] + 1.0)
        x_high = np.full((16, 16), cfg['bg'] + 20.0)
        var_low = noise_variance(x_low, cfg['beta1'], cfg['beta2'], cfg['bg'], cfg['filter_size'])
        var_high = noise_variance(x_high, cfg['beta1'], cfg['beta2'], cfg['bg'], cfg['filter_size'])
        self.assertTrue(np.all(var_high > var_low))


class TestAddNoise(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIX_DIR, 'noise_variance.npz'))
        self.cfg = _cfg()

    def test_shape(self):
        x = self.f['input_image']
        cfg = self.cfg
        result = add_noise(x, cfg['beta1'], cfg['beta2'], cfg['bg'], cfg['filter_size'],
                           rng=np.random.default_rng(0))
        self.assertEqual(result.shape, x.shape)

    def test_unbiased_over_trials(self):
        """Average of many noisy realisations should approximate the clean image."""
        x = self.f['input_image']
        cfg = self.cfg
        rng = np.random.default_rng(0)
        n_trials = 200
        stack = np.stack([
            add_noise(x, cfg['beta1'], cfg['beta2'], cfg['bg'], cfg['filter_size'], rng=rng)
            for _ in range(n_trials)
        ])
        np.testing.assert_allclose(stack.mean(axis=0), x, rtol=0.05, atol=0.5)


class TestPsfConvolve(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIX_DIR, 'psf_convolve.npz'))

    def test_shape(self):
        x = self.f['input_image']
        psf = self.f['input_psf']
        result = psf_convolve(x, psf)
        self.assertEqual(result.shape, x.shape)

    def test_deterministic(self):
        x = self.f['input_image']
        psf = self.f['input_psf']
        result = psf_convolve(x, psf)
        np.testing.assert_allclose(result, self.f['output_psf_convolve'], rtol=1e-10)

    def test_energy_conservation(self):
        """PSF sums to 1, so convolution should approximately preserve the image mean."""
        x = self.f['input_image']
        psf = self.f['input_psf']
        result = psf_convolve(x, psf)
        np.testing.assert_allclose(result.mean(), x.mean(), rtol=0.01)

    def test_identity_psf(self):
        """A delta PSF should leave the image unchanged."""
        x = self.f['input_image']
        delta = np.zeros((5, 5), dtype=np.float32)
        delta[2, 2] = 1.0
        result = psf_convolve(x, delta)
        np.testing.assert_allclose(result, x, rtol=1e-10)


class TestEstimateNoiseParams(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIX_DIR, 'estimate_noise_params.npz'))
        self.cfg = _cfg()

    def test_returns_positive(self):
        y = self.f['input_noisy']
        cfg = self.cfg
        b1, b2 = estimate_noise_params(y, bg=cfg['bg'], filter_size=cfg['filter_size'])
        self.assertGreater(b1, 0.0)
        self.assertGreater(b2, 0.0)

    def test_deterministic(self):
        """Called twice on same input should return same result."""
        y = self.f['input_noisy']
        cfg = self.cfg
        b1a, b2a = estimate_noise_params(y, bg=cfg['bg'], filter_size=cfg['filter_size'])
        b1b, b2b = estimate_noise_params(y, bg=cfg['bg'], filter_size=cfg['filter_size'])
        self.assertAlmostEqual(b1a, b1b)
        self.assertAlmostEqual(b2a, b2b)

    def test_consistent_with_reference(self):
        """Estimated values should be close to the stored reference."""
        y = self.f['input_noisy']
        cfg = self.cfg
        b1, b2 = estimate_noise_params(y, bg=cfg['bg'], filter_size=cfg['filter_size'])
        np.testing.assert_allclose(b1, float(self.f['output_beta1']), rtol=1e-10)
        np.testing.assert_allclose(b2, float(self.f['output_beta2']), rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
