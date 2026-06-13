"""Unit tests for src/preprocessing.py"""

import os
import sys
import json
import numpy as np
import unittest

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIX_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures', 'preprocessing')
sys.path.insert(0, TASK_DIR)

from src.preprocessing import recorrupt, prctile_norm, extract_patches


def _cfg():
    with open(os.path.join(TASK_DIR, 'evaluation', 'fixtures', 'physics_model',
                           'config_noise.json')) as f:
        return json.load(f)


class TestRecorrupt(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIX_DIR, 'recorrupt.npz'))
        self.cfg = _cfg()

    def test_shapes(self):
        y = self.f['input_noisy']
        cfg = self.cfg
        y_hat, y_bar = recorrupt(y, cfg['beta1'], cfg['beta2'],
                                  alpha=cfg['alpha'], bg=cfg['bg'],
                                  filter_size=cfg['filter_size'],
                                  rng=np.random.default_rng(42))
        self.assertEqual(y_hat.shape, y.shape)
        self.assertEqual(y_bar.shape, y.shape)

    def test_deterministic(self):
        """Same seed → identical outputs."""
        y = self.f['input_noisy']
        cfg = self.cfg
        y_hat, y_bar = recorrupt(y, cfg['beta1'], cfg['beta2'],
                                  alpha=cfg['alpha'], bg=cfg['bg'],
                                  filter_size=cfg['filter_size'],
                                  rng=np.random.default_rng(42))
        np.testing.assert_allclose(y_hat, self.f['output_yhat'], rtol=1e-10)
        np.testing.assert_allclose(y_bar, self.f['output_ybar'], rtol=1e-10)

    def test_yhat_more_noisy_than_ybar(self):
        """y_hat = y + alpha*g has larger variance than y_bar = y - g/alpha (alpha > 1)."""
        y = self.f['input_noisy']
        cfg = self.cfg
        var_hat_list, var_bar_list = [], []
        for seed in range(20):
            yh, yb = recorrupt(y, cfg['beta1'], cfg['beta2'],
                                alpha=cfg['alpha'], bg=cfg['bg'],
                                filter_size=cfg['filter_size'],
                                rng=np.random.default_rng(seed))
            var_hat_list.append((yh - y).var())
            var_bar_list.append((yb - y).var())
        self.assertGreater(np.mean(var_hat_list), np.mean(var_bar_list))

    def test_mean_unbiased_yhat(self):
        """E[y_hat] ≈ y (noise has zero mean)."""
        y = self.f['input_noisy']
        cfg = self.cfg
        n_trials = 300
        acc = np.zeros_like(y)
        rng = np.random.default_rng(0)
        for _ in range(n_trials):
            yh, _ = recorrupt(y, cfg['beta1'], cfg['beta2'],
                               alpha=cfg['alpha'], bg=cfg['bg'],
                               filter_size=cfg['filter_size'], rng=rng)
            acc += yh
        np.testing.assert_allclose(acc / n_trials, y, rtol=0.05, atol=0.5)

    def test_mean_unbiased_ybar(self):
        """E[y_bar] ≈ y (noise has zero mean)."""
        y = self.f['input_noisy']
        cfg = self.cfg
        n_trials = 300
        acc = np.zeros_like(y)
        rng = np.random.default_rng(0)
        for _ in range(n_trials):
            _, yb = recorrupt(y, cfg['beta1'], cfg['beta2'],
                               alpha=cfg['alpha'], bg=cfg['bg'],
                               filter_size=cfg['filter_size'], rng=rng)
            acc += yb
        np.testing.assert_allclose(acc / n_trials, y, rtol=0.05, atol=0.5)


class TestPrctileNorm(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIX_DIR, 'prctile_norm.npz'))

    def test_range(self):
        x = self.f['input_image']
        result = prctile_norm(x)
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_dtype(self):
        x = self.f['input_image']
        result = prctile_norm(x)
        self.assertEqual(result.dtype, np.float32)

    def test_deterministic(self):
        x = self.f['input_image']
        result = prctile_norm(x)
        np.testing.assert_allclose(result, self.f['output_norm'], rtol=1e-6)

    def test_constant_image_returns_zeros(self):
        x = np.ones((10, 10)) * 5.0
        result = prctile_norm(x)
        np.testing.assert_allclose(result, np.zeros((10, 10), dtype=np.float32))

    def test_percentile_clipping(self):
        """With pmin=1, pmax=99, values outside that range are clipped to 0/1."""
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 100, (50, 50))
        result = prctile_norm(x, pmin=1, pmax=99)
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)


class TestExtractPatches(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIX_DIR, 'extract_patches.npz'))
        self.cfg = _cfg()

    def test_output_shapes(self):
        images = self.f['input_images']
        cfg = self.cfg
        y_hat, y_bar = extract_patches(images, patch_size=32, n_patches=12,
                                        beta1=cfg['beta1'], beta2=cfg['beta2'],
                                        alpha=cfg['alpha'], bg=cfg['bg'],
                                        filter_size=cfg['filter_size'], seed=7)
        self.assertEqual(y_hat.ndim, 4)
        self.assertEqual(y_hat.shape[1:], (1, 32, 32))
        self.assertEqual(y_hat.shape, y_bar.shape)

    def test_patch_shapes_match_fixture(self):
        images = self.f['input_images']
        cfg = self.cfg
        y_hat, y_bar = extract_patches(images, patch_size=32, n_patches=12,
                                        beta1=cfg['beta1'], beta2=cfg['beta2'],
                                        alpha=cfg['alpha'], bg=cfg['bg'],
                                        filter_size=cfg['filter_size'], seed=7)
        np.testing.assert_array_equal(y_hat.shape, self.f['output_yhat_shape'])
        np.testing.assert_array_equal(y_bar.shape, self.f['output_ybar_shape'])

    def test_patches_dtype(self):
        images = self.f['input_images']
        cfg = self.cfg
        y_hat, y_bar = extract_patches(images, patch_size=32, n_patches=12,
                                        beta1=cfg['beta1'], beta2=cfg['beta2'],
                                        alpha=cfg['alpha'], bg=cfg['bg'],
                                        filter_size=cfg['filter_size'], seed=7)
        self.assertEqual(y_hat.dtype, np.float32)
        self.assertEqual(y_bar.dtype, np.float32)

    def test_reproducible_with_seed(self):
        images = self.f['input_images']
        cfg = self.cfg
        y_hat1, y_bar1 = extract_patches(images, patch_size=32, n_patches=12,
                                          beta1=cfg['beta1'], beta2=cfg['beta2'],
                                          alpha=cfg['alpha'], bg=cfg['bg'],
                                          filter_size=cfg['filter_size'], seed=7)
        y_hat2, y_bar2 = extract_patches(images, patch_size=32, n_patches=12,
                                          beta1=cfg['beta1'], beta2=cfg['beta2'],
                                          alpha=cfg['alpha'], bg=cfg['bg'],
                                          filter_size=cfg['filter_size'], seed=7)
        np.testing.assert_array_equal(y_hat1, y_hat2)
        np.testing.assert_array_equal(y_bar1, y_bar2)


if __name__ == '__main__':
    unittest.main()
