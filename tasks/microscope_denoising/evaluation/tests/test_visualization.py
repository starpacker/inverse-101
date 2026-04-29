"""Unit tests for src/visualization.py"""

import os
import sys
import numpy as np
import unittest

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIX_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures', 'visualization')
sys.path.insert(0, TASK_DIR)

from src.visualization import (compute_psnr, compute_ssim, compute_nrmse,
                                compute_snr_improvement, compute_all_metrics,
                                compute_psf_residual)


class TestComputePsnr(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIX_DIR, 'metrics.npz'))

    def test_deterministic(self):
        pred = self.f['input_pred']
        gt = self.f['input_gt']
        data_range = float(gt.max() - gt.min())
        result = compute_psnr(pred, gt, data_range)
        np.testing.assert_allclose(result, float(self.f['output_psnr']), rtol=1e-10)

    def test_positive(self):
        pred = self.f['input_pred']
        gt = self.f['input_gt']
        self.assertGreater(compute_psnr(pred, gt), 0.0)

    def test_perfect_prediction_high(self):
        """Identical images → very high (or infinite) PSNR."""
        x = np.random.default_rng(0).uniform(0, 1, (10, 10))
        psnr = compute_psnr(x, x, data_range=1.0)
        self.assertGreater(psnr, 100.0)

    def test_symmetric(self):
        """PSNR(pred, gt) with explicit data_range should be the same regardless of call order
        (since we pass data_range explicitly from gt)."""
        pred = self.f['input_pred']
        gt = self.f['input_gt']
        data_range = float(gt.max() - gt.min())
        r1 = compute_psnr(pred, gt, data_range)
        r2 = compute_psnr(pred, gt, data_range)
        self.assertAlmostEqual(r1, r2)


class TestComputeSsim(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIX_DIR, 'metrics.npz'))

    def test_deterministic(self):
        pred = self.f['input_pred']
        gt = self.f['input_gt']
        data_range = float(gt.max() - gt.min())
        result = compute_ssim(pred, gt, data_range)
        np.testing.assert_allclose(result, float(self.f['output_ssim']), rtol=1e-10)

    def test_range(self):
        pred = self.f['input_pred']
        gt = self.f['input_gt']
        ssim_val = compute_ssim(pred, gt)
        self.assertGreaterEqual(ssim_val, -1.0)
        self.assertLessEqual(ssim_val, 1.0)

    def test_perfect_prediction_is_one(self):
        x = np.random.default_rng(0).uniform(0, 1, (32, 32))
        np.testing.assert_allclose(compute_ssim(x, x, data_range=1.0), 1.0, atol=1e-5)


class TestComputeNrmse(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIX_DIR, 'metrics.npz'))

    def test_deterministic(self):
        pred = self.f['input_pred']
        gt = self.f['input_gt']
        result = compute_nrmse(pred, gt)
        np.testing.assert_allclose(result, float(self.f['output_nrmse']), rtol=1e-10)

    def test_nonneg(self):
        pred = self.f['input_pred']
        gt = self.f['input_gt']
        self.assertGreaterEqual(compute_nrmse(pred, gt), 0.0)

    def test_zero_for_perfect(self):
        x = np.ones((10, 10))
        self.assertAlmostEqual(compute_nrmse(x, x), 0.0)

    def test_worse_pred_higher_nrmse(self):
        rng = np.random.default_rng(0)
        gt = rng.uniform(0, 10, (20, 20))
        good = gt + rng.uniform(-0.1, 0.1, (20, 20))
        bad = gt + rng.uniform(-5.0, 5.0, (20, 20))
        self.assertLess(compute_nrmse(good, gt), compute_nrmse(bad, gt))


class TestComputeSnrImprovement(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.gt = rng.uniform(10, 20, (20, 20))
        self.noisy = self.gt + rng.uniform(-3, 3, (20, 20))
        self.denoised = self.gt + rng.uniform(-0.5, 0.5, (20, 20))

    def test_return_three_values(self):
        result = compute_snr_improvement(self.noisy, self.denoised, self.gt)
        self.assertEqual(len(result), 3)

    def test_improvement_positive(self):
        """Denoised should have higher PSNR than noisy → improvement > 0."""
        _, _, improvement = compute_snr_improvement(self.noisy, self.denoised, self.gt)
        self.assertGreater(improvement, 0.0)


class TestComputeAllMetrics(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIX_DIR, 'metrics.npz'))

    def test_keys_present(self):
        pred = self.f['input_pred']
        gt = self.f['input_gt']
        result = compute_all_metrics(pred, gt)
        for key in ('psnr', 'ssim', 'nrmse'):
            self.assertIn(key, result)

    def test_consistent_with_individual(self):
        pred = self.f['input_pred']
        gt = self.f['input_gt']
        metrics = compute_all_metrics(pred, gt)
        data_range = float(gt.max() - gt.min())
        np.testing.assert_allclose(metrics['psnr'],
                                    compute_psnr(pred, gt, data_range), rtol=1e-10)
        np.testing.assert_allclose(metrics['nrmse'],
                                    compute_nrmse(pred, gt), rtol=1e-10)


class TestComputePsfResidual(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIX_DIR, 'metrics.npz'))

    def test_deterministic(self):
        deconv = self.f['input_deconv']
        noisy = self.f['input_noisy']
        psf = self.f['input_psf']
        result = compute_psf_residual(deconv, noisy, psf)
        np.testing.assert_allclose(result, float(self.f['output_psf_residual']), rtol=1e-10)

    def test_nonneg(self):
        deconv = self.f['input_deconv']
        noisy = self.f['input_noisy']
        psf = self.f['input_psf']
        result = compute_psf_residual(deconv, noisy, psf)
        self.assertGreaterEqual(result, 0.0)

    def test_zero_for_delta_psf(self):
        """If PSF is a delta and deconvolved == noisy, residual should be near zero."""
        rng = np.random.default_rng(0)
        img = rng.uniform(100, 120, (16, 16))
        delta = np.zeros((5, 5), dtype=np.float32)
        delta[2, 2] = 1.0
        result = compute_psf_residual(img, img, delta)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
