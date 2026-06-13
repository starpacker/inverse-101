"""Integration tests: verify reference outputs meet expected quality bounds."""

import os
import sys
import json
import numpy as np
import unittest

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
REF_DIR = os.path.join(TASK_DIR, 'evaluation', 'reference_outputs')
EVAL_DIR = os.path.join(TASK_DIR, 'evaluation')
sys.path.insert(0, TASK_DIR)


class TestReferenceOutputsExist(unittest.TestCase):
    def test_metrics_json_exists(self):
        self.assertTrue(os.path.exists(os.path.join(EVAL_DIR, 'metrics.json')))

    def test_denoised_npy_exists(self):
        self.assertTrue(os.path.exists(os.path.join(REF_DIR, 'denoised.npy')))

    def test_deconvolved_npy_exists(self):
        self.assertTrue(os.path.exists(os.path.join(REF_DIR, 'deconvolved.npy')))

    def test_model_weights_exist(self):
        self.assertTrue(os.path.exists(os.path.join(REF_DIR, 'model_den_weights.pt')))
        self.assertTrue(os.path.exists(os.path.join(REF_DIR, 'model_dec_weights.pt')))

    def test_loss_history_exists(self):
        self.assertTrue(os.path.exists(os.path.join(REF_DIR, 'loss_history.npy')))


class TestOutputShapes(unittest.TestCase):
    def setUp(self):
        raw = np.load(os.path.join(TASK_DIR, 'data', 'raw_data.npz'))
        self.H, self.W = raw['measurements'].shape[1], raw['measurements'].shape[2]

    def test_denoised_shape(self):
        denoised = np.load(os.path.join(REF_DIR, 'denoised.npy'))
        self.assertEqual(denoised.shape, (self.H, self.W))

    def test_deconvolved_shape(self):
        deconvolved = np.load(os.path.join(REF_DIR, 'deconvolved.npy'))
        self.assertEqual(deconvolved.shape, (self.H, self.W))

    def test_loss_history_shape(self):
        loss_history = np.load(os.path.join(REF_DIR, 'loss_history.npy'))
        # Should have columns: (total, den, dec)
        self.assertEqual(loss_history.ndim, 2)
        self.assertEqual(loss_history.shape[1], 3)


class TestReferenceMetricsQuality(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(EVAL_DIR, 'metrics.json')) as f:
            self.metrics = json.load(f)

    def test_noise_reduction_factor(self):
        """Stage 1 should achieve at least 3× background noise reduction."""
        factor = self.metrics['stage1_denoising']['noise_reduction_factor']
        self.assertGreaterEqual(factor, 3.0,
            f"Noise reduction factor {factor:.2f} < 3.0×")

    def test_snr_improvement_factor(self):
        """Stage 1 should improve SNR by at least 3×."""
        factor = self.metrics['stage1_denoising']['snr_improvement_factor']
        self.assertGreaterEqual(factor, 3.0,
            f"SNR improvement factor {factor:.2f} < 3.0×")

    def test_sharpness_improvement_factor(self):
        """Stage 2 should improve Laplacian sharpness by at least 3×."""
        factor = self.metrics['stage2_deconvolution']['sharpness_improvement_factor']
        self.assertGreaterEqual(factor, 3.0,
            f"Sharpness improvement factor {factor:.2f} < 3.0×")

    def test_psf_residual_small(self):
        """PSF consistency residual should be below 0.05."""
        residual = self.metrics['stage2_deconvolution']['psf_residual']
        self.assertLess(residual, 0.05,
            f"PSF residual {residual:.4f} >= 0.05")

    def test_training_config_recorded(self):
        """Training hyperparameters should be recorded in metrics."""
        training = self.metrics['training']
        self.assertIn('n_iters', training)
        self.assertIn('patch_size', training)
        self.assertIn('lr_initial', training)


class TestWeightsLoadable(unittest.TestCase):
    def test_model_den_weights_loadable(self):
        import torch
        from src.solvers import UNet
        model = UNet(base=32)
        state = torch.load(os.path.join(REF_DIR, 'model_den_weights.pt'),
                           map_location='cpu', weights_only=True)
        model.load_state_dict(state)  # should not raise

    def test_model_dec_weights_loadable(self):
        import torch
        from src.solvers import UNet
        model = UNet(base=32)
        state = torch.load(os.path.join(REF_DIR, 'model_dec_weights.pt'),
                           map_location='cpu', weights_only=True)
        model.load_state_dict(state)  # should not raise


class TestLossDecreasing(unittest.TestCase):
    def test_loss_decreases_overall(self):
        """Total training loss should be lower at the end than at the start."""
        loss_history = np.load(os.path.join(REF_DIR, 'loss_history.npy'))
        first_quarter_mean = loss_history[:len(loss_history)//4, 0].mean()
        last_quarter_mean = loss_history[-len(loss_history)//4:, 0].mean()
        self.assertLess(last_quarter_mean, first_quarter_mean,
            f"Loss did not decrease: start={first_quarter_mean:.4f}, end={last_quarter_mean:.4f}")


if __name__ == '__main__':
    unittest.main()
