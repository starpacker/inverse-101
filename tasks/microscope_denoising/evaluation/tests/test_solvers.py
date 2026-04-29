"""Unit tests for src/solvers.py"""

import os
import sys
import numpy as np
import unittest
import torch

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIX_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures', 'solvers')
sys.path.insert(0, TASK_DIR)

from src.solvers import UNet, hessian_loss, train_zs_deconvnet, denoise_image, deconvolve_image


class TestUNet(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = UNet(base=8)
        self.model.eval()

    def test_output_shape_square(self):
        x = torch.zeros(2, 1, 64, 64)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (2, 1, 64, 64))

    def test_output_shape_non_square(self):
        x = torch.zeros(1, 1, 48, 80)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (1, 1, 48, 80))

    def test_output_shape_small(self):
        x = torch.zeros(1, 1, 32, 32)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (1, 1, 32, 32))

    def test_output_not_all_zeros(self):
        """Random weights should produce non-trivial output."""
        x = torch.ones(1, 1, 32, 32)
        with torch.no_grad():
            out = self.model(x)
        self.assertGreater(out.abs().max().item(), 0.0)

    def test_gradient_flows(self):
        """Loss backward should not raise and should produce non-zero gradients."""
        model = UNet(base=8)
        x = torch.randn(1, 1, 32, 32)
        target = torch.randn(1, 1, 32, 32)
        out = model(x)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        total_grad = sum(p.grad.abs().sum().item()
                         for p in model.parameters() if p.grad is not None)
        self.assertGreater(total_grad, 0.0)

    def test_base_channel_count(self):
        """base=32 model should have more parameters than base=8."""
        m8 = UNet(base=8)
        m32 = UNet(base=32)
        n8 = sum(p.numel() for p in m8.parameters())
        n32 = sum(p.numel() for p in m32.parameters())
        self.assertGreater(n32, n8)


class TestHessianLoss(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIX_DIR, 'hessian_loss.npz'))

    def test_nonneg(self):
        x = torch.randn(2, 1, 16, 16)
        loss = hessian_loss(x)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_zero_for_constant(self):
        """Constant image → all second differences = 0 → Hessian loss = 0."""
        x = torch.ones(1, 1, 16, 16)
        np.testing.assert_allclose(hessian_loss(x).item(), 0.0, atol=1e-10)

    def test_zero_for_linear(self):
        """Linearly varying image has zero second differences."""
        row = torch.arange(16, dtype=torch.float32)
        x = row.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, 16, 16)
        np.testing.assert_allclose(hessian_loss(x).item(), 0.0, atol=1e-10)

    def test_deterministic(self):
        x = torch.from_numpy(self.f['input_image'])
        result = hessian_loss(x).item()
        np.testing.assert_allclose(result, float(self.f['output_hessian_loss']), rtol=1e-6)

    def test_increases_with_noise(self):
        """Adding high-frequency noise should increase Hessian loss."""
        torch.manual_seed(0)
        x_smooth = torch.zeros(1, 1, 16, 16)
        x_noisy = torch.randn(1, 1, 16, 16) * 10.0
        self.assertGreater(hessian_loss(x_noisy).item(), hessian_loss(x_smooth).item())


class TestTrainZsDeconvnet(unittest.TestCase):
    """Smoke tests for training: verify output types, shapes, and loss decrease."""

    def setUp(self):
        rng = np.random.default_rng(0)
        N = 8
        self.y_hat = rng.uniform(0, 1, (N, 1, 32, 32)).astype(np.float32)
        self.y_bar = rng.uniform(0, 1, (N, 1, 32, 32)).astype(np.float32)
        self.psf = np.zeros((5, 5), dtype=np.float32)
        self.psf[2, 2] = 1.0  # delta PSF

    def test_returns_trained_models_and_history(self):
        model_den, model_dec, history = train_zs_deconvnet(
            self.y_hat, self.y_bar, self.psf,
            n_iters=200, batch_size=4, base=8, verbose=False)
        self.assertIsInstance(model_den, UNet)
        self.assertIsInstance(model_dec, UNet)
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)

    def test_history_tuples(self):
        _, _, history = train_zs_deconvnet(
            self.y_hat, self.y_bar, self.psf,
            n_iters=200, batch_size=4, base=8, verbose=False)
        total, den, dec = history[-1]
        self.assertGreater(total, 0.0)
        self.assertGreater(den, 0.0)
        self.assertGreater(dec, 0.0)

    def test_models_on_cpu_after_training(self):
        model_den, model_dec, _ = train_zs_deconvnet(
            self.y_hat, self.y_bar, self.psf,
            n_iters=200, batch_size=4, base=8, verbose=False)
        for p in model_den.parameters():
            self.assertEqual(p.device.type, 'cpu')
        for p in model_dec.parameters():
            self.assertEqual(p.device.type, 'cpu')


class TestInference(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        # Train a tiny model for inference tests
        N = 8
        y_hat = rng.uniform(0, 1, (N, 1, 32, 32)).astype(np.float32)
        y_bar = rng.uniform(0, 1, (N, 1, 32, 32)).astype(np.float32)
        psf = np.zeros((5, 5), dtype=np.float32)
        psf[2, 2] = 1.0
        self.model_den, self.model_dec, _ = train_zs_deconvnet(
            y_hat, y_bar, psf, n_iters=100, batch_size=4, base=8, verbose=False)
        self.y = rng.uniform(100, 120, (64, 64)).astype(np.float64)

    def test_denoise_output_shape(self):
        result = denoise_image(self.model_den, self.y, patch_size=32, overlap=8)
        self.assertEqual(result.shape, self.y.shape)

    def test_denoise_output_dtype(self):
        result = denoise_image(self.model_den, self.y, patch_size=32, overlap=8)
        self.assertEqual(result.dtype, np.float32)

    def test_deconvolve_output_shapes(self):
        denoised, deconvolved = deconvolve_image(
            self.model_den, self.model_dec, self.y, patch_size=32, overlap=8)
        self.assertEqual(denoised.shape, self.y.shape)
        self.assertEqual(deconvolved.shape, self.y.shape)


if __name__ == '__main__':
    unittest.main()
