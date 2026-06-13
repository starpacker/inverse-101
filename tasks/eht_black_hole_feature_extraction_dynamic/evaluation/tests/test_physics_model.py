"""Unit tests for physics_model module."""

import os
import sys
import unittest
import numpy as np

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, TASK_DIR)

import torch
from src.physics_model import (
    torch_complex_mul,
    SimpleCrescentParam2Img,
    SimpleCrescentNuisanceParam2Img,
    SimpleCrescentNuisanceFloorParam2Img,
    Loss_angle_diff,
    Loss_logca_diff2,
    Loss_visamp_diff,
)


class TestTorchComplexMul(unittest.TestCase):
    """Tests for torch_complex_mul."""

    def test_output_shape(self):
        """Output shape should match input batch shape."""
        B, M = 4, 8
        x = torch.randn(B, 2, M)  # (B, 2, M) where dim 1 is [real, imag]
        # Note: the function slices x[:, :, 0:1] and x[:, :, 1::],
        # splitting the M dim. So M must be 2 for the split to work.
        x2 = torch.randn(B, 2, 2)
        y = torch.randn(2, 2)
        out = torch_complex_mul(x2, y)
        self.assertEqual(out.shape[0], B)
        self.assertEqual(out.shape[1], 2)

    def test_real_multiplication(self):
        """Multiplying two purely real numbers: (a+0i)*(b+0i) = ab."""
        B = 2
        # x: shape (B, 2, 2) where [:, :, 0] is real, [:, :, 1] is imag
        x = torch.tensor([[[3.0, 0.0], [0.0, 0.0]],
                          [[2.0, 0.0], [0.0, 0.0]]])  # (2, 2, 2)
        y = torch.tensor([[4.0, 0.0], [0.0, 0.0]])  # (2, 2)
        out = torch_complex_mul(x, y)
        # real part should be 3*4=12, 2*4=8
        np.testing.assert_allclose(out[:, 0, 0].numpy(), [12.0, 8.0], rtol=1e-6)
        # imag part should be 0
        np.testing.assert_allclose(out[:, 1, 0].numpy(), [0.0, 0.0], atol=1e-6)

    def test_imaginary_multiplication(self):
        """Multiplying two purely imaginary numbers: (0+ai)*(0+bi) = -ab."""
        x = torch.tensor([[[0.0, 0.0], [3.0, 0.0]]])  # (1, 2, 2): real=0, imag=3
        y = torch.tensor([[0.0, 0.0], [2.0, 0.0]])     # (2, 2): real=0, imag=2
        out = torch_complex_mul(x, y)
        # (0+3i)*(0+2i) = -6 + 0i
        np.testing.assert_allclose(out[0, 0, 0].item(), -6.0, rtol=1e-6)
        np.testing.assert_allclose(out[0, 1, 0].item(), 0.0, atol=1e-6)


class TestSimpleCrescentParam2Img(unittest.TestCase):
    """Tests for SimpleCrescentParam2Img."""

    def setUp(self):
        self.npix = 16
        self.model = SimpleCrescentParam2Img(
            self.npix, fov=120, r_range=[10.0, 40.0], width_range=[1.0, 40.0])

    def test_output_shape(self):
        """Output should be (B, npix, npix)."""
        B = 4
        params = torch.rand(B, 4)
        img = self.model(params)
        self.assertEqual(img.shape, (B, self.npix, self.npix))

    def test_output_non_negative(self):
        """All pixel values should be non-negative."""
        params = torch.rand(2, 4)
        img = self.model(params)
        self.assertTrue(torch.all(img >= 0).item())

    def test_unit_normalization(self):
        """Each image should sum to approximately 1."""
        params = torch.rand(3, 4)
        img = self.model(params)
        sums = img.sum(dim=(-1, -2))
        np.testing.assert_allclose(sums.detach().numpy(), 1.0, atol=1e-3)

    def test_nparams(self):
        """Model should have 4 parameters."""
        self.assertEqual(self.model.nparams, 4)

    def test_compute_features_shapes(self):
        """compute_features should return tensors with correct broadcast shape."""
        params = torch.rand(5, 4)
        r, sigma, s, eta = self.model.compute_features(params)
        self.assertEqual(r.shape, (5, 1, 1))
        self.assertEqual(sigma.shape, (5, 1, 1))
        self.assertEqual(s.shape, (5, 1, 1))
        self.assertEqual(eta.shape, (5, 1, 1))

    def test_deterministic(self):
        """Same input should produce same output."""
        params = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        img1 = self.model(params)
        img2 = self.model(params)
        np.testing.assert_allclose(
            img1.detach().numpy(), img2.detach().numpy(), rtol=1e-12)


class TestSimpleCrescentNuisanceParam2Img(unittest.TestCase):
    """Tests for SimpleCrescentNuisanceParam2Img."""

    def setUp(self):
        self.npix = 16
        self.n_gaussian = 2
        self.model = SimpleCrescentNuisanceParam2Img(
            self.npix, n_gaussian=self.n_gaussian, fov=120)

    def test_nparams(self):
        """nparams should be 4 + 6 * n_gaussian."""
        expected = 4 + 6 * self.n_gaussian
        self.assertEqual(self.model.nparams, expected)

    def test_output_shape(self):
        """Output shape should be (B, npix, npix)."""
        B = 3
        params = torch.rand(B, self.model.nparams)
        img = self.model(params)
        self.assertEqual(img.shape, (B, self.npix, self.npix))

    def test_unit_normalization(self):
        """Each image should sum to approximately 1."""
        params = torch.rand(2, self.model.nparams)
        img = self.model(params)
        sums = img.sum(dim=(-1, -2))
        np.testing.assert_allclose(sums.detach().numpy(), 1.0, atol=1e-3)


class TestSimpleCrescentNuisanceFloorParam2Img(unittest.TestCase):
    """Tests for SimpleCrescentNuisanceFloorParam2Img."""

    def setUp(self):
        self.npix = 16
        self.n_gaussian = 1
        self.model = SimpleCrescentNuisanceFloorParam2Img(
            self.npix, n_gaussian=self.n_gaussian, fov=120)

    def test_nparams(self):
        """nparams should be 4 + 6 * n_gaussian + 2."""
        expected = 4 + 6 * self.n_gaussian + 2
        self.assertEqual(self.model.nparams, expected)

    def test_output_shape(self):
        """Output shape should be (B, npix, npix)."""
        B = 2
        params = torch.rand(B, self.model.nparams)
        img = self.model(params)
        self.assertEqual(img.shape, (B, self.npix, self.npix))

    def test_non_negative(self):
        """All pixel values should be non-negative."""
        params = torch.rand(3, self.model.nparams)
        img = self.model(params)
        self.assertTrue(torch.all(img >= 0).item())

    def test_unit_normalization(self):
        """Each image should sum to approximately 1."""
        params = torch.rand(2, self.model.nparams)
        img = self.model(params)
        sums = img.sum(dim=(-1, -2))
        np.testing.assert_allclose(sums.detach().numpy(), 1.0, atol=1e-3)


class TestLossAngleDiff(unittest.TestCase):
    """Tests for Loss_angle_diff."""

    def test_zero_loss_for_identical_input(self):
        """Loss should be zero when prediction matches truth."""
        device = torch.device('cpu')
        sigma = np.array([1.0, 1.0, 1.0])
        loss_fn = Loss_angle_diff(sigma, device)
        y = torch.tensor([[30.0, 60.0, 90.0]])
        loss = loss_fn(y, y)
        np.testing.assert_allclose(loss.item(), 0.0, atol=1e-6)

    def test_positive_loss_for_different_input(self):
        """Loss should be positive when prediction differs from truth."""
        device = torch.device('cpu')
        sigma = np.array([10.0, 10.0])
        loss_fn = Loss_angle_diff(sigma, device)
        y_true = torch.tensor([[0.0, 0.0]])
        y_pred = torch.tensor([[45.0, 90.0]])
        loss = loss_fn(y_true, y_pred)
        self.assertTrue(loss.item() > 0)

    def test_output_shape(self):
        """Output should have shape (B,)."""
        device = torch.device('cpu')
        sigma = np.array([5.0, 5.0, 5.0])
        loss_fn = Loss_angle_diff(sigma, device)
        B = 4
        y_true = torch.randn(B, 3)
        y_pred = torch.randn(B, 3)
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(loss.shape, (B,))


class TestLossLogcaDiff2(unittest.TestCase):
    """Tests for Loss_logca_diff2."""

    def test_zero_loss_for_identical_input(self):
        """Loss should be zero when prediction matches truth."""
        device = torch.device('cpu')
        sigma = np.array([0.1, 0.1])
        loss_fn = Loss_logca_diff2(sigma, device)
        y = torch.tensor([[0.5, -0.3]])
        loss = loss_fn(y, y)
        np.testing.assert_allclose(loss.item(), 0.0, atol=1e-6)

    def test_positive_loss_for_different_input(self):
        """Loss should be positive when prediction differs."""
        device = torch.device('cpu')
        sigma = np.array([0.1, 0.1])
        loss_fn = Loss_logca_diff2(sigma, device)
        y_true = torch.tensor([[0.0, 0.0]])
        y_pred = torch.tensor([[0.5, 0.5]])
        loss = loss_fn(y_true, y_pred)
        self.assertTrue(loss.item() > 0)


class TestLossVisampDiff(unittest.TestCase):
    """Tests for Loss_visamp_diff."""

    def test_zero_loss_for_identical_input(self):
        """Loss should be zero when prediction matches truth."""
        device = torch.device('cpu')
        sigma = np.array([0.01, 0.01, 0.01])
        loss_fn = Loss_visamp_diff(sigma, device)
        y = torch.tensor([[1.0, 2.0, 3.0]])
        loss = loss_fn(y, y)
        np.testing.assert_allclose(loss.item(), 0.0, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
