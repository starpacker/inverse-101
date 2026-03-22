"""
Unit tests for solvers.py — Real-NVP architecture and DPI solver
"""

import os
import unittest
import numpy as np
import torch

TASK_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "solvers")


class TestActNorm(unittest.TestCase):
    """Test ActNorm forward-reverse invertibility."""

    def setUp(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.solvers import ActNorm
        self.actnorm = ActNorm()
        torch.manual_seed(42)
        self.x = torch.randn(8, 64)

    def test_forward_reverse_invertibility(self):
        # Forward pass (initializes)
        z, logdet_fwd = self.actnorm(self.x)
        # Reverse pass
        x_recon, logdet_rev = self.actnorm.reverse(z)
        np.testing.assert_allclose(
            x_recon.detach().numpy(), self.x.numpy(), rtol=1e-5, atol=1e-6)

    def test_logdet_consistency(self):
        z, logdet_fwd = self.actnorm(self.x)
        _, logdet_rev = self.actnorm.reverse(z)
        np.testing.assert_allclose(
            (logdet_fwd + logdet_rev).item(), 0.0, atol=1e-4)


class TestAffineCoupling(unittest.TestCase):
    """Test AffineCoupling forward-reverse invertibility."""

    def setUp(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.solvers import AffineCoupling
        self.coupling = AffineCoupling(64, seqfrac=4, affine=True, batch_norm=True)
        self.coupling.eval()
        torch.manual_seed(42)
        self.x = torch.randn(8, 64)

    def test_forward_reverse_invertibility(self):
        z, logdet_fwd = self.coupling(self.x)
        x_recon, logdet_rev = self.coupling.reverse(z)
        np.testing.assert_allclose(
            x_recon.detach().numpy(), self.x.numpy(), rtol=1e-5, atol=1e-6)

    def test_logdet_consistency(self):
        z, logdet_fwd = self.coupling(self.x)
        _, logdet_rev = self.coupling.reverse(z)
        np.testing.assert_allclose(
            (logdet_fwd + logdet_rev).detach().numpy(),
            np.zeros(8), atol=1e-4)


class TestFlow(unittest.TestCase):
    """Test Flow block forward-reverse invertibility."""

    def setUp(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.solvers import Flow
        self.flow = Flow(64, affine=True, seqfrac=4, batch_norm=True)
        self.flow.eval()
        torch.manual_seed(42)
        self.x = torch.randn(8, 64)

    def test_forward_reverse_invertibility(self):
        z, _ = self.flow(self.x)
        x_recon, _ = self.flow.reverse(z)
        np.testing.assert_allclose(
            x_recon.detach().numpy(), self.x.numpy(), rtol=1e-4, atol=1e-5)


class TestRealNVP(unittest.TestCase):
    """Test RealNVP forward-reverse invertibility."""

    def setUp(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.solvers import RealNVP
        self.model = RealNVP(64, n_flow=4, affine=True, seqfrac=4)
        self.model.eval()
        torch.manual_seed(42)
        self.z = torch.randn(8, 64)

    def test_forward_reverse_invertibility(self):
        """z → forward → reverse → z' ≈ z"""
        x, logdet_fwd = self.model.forward(self.z)
        z_recon, logdet_rev = self.model.reverse(x)
        # Use reverse(forward(z)) since training uses reverse direction
        np.testing.assert_allclose(
            z_recon.detach().numpy(), self.z.numpy(), rtol=1e-3, atol=1e-4)

    def test_reverse_forward_invertibility(self):
        """z → reverse → forward → z' ≈ z"""
        x, logdet_rev = self.model.reverse(self.z)
        z_recon, logdet_fwd = self.model.forward(x)
        np.testing.assert_allclose(
            z_recon.detach().numpy(), self.z.numpy(), rtol=1e-3, atol=1e-4)

    def test_output_shape(self):
        x, logdet = self.model.reverse(self.z)
        self.assertEqual(x.shape, self.z.shape)
        self.assertEqual(logdet.shape, (8,))

    def test_permutation_reproducibility(self):
        """Same n_flow → same permutations."""
        from src.solvers import RealNVP
        model2 = RealNVP(64, n_flow=4)
        for o1, o2 in zip(self.model.orders, model2.orders):
            np.testing.assert_array_equal(o1, o2)


class TestDPISolverSample(unittest.TestCase):
    """Test DPI solver sampling (statistical, no training quality check)."""

    def setUp(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.solvers import DPISolver, RealNVP, Img_logscale

        self.solver = DPISolver(npix=8, n_flow=2, device=torch.device("cpu"))
        # Manually set up a tiny model for testing
        self.solver.img_generator = RealNVP(64, 2, affine=True, seqfrac=4).to("cpu")
        self.solver.logscale_factor = Img_logscale(scale=0.01).to("cpu")

    def test_sample_shape(self):
        samples = self.solver.sample(n_samples=10)
        self.assertEqual(samples.shape, (10, 8, 8))

    def test_sample_positivity(self):
        """Softplus ensures non-negative images."""
        samples = self.solver.sample(n_samples=10)
        self.assertTrue(np.all(samples >= 0))

    def test_posterior_statistics_keys(self):
        stats = self.solver.posterior_statistics(n_samples=10)
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('samples', stats)

    def test_posterior_mean_shape(self):
        stats = self.solver.posterior_statistics(n_samples=10)
        self.assertEqual(stats['mean'].shape, (8, 8))
        self.assertEqual(stats['std'].shape, (8, 8))


if __name__ == "__main__":
    unittest.main()
