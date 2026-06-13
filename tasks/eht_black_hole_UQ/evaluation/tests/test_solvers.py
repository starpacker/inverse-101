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


class TestImgLogscale(unittest.TestCase):
    """Test learnable log-scale factor."""

    def setUp(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.solvers import Img_logscale
        self.Img_logscale = Img_logscale

    def test_initial_scale_value(self):
        scale = self.Img_logscale(scale=0.5)
        output = torch.exp(scale.forward())
        np.testing.assert_allclose(output.item(), 0.5, rtol=1e-5)

    def test_initial_scale_one(self):
        scale = self.Img_logscale(scale=1.0)
        output = torch.exp(scale.forward())
        np.testing.assert_allclose(output.item(), 1.0, rtol=1e-5)

    def test_parameter_is_learnable(self):
        scale = self.Img_logscale(scale=1.0)
        params = list(scale.parameters())
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0].shape, (1,))

    def test_gradient_flows(self):
        scale = self.Img_logscale(scale=1.0)
        out = torch.exp(scale.forward())
        loss = (out - 2.0) ** 2
        loss.backward()
        self.assertIsNotNone(scale.log_scale.grad)
        self.assertNotEqual(scale.log_scale.grad.item(), 0.0)


class TestRealNVPFixture(unittest.TestCase):
    """Test RealNVP produces exact deterministic outputs with fixed weights."""

    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.solvers import RealNVP

        fixture_path = os.path.join(FIXTURE_DIR, "realnvp.npz")
        state_path = os.path.join(FIXTURE_DIR, "realnvp_state_dict.pt")
        if not os.path.exists(fixture_path):
            raise unittest.SkipTest("Solver fixtures not generated yet")

        cls.fixture = np.load(fixture_path, allow_pickle=False)
        cls.model = RealNVP(64, n_flow=4, affine=True, seqfrac=4)
        cls.model.load_state_dict(torch.load(state_path, map_location="cpu"))
        cls.model.eval()

    def test_reverse_output_values(self):
        z = torch.tensor(self.fixture['input_z'], dtype=torch.float32)
        with torch.no_grad():
            x, logdet = self.model.reverse(z)
        np.testing.assert_allclose(
            x.numpy(), self.fixture['output_reverse_x'], rtol=1e-5, atol=1e-6)

    def test_reverse_logdet_values(self):
        z = torch.tensor(self.fixture['input_z'], dtype=torch.float32)
        with torch.no_grad():
            _, logdet = self.model.reverse(z)
        np.testing.assert_allclose(
            logdet.numpy(), self.fixture['output_reverse_logdet'], rtol=1e-5, atol=1e-6)

    def test_forward_output_values(self):
        x = torch.tensor(self.fixture['output_reverse_x'], dtype=torch.float32)
        with torch.no_grad():
            z, logdet = self.model.forward(x)
        np.testing.assert_allclose(
            z.numpy(), self.fixture['output_forward_z'], rtol=1e-5, atol=1e-6)

    def test_forward_logdet_values(self):
        x = torch.tensor(self.fixture['output_reverse_x'], dtype=torch.float32)
        with torch.no_grad():
            _, logdet = self.model.forward(x)
        np.testing.assert_allclose(
            logdet.numpy(), self.fixture['output_forward_logdet'], rtol=1e-5, atol=1e-6)


class TestAffineCouplingFixture(unittest.TestCase):
    """Test AffineCoupling produces exact deterministic outputs with fixed weights."""

    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.solvers import AffineCoupling

        fixture_path = os.path.join(FIXTURE_DIR, "affine_coupling.npz")
        state_path = os.path.join(FIXTURE_DIR, "affine_coupling_state_dict.pt")
        if not os.path.exists(fixture_path):
            raise unittest.SkipTest("Solver fixtures not generated yet")

        cls.fixture = np.load(fixture_path, allow_pickle=False)
        cls.coupling = AffineCoupling(64, seqfrac=4, affine=True, batch_norm=True)
        cls.coupling.load_state_dict(torch.load(state_path, map_location="cpu"))
        cls.coupling.eval()

    def test_forward_output_values(self):
        x = torch.tensor(self.fixture['input_x'], dtype=torch.float32)
        with torch.no_grad():
            z, logdet = self.coupling(x)
        np.testing.assert_allclose(
            z.numpy(), self.fixture['output_z'], rtol=1e-5, atol=1e-6)

    def test_forward_logdet_values(self):
        x = torch.tensor(self.fixture['input_x'], dtype=torch.float32)
        with torch.no_grad():
            _, logdet = self.coupling(x)
        np.testing.assert_allclose(
            logdet.numpy(), self.fixture['output_logdet'], rtol=1e-5, atol=1e-6)


class TestImgLogscaleFixture(unittest.TestCase):
    """Test Img_logscale deterministic output from fixture."""

    @classmethod
    def setUpClass(cls):
        fixture_path = os.path.join(FIXTURE_DIR, "img_logscale.npz")
        if not os.path.exists(fixture_path):
            raise unittest.SkipTest("Solver fixtures not generated yet")
        cls.fixture = np.load(fixture_path, allow_pickle=False)

    def test_exp_logscale_value(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.solvers import Img_logscale
        scale_val = float(self.fixture['config_scale'])
        logscale = Img_logscale(scale=scale_val)
        output = torch.exp(logscale.forward()).item()
        np.testing.assert_allclose(
            output, float(self.fixture['output_exp_logscale']), rtol=1e-5)


class TestDPISolverReconstruct(unittest.TestCase):
    """Test DPI solver training (reduced epochs, auto-detect GPU)."""

    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.preprocessing import prepare_data
        from src.solvers import DPISolver

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        (obs, obs_data, closure_indices, nufft_params,
         prior_image, flux_const, metadata) = prepare_data(
            os.path.join(TASK_DIR, "data"))

        solver = DPISolver(
            npix=metadata["npix"],
            n_flow=4,
            n_epoch=200,
            batch_size=8,
            device=device,
        )

        cls.result = solver.reconstruct(
            obs_data, closure_indices, nufft_params, prior_image, flux_const)
        cls.solver = solver
        cls.metadata = metadata

    def test_loss_history_exists(self):
        self.assertIn('loss_history', self.result)

    def test_loss_history_length(self):
        self.assertEqual(len(self.result['loss_history']['total']), 200)

    def test_loss_history_keys(self):
        expected_keys = {'total', 'cphase', 'logca', 'visamp', 'logdet',
                         'flux', 'tsv', 'center', 'mem', 'l1'}
        self.assertEqual(set(self.result['loss_history'].keys()), expected_keys)

    def test_loss_decreases(self):
        """Loss should generally decrease (compare first 20 vs last 20 epochs)."""
        losses = self.result['loss_history']['total']
        initial_mean = np.mean(losses[:20])
        final_mean = np.mean(losses[-20:])
        self.assertLess(final_mean, initial_mean)

    def test_returns_model(self):
        self.assertIn('img_generator', self.result)
        self.assertIn('logscale_factor', self.result)

    def test_can_sample_after_training(self):
        samples = self.solver.sample(n_samples=5)
        npix = self.metadata['npix']
        self.assertEqual(samples.shape, (5, npix, npix))
        self.assertTrue(np.all(samples >= 0))


if __name__ == "__main__":
    unittest.main()
