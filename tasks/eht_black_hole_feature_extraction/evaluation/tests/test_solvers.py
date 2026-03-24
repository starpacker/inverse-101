"""Tests for solver module."""
import os
import sys
import unittest
import numpy as np
import torch

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIX_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures', 'solvers')
sys.path.insert(0, TASK_DIR)

from src.solvers import (
    ActNorm, ZeroFC, AffineCoupling, Flow, RealNVP, AlphaDPISolver,
)


class TestActNorm(unittest.TestCase):
    def test_forward_reverse_invertible(self):
        actnorm = ActNorm()
        x = torch.randn(16, 10)
        y, logdet_fwd = actnorm(x)
        x_back, logdet_rev = actnorm.reverse(y)
        np.testing.assert_allclose(
            x.detach().numpy(), x_back.detach().numpy(), atol=1e-5)

    def test_logdet_sign(self):
        actnorm = ActNorm()
        x = torch.randn(16, 10)
        _, logdet_fwd = actnorm(x)
        self.assertEqual(logdet_fwd.shape, ())


class TestAffineCoupling(unittest.TestCase):
    def test_forward_reverse_invertible(self):
        coupling = AffineCoupling(16, seqfrac=1 / 16, affine=True, batch_norm=False)
        coupling.eval()
        x = torch.randn(8, 16)
        with torch.no_grad():
            y, logdet_fwd = coupling(x)
            x_back, logdet_rev = coupling.reverse(y)
        np.testing.assert_allclose(
            x.numpy(), x_back.numpy(), atol=1e-5)

    def test_output_shape(self):
        coupling = AffineCoupling(16, seqfrac=1 / 16)
        x = torch.randn(8, 16)
        y, logdet = coupling(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(logdet.shape, (8,))


class TestFlow(unittest.TestCase):
    def test_forward_reverse_invertible(self):
        flow = Flow(16, affine=True, seqfrac=1 / 16, batch_norm=False)
        flow.eval()
        x = torch.randn(8, 16)
        with torch.no_grad():
            y, logdet_fwd = flow(x)
            x_back, logdet_rev = flow.reverse(y)
        np.testing.assert_allclose(
            x.numpy(), x_back.numpy(), atol=1e-4)


class TestRealNVP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.load(os.path.join(FIX_DIR, 'realnvp.npz'))
        cls.flow = RealNVP(16, n_flow=4, affine=True, seqfrac=1 / 16,
                            permute='random', batch_norm=True)
        state_dict = torch.load(os.path.join(FIX_DIR, 'realnvp_state_dict.pt'),
                                 map_location='cpu', weights_only=True)
        cls.flow.load_state_dict(state_dict)
        cls.flow.eval()

    def test_forward_reverse_invertible(self):
        z = torch.tensor(self.data['input_z'])
        with torch.no_grad():
            out, logdet = self.flow.forward(z)
            z_back, logdet_back = self.flow.reverse(out)
        np.testing.assert_allclose(z.numpy(), z_back.numpy(), atol=1e-4)

    def test_forward_deterministic(self):
        z = torch.tensor(self.data['input_z'])
        with torch.no_grad():
            out, logdet = self.flow.forward(z)
        np.testing.assert_allclose(out.numpy(), self.data['output_forward'], rtol=1e-4)
        np.testing.assert_allclose(logdet.numpy(), self.data['output_logdet'], rtol=1e-4)

    def test_reverse_deterministic(self):
        out = torch.tensor(self.data['output_forward'])
        with torch.no_grad():
            z_back, logdet_back = self.flow.reverse(out)
        np.testing.assert_allclose(z_back.numpy(), self.data['output_reverse'], rtol=1e-4)

    def test_permutation_reproducibility(self):
        """Test that permutations are deterministic across instances."""
        flow2 = RealNVP(16, n_flow=4, affine=True, seqfrac=1 / 16,
                         permute='random', batch_norm=True)
        for i in range(len(self.flow.orders)):
            np.testing.assert_array_equal(self.flow.orders[i], flow2.orders[i])


class TestAlphaDPISolverSample(unittest.TestCase):
    """Test solver sampling (statistical checks, not exact values)."""

    def test_sample_shapes(self):
        solver = AlphaDPISolver(
            npix=64, fov_uas=120.0, n_flow=4, seqfrac=1 / 16,
            geometric_model='simple_crescent_nuisance', n_gaussian=2,
            device=torch.device('cpu')
        )
        # Build the model manually
        nparams = solver._build_geometric_model()
        solver.params_generator = RealNVP(
            nparams, 4, affine=True, seqfrac=1 / 16,
            permute='random', batch_norm=True
        ).to(solver.device)

        samples = solver.sample(n_samples=50)
        self.assertEqual(samples['params_unit'].shape, (50, 16))
        self.assertEqual(samples['params_samp'].shape, (50, 16))
        self.assertEqual(samples['z_samples'].shape, (50, 16))

    def test_params_unit_bounded(self):
        """Sigmoid output should be in (0, 1)."""
        solver = AlphaDPISolver(
            npix=64, fov_uas=120.0, n_flow=4, seqfrac=1 / 16,
            geometric_model='simple_crescent_nuisance', n_gaussian=2,
            device=torch.device('cpu')
        )
        nparams = solver._build_geometric_model()
        solver.params_generator = RealNVP(
            nparams, 4, affine=True, seqfrac=1 / 16
        ).to(solver.device)

        samples = solver.sample(n_samples=100)
        self.assertTrue((samples['params_unit'] > 0).all())
        self.assertTrue((samples['params_unit'] < 1).all())


class TestExtractPhysicalParams(unittest.TestCase):
    def test_crescent_params(self):
        solver = AlphaDPISolver(
            npix=64, fov_uas=120.0,
            geometric_model='simple_crescent', n_gaussian=0,
            r_range=[10.0, 40.0], width_range=[1.0, 40.0],
        )
        solver._build_geometric_model()

        # All params at 0.5
        params_unit = np.full((1, 4), 0.5)
        physical = solver.extract_physical_params(params_unit)
        self.assertEqual(physical.shape, (1, 4))

        # diameter = 2 * (10 + 0.5 * 30) = 50 uas
        np.testing.assert_allclose(physical[0, 0], 50.0, rtol=1e-10)
        # width = 1 + 0.5 * 39 = 20.5 uas
        np.testing.assert_allclose(physical[0, 1], 20.5, rtol=1e-10)
        # asymmetry = 0.5
        np.testing.assert_allclose(physical[0, 2], 0.5, rtol=1e-10)
        # PA at 0.5 → 0 degrees (mid-range)
        np.testing.assert_allclose(physical[0, 3], 0.0, atol=0.1)

    def test_nuisance_params(self):
        solver = AlphaDPISolver(
            npix=64, fov_uas=120.0,
            geometric_model='simple_crescent_nuisance', n_gaussian=2,
        )
        solver._build_geometric_model()

        params_unit = np.full((1, 16), 0.5)
        physical = solver.extract_physical_params(params_unit)
        # 4 crescent + 6*2 Gaussian = 16
        self.assertEqual(physical.shape, (1, 16))


if __name__ == '__main__':
    unittest.main()
