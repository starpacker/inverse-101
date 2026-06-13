"""
Tests for src/solvers.py
"""

import os
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.solvers import (
    positional_encoding, MLP, BHNeRFModel, loss_fn_image,
    loss_fn_lightcurve, loss_fn_visibility, BHNeRFSolver,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'solvers')


class TestPositionalEncoding(unittest.TestCase):

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, 'positional_encoding.npz'))

    def test_output_shape(self):
        x = torch.tensor(self.f['input_x'])
        deg = int(self.f['param_deg'])
        encoded = positional_encoding(x, deg)
        D = x.shape[-1]
        expected_dim = D + 2 * D * deg
        self.assertEqual(encoded.shape[-1], expected_dim)

    def test_output_values(self):
        x = torch.tensor(self.f['input_x'])
        deg = int(self.f['param_deg'])
        encoded = positional_encoding(x, deg)
        np.testing.assert_allclose(encoded.numpy(), self.f['output_encoded'],
                                   rtol=1e-5, atol=1e-6)

    def test_identity_deg0(self):
        x = torch.randn(5, 3)
        encoded = positional_encoding(x, deg=0)
        np.testing.assert_allclose(encoded.numpy(), x.numpy())


class TestMLP(unittest.TestCase):

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, 'mlp_forward.npz'))

    def test_output_shape(self):
        in_features = int(self.f['param_in_features'])
        torch.manual_seed(42)
        mlp = MLP(in_features, net_depth=4, net_width=128, out_channel=1)
        x = torch.randn(10, in_features)
        with torch.no_grad():
            out = mlp(x)
        self.assertEqual(out.shape, (10, 1))

    def test_output_determinism(self):
        in_features = int(self.f['param_in_features'])
        # Load state dict from fixture
        state_dict = {}
        for key in self.f.files:
            if key.startswith('state_'):
                param_name = key[6:]  # Remove 'state_' prefix
                state_dict[param_name] = torch.tensor(self.f[key])

        torch.manual_seed(42)
        mlp = MLP(in_features, net_depth=4, net_width=128, out_channel=1)
        mlp.load_state_dict(state_dict)
        mlp.eval()

        x = torch.tensor(self.f['input_x'])
        with torch.no_grad():
            out = mlp(x)
        np.testing.assert_allclose(out.numpy(), self.f['output_y'],
                                   rtol=1e-5, atol=1e-6)


class TestBHNeRFModel(unittest.TestCase):

    def test_output_shape(self):
        torch.manual_seed(42)
        model = BHNeRFModel(scale=12.0, rmin=6.0, rmax=12.0, z_width=4.0,
                            posenc_deg=3, net_depth=4, net_width=128)
        coords = torch.randn(3, 4, 4, 5)
        Omega = torch.abs(torch.randn(4, 4, 5)) * 0.01
        t_geo = torch.randn(4, 4, 5) * 0.1
        rot_axis = torch.tensor([0.0, 0.0, 1.0])
        with torch.no_grad():
            emission = model(0.0, coords, Omega, 0.0, t_geo, 0.0, rot_axis)
        self.assertEqual(emission.shape, (4, 4, 5))

    def test_output_nonnegative(self):
        torch.manual_seed(42)
        model = BHNeRFModel(scale=12.0, rmin=0.0, rmax=100.0, z_width=100.0)
        coords = torch.randn(3, 4, 4, 5)
        Omega = torch.zeros(4, 4, 5)
        t_geo = torch.zeros(4, 4, 5)
        rot_axis = torch.tensor([0.0, 0.0, 1.0])
        with torch.no_grad():
            emission = model(0.0, coords, Omega, 0.0, t_geo, 0.0, rot_axis)
        self.assertTrue((emission >= 0).all())


class TestLossFunctions(unittest.TestCase):

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, 'loss_fn_image.npz'))

    def test_loss_fn_image_value(self):
        pred = torch.tensor(self.f['input_pred'])
        target = torch.tensor(self.f['input_target'])
        sigma = float(self.f['param_sigma'])
        loss = loss_fn_image(pred, target, sigma)
        np.testing.assert_allclose(loss.item(), float(self.f['output_loss']),
                                   rtol=1e-5)

    def test_loss_fn_image_zero(self):
        x = torch.randn(8, 8)
        loss = loss_fn_image(x, x, sigma=1.0)
        np.testing.assert_allclose(loss.item(), 0.0, atol=1e-10)

    def test_loss_fn_lightcurve_shape(self):
        pred = torch.randn(5, 8, 8)
        target = torch.randn(5)
        loss = loss_fn_lightcurve(pred, target)
        self.assertEqual(loss.shape, ())

    def test_loss_fn_visibility_shape(self):
        pred = torch.randn(5, 10) + 1j * torch.randn(5, 10)
        target = torch.randn(5, 10) + 1j * torch.randn(5, 10)
        sigma = torch.ones(5, 10)
        loss = loss_fn_visibility(pred, target, sigma)
        self.assertEqual(loss.shape, ())


class TestBHNeRFSolver(unittest.TestCase):

    def test_training_reduces_loss(self):
        """Training should reduce loss (statistical test)."""
        from src.preprocessing import load_observation, load_metadata

        metadata = load_metadata("data")
        # Reduce for faster test
        metadata['n_iters'] = 50
        metadata['batch_size'] = 4

        obs_data = load_observation("data")

        solver = BHNeRFSolver(metadata, device='cpu')
        result = solver.reconstruct(obs_data, seed=42)

        loss_history = result['loss_history']
        # Average first 5 losses should be > average last 5 losses
        early = np.mean(loss_history[:5])
        late = np.mean(loss_history[-5:])
        self.assertLess(late, early,
                        f"Loss did not decrease: early={early:.4f}, late={late:.4f}")


if __name__ == '__main__':
    unittest.main()
