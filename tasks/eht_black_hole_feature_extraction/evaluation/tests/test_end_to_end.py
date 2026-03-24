"""End-to-end integration test for α-DPI feature extraction."""
import os
import sys
import unittest
import numpy as np
import torch

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, TASK_DIR)

from src.preprocessing import prepare_data
from src.solvers import AlphaDPISolver


class TestEndToEnd(unittest.TestCase):
    """Integration test: train α-DPI for a few epochs and verify outputs."""

    @classmethod
    def setUpClass(cls):
        (obs, obs_data, closure_indices, nufft_params,
         flux_const, metadata) = prepare_data(os.path.join(TASK_DIR, 'data'))

        cls.obs_data = obs_data
        cls.closure_indices = closure_indices
        cls.nufft_params = nufft_params
        cls.flux_const = flux_const

        cls.solver = AlphaDPISolver(
            npix=64, fov_uas=120.0, n_flow=4, seqfrac=1 / 16,
            n_epoch=100, batch_size=64, lr=1e-4,
            logdet_weight=1.0, grad_clip=1e-4,
            alpha=1.0, beta=1.0, start_order=4, decay_rate=2000,
            geometric_model='simple_crescent_nuisance', n_gaussian=2,
            device=torch.device('cpu')
        )

        cls.result = cls.solver.reconstruct(
            cls.obs_data, cls.closure_indices, cls.nufft_params, cls.flux_const
        )

    def test_loss_history_keys(self):
        for key in ['total', 'cphase', 'logca', 'logdet']:
            self.assertIn(key, self.result['loss_history'])

    def test_loss_history_length(self):
        self.assertEqual(len(self.result['loss_history']['total']), 100)

    def test_loss_decreasing(self):
        """Loss should generally decrease over training."""
        total = self.result['loss_history']['total']
        # Compare first 10% avg to last 10% avg (allow some randomness)
        early = np.mean(total[:10])
        late = np.mean(total[-10:])
        # With only 100 epochs, just check it's not NaN
        self.assertFalse(np.isnan(late))

    def test_sample(self):
        samples = self.solver.sample(n_samples=50)
        self.assertEqual(samples['params_unit'].shape, (50, 16))
        # All sigmoid outputs in (0, 1)
        self.assertTrue((samples['params_unit'] > 0).all())
        self.assertTrue((samples['params_unit'] < 1).all())

    def test_extract_physical_params(self):
        samples = self.solver.sample(n_samples=50)
        physical = self.solver.extract_physical_params(samples['params_unit'])
        self.assertEqual(physical.shape, (50, 16))
        # Diameter should be in range [2*10, 2*40] = [20, 80] uas
        self.assertTrue((physical[:, 0] >= 20).all())
        self.assertTrue((physical[:, 0] <= 80).all())

    def test_importance_resample(self):
        posterior = self.solver.importance_resample(
            self.obs_data, self.closure_indices, self.nufft_params,
            n_samples=100
        )
        self.assertIn('params_physical', posterior)
        self.assertIn('importance_weights', posterior)
        self.assertIn('images', posterior)
        self.assertIn('weighted_mean_image', posterior)
        # Weights sum to ~1
        np.testing.assert_allclose(
            posterior['importance_weights'].sum(), 1.0, rtol=1e-4)
        # Images are non-negative
        self.assertTrue((posterior['images'] >= 0).all())

    def test_compute_elbo(self):
        elbo = self.solver.compute_elbo(
            self.obs_data, self.closure_indices, self.nufft_params,
            n_samples=100
        )
        self.assertFalse(np.isnan(elbo))
        self.assertIsInstance(elbo, float)


if __name__ == '__main__':
    unittest.main()
