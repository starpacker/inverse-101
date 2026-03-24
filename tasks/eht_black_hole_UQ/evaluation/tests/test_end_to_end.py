"""
End-to-end integration test for DPI pipeline.
Uses reduced epochs to verify the full pipeline runs without errors.
"""

import os
import unittest
import numpy as np

TASK_DIR = os.path.join(os.path.dirname(__file__), "..", "..")


class TestEndToEnd(unittest.TestCase):
    """Integration test: full DPI pipeline with minimal training."""

    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, TASK_DIR)
        import torch
        from src.preprocessing import prepare_data, load_ground_truth
        from src.solvers import DPISolver

        (obs, obs_data, closure_indices, nufft_params,
         prior_image, flux_const, metadata) = prepare_data(
            os.path.join(TASK_DIR, "data"))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use reduced epochs for testing (enough to see loss decrease)
        solver = DPISolver(
            npix=metadata["npix"],
            n_flow=4,  # fewer flows for speed
            n_epoch=200,  # enough for loss to decrease
            batch_size=8,
            device=device,
        )

        cls.result = solver.reconstruct(
            obs_data, closure_indices, nufft_params, prior_image, flux_const)
        cls.posterior = solver.posterior_statistics(n_samples=20)
        cls.gt = load_ground_truth(
            os.path.join(TASK_DIR, "data"),
            metadata["npix"], metadata["fov_uas"])
        cls.metadata = metadata

    def test_loss_history_exists(self):
        self.assertIn('loss_history', self.result)
        self.assertEqual(len(self.result['loss_history']['total']), 200)

    def test_loss_decreases(self):
        """Loss should generally decrease (compare first 20 vs last 20 epochs)."""
        losses = self.result['loss_history']['total']
        initial_mean = np.mean(losses[:20])
        final_mean = np.mean(losses[-20:])
        self.assertLess(final_mean, initial_mean)

    def test_posterior_mean_shape(self):
        npix = self.metadata["npix"]
        self.assertEqual(self.posterior['mean'].shape, (npix, npix))

    def test_posterior_std_shape(self):
        npix = self.metadata["npix"]
        self.assertEqual(self.posterior['std'].shape, (npix, npix))

    def test_posterior_samples_shape(self):
        npix = self.metadata["npix"]
        self.assertEqual(self.posterior['samples'].shape, (20, npix, npix))

    def test_posterior_positivity(self):
        self.assertTrue(np.all(self.posterior['samples'] >= 0))

    def test_posterior_mean_positive(self):
        self.assertTrue(np.all(self.posterior['mean'] >= 0))

    def test_metrics_computable(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.visualization import compute_metrics
        metrics = compute_metrics(self.posterior['mean'], self.gt)
        self.assertIn('nrmse', metrics)
        self.assertIn('ncc', metrics)
        self.assertTrue(0 <= metrics['nrmse'] <= 10)


if __name__ == "__main__":
    unittest.main()
