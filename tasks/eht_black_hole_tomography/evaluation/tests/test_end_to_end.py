"""
End-to-end tests for the BH-NeRF pipeline.
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestDataGeneration(unittest.TestCase):
    """Test that synthetic data was generated correctly."""

    def test_raw_data_exists(self):
        self.assertTrue(os.path.exists("data/raw_data.npz"))

    def test_meta_data_exists(self):
        self.assertTrue(os.path.exists("data/meta_data"))

    def test_raw_data_keys(self):
        data = np.load("data/raw_data.npz")
        required = ['ray_x', 'ray_y', 'ray_z', 'ray_r', 'ray_dtau',
                     'ray_Sigma', 'ray_t_geo', 'Omega', 'g_doppler',
                     't_frames', 'images_true', 'emission_true',
                     'rot_axis_true']
        for key in required:
            self.assertIn(key, data.files, f"Missing key: {key}")

    def test_emission_nonnegative(self):
        data = np.load("data/raw_data.npz")
        self.assertTrue((data['emission_true'] >= 0).all())

    def test_images_finite(self):
        data = np.load("data/raw_data.npz")
        self.assertTrue(np.all(np.isfinite(data['images_true'])))


class TestEndToEndPipeline(unittest.TestCase):
    """Test the full reconstruction pipeline with minimal iterations."""

    def test_mini_pipeline(self):
        from src.preprocessing import prepare_data
        from src.solvers import BHNeRFSolver
        from src.visualization import compute_metrics, compute_image_metrics

        obs_data, ground_truth, metadata = prepare_data("data")

        # Use minimal iterations for testing
        metadata['n_iters'] = 20
        metadata['batch_size'] = 4

        solver = BHNeRFSolver(metadata, device='cpu')
        result = solver.reconstruct(obs_data, seed=42)

        # Check loss history
        self.assertGreater(len(result['loss_history']), 0)
        self.assertTrue(all(np.isfinite(l) for l in result['loss_history']))

        # Check rotation axis
        self.assertEqual(result['rot_axis'].shape, (3,))
        np.testing.assert_allclose(
            np.linalg.norm(result['rot_axis']), 1.0, atol=1e-5
        )

        # Check emission prediction
        emission_3d = solver.predict_emission_3d(
            fov_M=obs_data['fov_M'], resolution=16
        )
        self.assertEqual(emission_3d.shape, (16, 16, 16))
        self.assertTrue(np.all(np.isfinite(emission_3d)))

        # Check movie prediction
        pred_movie = solver.predict_movie(obs_data)
        self.assertEqual(pred_movie.shape, ground_truth['images'].shape)
        self.assertTrue(np.all(np.isfinite(pred_movie)))

        # Check metrics compute without error
        metrics_3d = compute_metrics(emission_3d,
                                     ground_truth['emission_3d'][:16, :16, :16])
        self.assertIn('nrmse', metrics_3d)
        self.assertIn('ncc', metrics_3d)

        metrics_img = compute_image_metrics(pred_movie, ground_truth['images'])
        self.assertIn('nrmse_image', metrics_img)


if __name__ == '__main__':
    unittest.main()
