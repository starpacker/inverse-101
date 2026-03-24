"""
Tests for src/preprocessing.py
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing import load_metadata, load_observation, load_ground_truth, prepare_data

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'preprocessing')


class TestLoadMetadata(unittest.TestCase):

    def test_returns_dict(self):
        meta = load_metadata("data")
        self.assertIsInstance(meta, dict)

    def test_required_keys(self):
        meta = load_metadata("data")
        required = ['spin', 'inclination_deg', 'fov_M', 'num_alpha', 'num_beta',
                     'ngeo', 'emission_resolution', 'net_depth', 'net_width',
                     'posenc_deg', 'n_iters', 'lr_init', 'lr_final', 'batch_size']
        for key in required:
            self.assertIn(key, meta)


class TestLoadObservation(unittest.TestCase):

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, 'load_observation.npz'))
        self.obs = load_observation("data")

    def test_ray_coords_shape(self):
        expected = tuple(self.f['ray_coords_shape'])
        self.assertEqual(tuple(self.obs['ray_coords'].shape), expected)

    def test_omega_shape(self):
        expected = tuple(self.f['Omega_shape'])
        self.assertEqual(tuple(self.obs['Omega'].shape), expected)

    def test_n_frames(self):
        expected = int(self.f['n_frames'])
        self.assertEqual(len(self.obs['t_frames']), expected)

    def test_images_shape(self):
        n = len(self.obs['t_frames'])
        shape = self.obs['images'].shape
        self.assertEqual(shape[0], n)
        self.assertEqual(len(shape), 3)

    def test_fov_positive(self):
        self.assertGreater(self.obs['fov_M'], 0)


class TestLoadGroundTruth(unittest.TestCase):

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, 'load_ground_truth.npz'))
        self.gt = load_ground_truth("data")

    def test_emission_shape(self):
        expected = tuple(self.f['emission_shape'])
        self.assertEqual(self.gt['emission_3d'].shape, expected)

    def test_images_shape(self):
        expected = tuple(self.f['images_shape'])
        self.assertEqual(self.gt['images'].shape, expected)

    def test_rot_axis_shape(self):
        self.assertEqual(self.gt['rot_axis'].shape, (3,))

    def test_rot_axis_unit(self):
        np.testing.assert_allclose(
            np.linalg.norm(self.gt['rot_axis']), 1.0, atol=1e-5
        )


class TestPrepareData(unittest.TestCase):

    def test_returns_tuple_of_three(self):
        result = prepare_data("data")
        self.assertEqual(len(result), 3)

    def test_types(self):
        obs_data, ground_truth, metadata = prepare_data("data")
        self.assertIsInstance(obs_data, dict)
        self.assertIsInstance(ground_truth, dict)
        self.assertIsInstance(metadata, dict)


if __name__ == '__main__':
    unittest.main()
