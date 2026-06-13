"""
Unit tests for preprocessing.py
"""

import os
import unittest
import json
import numpy as np
import torch

TASK_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "preprocessing")


class TestLoadObservation(unittest.TestCase):
    """Test loading observation data from UVFITS."""

    def setUp(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.preprocessing import load_observation
        self.data = load_observation(os.path.join(TASK_DIR, "data"))
        self.fixture = np.load(os.path.join(FIXTURE_DIR, "load_observation.npz"),
                               allow_pickle=False)

    def test_vis_shape(self):
        self.assertEqual(self.data['vis'].shape, tuple(self.fixture['output_vis_shape']))

    def test_vis_dtype(self):
        self.assertTrue(np.iscomplexobj(self.data['vis']))

    def test_uv_coords_shape(self):
        self.assertEqual(self.data['uv_coords'].shape,
                         tuple(self.fixture['output_uv_shape']))

    def test_sigma_positive(self):
        self.assertTrue(np.all(self.data['vis_sigma'] > 0))

    def test_vis_values(self):
        np.testing.assert_allclose(
            self.data['vis'][:5], self.fixture['output_vis_first5'], rtol=1e-10)


class TestLoadMetadata(unittest.TestCase):
    """Test loading metadata JSON."""

    def setUp(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.preprocessing import load_metadata
        self.meta = load_metadata(os.path.join(TASK_DIR, "data"))

    def test_npix(self):
        self.assertEqual(self.meta["npix"], 32)

    def test_fov(self):
        self.assertAlmostEqual(self.meta["fov_uas"], 160.0)

    def test_required_keys(self):
        required = ["npix", "fov_uas", "n_flow", "n_epoch", "batch_size",
                     "lr", "logdet_weight"]
        for key in required:
            self.assertIn(key, self.meta)


class TestExtractClosureIndices(unittest.TestCase):
    """Test closure phase/amplitude index extraction."""

    def setUp(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.preprocessing import load_observation, extract_closure_indices
        obs_data = load_observation(os.path.join(TASK_DIR, "data"))
        self.indices = extract_closure_indices(obs_data['obs'])
        self.fixture = np.load(os.path.join(FIXTURE_DIR, "extract_closure_indices.npz"),
                               allow_pickle=False)

    def test_cphase_ind_count(self):
        self.assertEqual(len(self.indices['cphase_ind_list']), 3)

    def test_camp_ind_count(self):
        self.assertEqual(len(self.indices['camp_ind_list']), 4)

    def test_cphase_ind_shape(self):
        n_cp = self.fixture['output_n_cphase']
        for ind in self.indices['cphase_ind_list']:
            self.assertEqual(len(ind), int(n_cp))

    def test_camp_ind_shape(self):
        n_ca = self.fixture['output_n_camp']
        for ind in self.indices['camp_ind_list']:
            self.assertEqual(len(ind), int(n_ca))

    def test_cphase_ind_values(self):
        for i in range(3):
            np.testing.assert_array_equal(
                self.indices['cphase_ind_list'][i],
                self.fixture[f'output_cphase_ind{i}'])

    def test_cphase_sign_values(self):
        for i in range(3):
            np.testing.assert_array_equal(
                self.indices['cphase_sign_list'][i],
                self.fixture[f'output_cphase_sign{i}'])

    def test_camp_ind_values(self):
        for i in range(4):
            np.testing.assert_array_equal(
                self.indices['camp_ind_list'][i],
                self.fixture[f'output_camp_ind{i}'])


class TestComputeNufftParams(unittest.TestCase):
    """Test NUFFT parameter computation."""

    def setUp(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.preprocessing import load_observation, compute_nufft_params
        obs_data = load_observation(os.path.join(TASK_DIR, "data"))
        self.params = compute_nufft_params(obs_data['obs'], npix=32, fov_uas=160.0)
        self.fixture = np.load(os.path.join(FIXTURE_DIR, "compute_nufft_params.npz"),
                               allow_pickle=False)

    def test_ktraj_shape(self):
        self.assertEqual(list(self.params['ktraj_vis'].shape),
                         list(self.fixture['output_ktraj_shape']))

    def test_pulsefac_shape(self):
        self.assertEqual(list(self.params['pulsefac_vis'].shape),
                         list(self.fixture['output_pulsefac_shape']))

    def test_ktraj_values(self):
        np.testing.assert_allclose(
            self.params['ktraj_vis'].numpy(),
            self.fixture['output_ktraj_vis'], rtol=1e-6)

    def test_pulsefac_values(self):
        np.testing.assert_allclose(
            self.params['pulsefac_vis'].numpy(),
            self.fixture['output_pulsefac_vis'], rtol=1e-6)


class TestBuildPriorImage(unittest.TestCase):
    """Test Gaussian prior image construction."""

    def setUp(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.preprocessing import load_observation, build_prior_image
        obs_data = load_observation(os.path.join(TASK_DIR, "data"))
        self.prior, self.flux = build_prior_image(obs_data['obs'], 32, 160.0)
        self.fixture = np.load(os.path.join(FIXTURE_DIR, "build_prior_image.npz"),
                               allow_pickle=False)

    def test_shape(self):
        self.assertEqual(self.prior.shape, (32, 32))

    def test_positive(self):
        self.assertTrue(np.all(self.prior >= 0))

    def test_flux_const(self):
        np.testing.assert_allclose(self.flux, float(self.fixture['output_flux_const']),
                                   rtol=1e-6)

    def test_prior_values(self):
        np.testing.assert_allclose(self.prior, self.fixture['output_prior'], rtol=1e-6)


class TestLoadGroundTruth(unittest.TestCase):
    """Test ground truth image loading and regridding."""

    def setUp(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.preprocessing import load_ground_truth
        self.gt = load_ground_truth(os.path.join(TASK_DIR, "data"), npix=32, fov_uas=160.0)
        self.fixture = np.load(os.path.join(FIXTURE_DIR, "load_ground_truth.npz"),
                               allow_pickle=False)

    def test_shape(self):
        self.assertEqual(self.gt.shape, (32, 32))

    def test_positive(self):
        self.assertTrue(np.all(self.gt >= 0))

    def test_nonzero(self):
        self.assertGreater(self.gt.sum(), 0)

    def test_values(self):
        np.testing.assert_allclose(self.gt, self.fixture['output_image'], rtol=1e-6)


class TestPrepareData(unittest.TestCase):
    """Test the combined prepare_data wrapper."""

    def setUp(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.preprocessing import prepare_data
        self.result = prepare_data(os.path.join(TASK_DIR, "data"))

    def test_returns_7_tuple(self):
        self.assertEqual(len(self.result), 7)

    def test_obs_data_has_vis(self):
        obs_data = self.result[1]
        self.assertIn('vis', obs_data)

    def test_closure_indices_has_cphase(self):
        closure = self.result[2]
        self.assertIn('cphase_ind_list', closure)
        self.assertEqual(len(closure['cphase_ind_list']), 3)

    def test_nufft_params_has_ktraj(self):
        nufft = self.result[3]
        self.assertIn('ktraj_vis', nufft)

    def test_prior_image_shape(self):
        prior = self.result[4]
        metadata = self.result[6]
        npix = metadata['npix']
        self.assertEqual(prior.shape, (npix, npix))

    def test_flux_const_positive(self):
        flux = self.result[5]
        self.assertGreater(flux, 0)

    def test_metadata_keys(self):
        metadata = self.result[6]
        self.assertIn('npix', metadata)
        self.assertIn('fov_uas', metadata)


if __name__ == "__main__":
    unittest.main()
