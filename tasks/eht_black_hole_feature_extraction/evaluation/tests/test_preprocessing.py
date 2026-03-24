"""Tests for preprocessing module."""
import os
import sys
import unittest
import numpy as np

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIX_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures', 'preprocessing')
sys.path.insert(0, TASK_DIR)

from src.preprocessing import (
    load_observation, load_metadata, extract_closure_indices,
    compute_nufft_params, estimate_flux, prepare_data,
)


class TestLoadObservation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.load(os.path.join(FIX_DIR, 'load_observation.npz'), allow_pickle=True)
        cls.result = load_observation(os.path.join(TASK_DIR, 'data'))

    def test_vis_shape(self):
        expected = tuple(self.data['vis_shape'])
        self.assertEqual(self.result['vis'].shape, expected)

    def test_vis_dtype(self):
        self.assertTrue(np.issubdtype(self.result['vis'].dtype, np.complexfloating))

    def test_uv_shape(self):
        expected = tuple(self.data['uv_shape'])
        self.assertEqual(self.result['uv_coords'].shape, expected)

    def test_vis_values(self):
        np.testing.assert_allclose(
            self.result['vis'][:5], self.data['vis_sample'], rtol=1e-10)

    def test_sigma_values(self):
        np.testing.assert_allclose(
            self.result['vis_sigma'][:5], self.data['sigma_sample'], rtol=1e-10)

    def test_obs_object(self):
        self.assertIsNotNone(self.result['obs'])


class TestLoadMetadata(unittest.TestCase):
    def test_loads_json(self):
        meta = load_metadata(os.path.join(TASK_DIR, 'data'))
        self.assertIn('npix', meta)
        self.assertIn('fov_uas', meta)
        self.assertEqual(meta['npix'], 64)
        self.assertEqual(meta['fov_uas'], 120.0)


class TestExtractClosureIndices(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.load(os.path.join(FIX_DIR, 'extract_closure_indices.npz'), allow_pickle=True)
        obs_data = load_observation(os.path.join(TASK_DIR, 'data'))
        cls.result = extract_closure_indices(obs_data['obs'])

    def test_n_cphase(self):
        self.assertEqual(len(self.result['cphase_data']['cphase']),
                         int(self.data['n_cphase']))

    def test_n_camp(self):
        self.assertEqual(len(self.result['camp_data']['camp']),
                         int(self.data['n_camp']))

    def test_cphase_ind_shape(self):
        self.assertEqual(len(self.result['cphase_ind_list']), 3)
        for ind in self.result['cphase_ind_list']:
            self.assertEqual(len(ind), int(self.data['n_cphase']))

    def test_camp_ind_shape(self):
        self.assertEqual(len(self.result['camp_ind_list']), 4)
        for ind in self.result['camp_ind_list']:
            self.assertEqual(len(ind), int(self.data['n_camp']))

    def test_cphase_ind_values(self):
        np.testing.assert_array_equal(
            self.result['cphase_ind_list'][0][:10],
            self.data['cphase_ind0_sample'])

    def test_cphase_sign_values(self):
        np.testing.assert_array_equal(
            self.result['cphase_sign_list'][0][:10],
            self.data['cphase_sign0_sample'])


class TestComputeNufftParams(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.load(os.path.join(FIX_DIR, 'compute_nufft_params.npz'), allow_pickle=True)
        obs_data = load_observation(os.path.join(TASK_DIR, 'data'))
        cls.result = compute_nufft_params(obs_data['obs'], 64, 120.0)

    def test_ktraj_shape(self):
        expected = tuple(self.data['ktraj_shape'])
        self.assertEqual(tuple(self.result['ktraj_vis'].shape), expected)

    def test_pulsefac_shape(self):
        expected = tuple(self.data['pulsefac_shape'])
        self.assertEqual(tuple(self.result['pulsefac_vis'].shape), expected)

    def test_ktraj_values(self):
        np.testing.assert_allclose(
            self.result['ktraj_vis'][0, :, :5].numpy(),
            self.data['ktraj_sample'], rtol=1e-6)

    def test_pulsefac_values(self):
        np.testing.assert_allclose(
            self.result['pulsefac_vis'][:, :5].numpy(),
            self.data['pulsefac_sample'], rtol=1e-6)


class TestEstimateFlux(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.load(os.path.join(FIX_DIR, 'estimate_flux.npz'), allow_pickle=True)
        obs_data = load_observation(os.path.join(TASK_DIR, 'data'))
        cls.flux = estimate_flux(obs_data['obs'])

    def test_flux_value(self):
        np.testing.assert_allclose(self.flux, float(self.data['flux_const']), rtol=1e-10)

    def test_flux_positive(self):
        self.assertGreater(self.flux, 0)


class TestPrepareData(unittest.TestCase):
    def test_returns_all(self):
        result = prepare_data(os.path.join(TASK_DIR, 'data'))
        self.assertEqual(len(result), 6)
        obs, obs_data, closure_indices, nufft_params, flux_const, metadata = result
        self.assertIsNotNone(obs)
        self.assertIn('vis', obs_data)
        self.assertIn('cphase_ind_list', closure_indices)
        self.assertIn('ktraj_vis', nufft_params)
        self.assertGreater(flux_const, 0)
        self.assertIn('npix', metadata)


if __name__ == '__main__':
    unittest.main()
