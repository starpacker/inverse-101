"""Tests for physics model module."""
import os
import sys
import unittest
import numpy as np
import torch

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIX_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures', 'physics_model')
sys.path.insert(0, TASK_DIR)

from src.physics_model import (
    SimpleCrescentParam2Img, SimpleCrescentNuisanceParam2Img,
    NUFFTForwardModel, Loss_angle_diff, Loss_logca_diff2,
)
from src.preprocessing import (
    load_observation, load_metadata, extract_closure_indices,
    compute_nufft_params,
)


class TestSimpleCrescentParam2Img(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.load(os.path.join(FIX_DIR, 'simple_crescent.npz'))
        cls.model = SimpleCrescentParam2Img(npix=64, fov=120.0)

    def test_nparams(self):
        self.assertEqual(self.model.nparams, 4)

    def test_output_shape(self):
        torch.manual_seed(42)
        params = torch.rand(4, 4)
        img = self.model.forward(params)
        self.assertEqual(img.shape, (4, 64, 64))

    def test_output_values(self):
        params = torch.tensor(self.data['input_params'])
        img = self.model.forward(params)
        np.testing.assert_allclose(
            img.detach().numpy(), self.data['output_img'], rtol=1e-5)

    def test_flux_normalized(self):
        params = torch.rand(2, 4)
        img = self.model.forward(params)
        flux = img.sum(dim=(-1, -2)).detach().numpy()
        np.testing.assert_allclose(flux, np.ones(2), rtol=1e-3)

    def test_non_negative(self):
        params = torch.rand(2, 4)
        img = self.model.forward(params)
        self.assertTrue((img >= 0).all())


class TestSimpleCrescentNuisanceParam2Img(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.load(os.path.join(FIX_DIR, 'simple_crescent_nuisance.npz'))
        cls.model = SimpleCrescentNuisanceParam2Img(npix=64, n_gaussian=2, fov=120.0)

    def test_nparams(self):
        self.assertEqual(self.model.nparams, 16)
        self.assertEqual(self.model.nparams, int(self.data['nparams']))

    def test_output_shape(self):
        torch.manual_seed(42)
        params = torch.rand(4, self.model.nparams)
        img = self.model.forward(params)
        self.assertEqual(img.shape, (4, 64, 64))

    def test_output_values(self):
        params = torch.tensor(self.data['input_params'])
        img = self.model.forward(params)
        np.testing.assert_allclose(
            img.detach().numpy(), self.data['output_img'], rtol=1e-5)

    def test_flux_normalized(self):
        params = torch.rand(2, self.model.nparams)
        img = self.model.forward(params)
        flux = img.sum(dim=(-1, -2)).detach().numpy()
        np.testing.assert_allclose(flux, np.ones(2), rtol=1e-3)


class TestNUFFTForwardModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.load(os.path.join(FIX_DIR, 'nufft_forward.npz'))
        obs_data = load_observation(os.path.join(TASK_DIR, 'data'))
        obs = obs_data['obs']
        metadata = load_metadata(os.path.join(TASK_DIR, 'data'))
        closure_indices = extract_closure_indices(obs)
        nufft_params = compute_nufft_params(obs, metadata['npix'], metadata['fov_uas'])

        cphase_ind_torch = [torch.tensor(a, dtype=torch.long)
                            for a in closure_indices["cphase_ind_list"]]
        cphase_sign_torch = [torch.tensor(a, dtype=torch.float32)
                             for a in closure_indices["cphase_sign_list"]]
        camp_ind_torch = [torch.tensor(a, dtype=torch.long)
                          for a in closure_indices["camp_ind_list"]]

        device = torch.device('cpu')
        cls.forward_model = NUFFTForwardModel(
            64, nufft_params["ktraj_vis"], nufft_params["pulsefac_vis"],
            cphase_ind_torch, cphase_sign_torch, camp_ind_torch, device
        )

    def test_output_shapes(self):
        test_img = torch.tensor(self.data['input_img'])
        vis, visamp, cphase, logcamp = self.forward_model(test_img)
        self.assertEqual(vis.shape[0], 2)
        self.assertEqual(vis.shape[1], 2)
        self.assertGreater(visamp.shape[1], 0)
        self.assertGreater(cphase.shape[1], 0)
        self.assertGreater(logcamp.shape[1], 0)

    def test_output_values(self):
        test_img = torch.tensor(self.data['input_img'])
        vis, visamp, cphase, logcamp = self.forward_model(test_img)
        np.testing.assert_allclose(
            vis.detach().numpy(), self.data['output_vis'], rtol=1e-4)
        np.testing.assert_allclose(
            visamp.detach().numpy(), self.data['output_visamp'], rtol=1e-4)
        np.testing.assert_allclose(
            cphase.detach().numpy(), self.data['output_cphase'], rtol=1e-4)
        np.testing.assert_allclose(
            logcamp.detach().numpy(), self.data['output_logcamp'], rtol=1e-4)

    def test_visamp_positive(self):
        test_img = torch.tensor(self.data['input_img'])
        _, visamp, _, _ = self.forward_model(test_img)
        self.assertTrue((visamp > 0).all())


class TestLossFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.load(os.path.join(FIX_DIR, 'loss_functions.npz'))

    def test_loss_cphase_shape(self):
        loss = self.data['output_loss_cphase']
        self.assertEqual(len(loss.shape), 1)
        self.assertEqual(loss.shape[0], 2)

    def test_loss_logca_shape(self):
        loss = self.data['output_loss_logca']
        self.assertEqual(len(loss.shape), 1)
        self.assertEqual(loss.shape[0], 2)

    def test_loss_positive(self):
        self.assertTrue((self.data['output_loss_cphase'] >= 0).all())
        self.assertTrue((self.data['output_loss_logca'] >= 0).all())


if __name__ == '__main__':
    unittest.main()
