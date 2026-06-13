"""
Unit tests for physics_model.py
"""

import os
import unittest
import numpy as np
import torch

TASK_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "physics_model")


def _build_forward_model():
    """Build forward model from observation data for testing.

    Uses analytical pulse function computation to avoid NFFTInfo/pynfft dependency.
    """
    import sys
    sys.path.insert(0, TASK_DIR)
    from src.preprocessing import load_observation, extract_closure_indices
    from src.physics_model import NUFFTForwardModel
    import ehtim as eh
    import ehtim.const_def as ehc

    obs_data = load_observation(os.path.join(TASK_DIR, "data"))
    obs = obs_data['obs']
    closure = extract_closure_indices(obs)

    # Compute NUFFT params without NFFTInfo (avoids pynfft hang)
    fov_uas = 160.0
    npix = 32
    fov = fov_uas * eh.RADPERUAS
    psize = fov / npix

    uv_data = obs.unpack(['u', 'v'])
    uv = np.hstack((uv_data['u'].reshape(-1, 1), uv_data['v'].reshape(-1, 1)))
    vu = np.hstack((uv_data['v'].reshape(-1, 1), uv_data['u'].reshape(-1, 1)))

    # Analytical pulsefac for trianglePulse2D: sinc^2(u*psize) * sinc^2(v*psize)
    pulsefac = np.sinc(uv[:, 0] * psize) ** 2 * np.sinc(uv[:, 1] * psize) ** 2
    vu_scaled = np.array(vu * psize * 2 * np.pi)
    ktraj_vis = torch.tensor(vu_scaled.T, dtype=torch.float32).unsqueeze(0)
    pulsefac_vis = torch.tensor(
        np.concatenate([np.expand_dims(pulsefac.real, 0),
                        np.expand_dims(pulsefac.imag, 0)], 0),
        dtype=torch.float32)

    device = torch.device("cpu")
    model = NUFFTForwardModel(
        npix, ktraj_vis, pulsefac_vis,
        [torch.tensor(a, dtype=torch.long) for a in closure['cphase_ind_list']],
        [torch.tensor(a, dtype=torch.float32) for a in closure['cphase_sign_list']],
        [torch.tensor(a, dtype=torch.long) for a in closure['camp_ind_list']],
        device
    )
    return model


class TestNUFFTForwardModel(unittest.TestCase):
    """Test GPU NUFFT forward model outputs."""

    @classmethod
    def setUpClass(cls):
        cls.model = _build_forward_model()
        cls.fixture = np.load(os.path.join(FIXTURE_DIR, "nufft_forward.npz"),
                              allow_pickle=False)
        # Use the fixture input image
        cls.images = torch.tensor(cls.fixture['input_images'], dtype=torch.float32)

    def test_vis_shape(self):
        vis, _, _, _ = self.model(self.images)
        self.assertEqual(list(vis.shape), list(self.fixture['output_vis_shape']))

    def test_visamp_shape(self):
        _, visamp, _, _ = self.model(self.images)
        self.assertEqual(list(visamp.shape), list(self.fixture['output_visamp_shape']))

    def test_cphase_shape(self):
        _, _, cphase, _ = self.model(self.images)
        self.assertEqual(list(cphase.shape), list(self.fixture['output_cphase_shape']))

    def test_logcamp_shape(self):
        _, _, _, logcamp = self.model(self.images)
        self.assertEqual(list(logcamp.shape), list(self.fixture['output_logcamp_shape']))

    def test_vis_values(self):
        vis, _, _, _ = self.model(self.images)
        np.testing.assert_allclose(
            vis.detach().numpy(), self.fixture['output_vis'], rtol=1e-4)

    def test_cphase_values(self):
        _, _, cphase, _ = self.model(self.images)
        np.testing.assert_allclose(
            cphase.detach().numpy(), self.fixture['output_cphase'], rtol=1e-4)

    def test_logcamp_values(self):
        _, _, _, logcamp = self.model(self.images)
        np.testing.assert_allclose(
            logcamp.detach().numpy(), self.fixture['output_logcamp'], rtol=1e-4)


class TestLossAngleDiff(unittest.TestCase):
    """Test closure phase loss function."""

    def setUp(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.physics_model import Loss_angle_diff
        self.fixture = np.load(os.path.join(FIXTURE_DIR, "loss_angle_diff.npz"),
                               allow_pickle=False)
        self.loss_fn = Loss_angle_diff(self.fixture['input_sigma'], torch.device("cpu"))

    def test_output_value(self):
        y_true = torch.tensor(self.fixture['input_true'], dtype=torch.float32)
        y_pred = torch.tensor(self.fixture['input_pred'], dtype=torch.float32)
        loss = self.loss_fn(y_true, y_pred)
        np.testing.assert_allclose(
            loss.detach().numpy(), self.fixture['output_loss'], rtol=1e-5)


class TestLossLogcaDiff2(unittest.TestCase):
    """Test log closure amplitude loss function."""

    def setUp(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.physics_model import Loss_logca_diff2
        self.fixture = np.load(os.path.join(FIXTURE_DIR, "loss_logca_diff2.npz"),
                               allow_pickle=False)
        self.loss_fn = Loss_logca_diff2(self.fixture['input_sigma'], torch.device("cpu"))

    def test_output_value(self):
        y_true = torch.tensor(self.fixture['input_true'], dtype=torch.float32)
        y_pred = torch.tensor(self.fixture['input_pred'], dtype=torch.float32)
        loss = self.loss_fn(y_true, y_pred)
        np.testing.assert_allclose(
            loss.detach().numpy(), self.fixture['output_loss'], rtol=1e-5)


class TestLossImagePriors(unittest.TestCase):
    """Test image prior loss functions."""

    def setUp(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        self.fixture = np.load(os.path.join(FIXTURE_DIR, "loss_priors.npz"),
                               allow_pickle=False)
        self.img = torch.tensor(self.fixture['input_image'], dtype=torch.float32)

    def test_l1(self):
        from src.physics_model import Loss_l1
        loss = Loss_l1(self.img)
        np.testing.assert_allclose(
            loss.detach().numpy(), self.fixture['output_l1'], rtol=1e-5)

    def test_tsv(self):
        from src.physics_model import Loss_TSV
        loss = Loss_TSV(self.img)
        np.testing.assert_allclose(
            loss.detach().numpy(), self.fixture['output_tsv'], rtol=1e-5)

    def test_tv(self):
        from src.physics_model import Loss_TV
        loss = Loss_TV(self.img)
        np.testing.assert_allclose(
            loss.detach().numpy(), self.fixture['output_tv'], rtol=1e-5)

    def test_flux(self):
        from src.physics_model import Loss_flux
        flux = float(self.fixture['config_flux'])
        loss_fn = Loss_flux(flux)
        loss = loss_fn(self.img)
        np.testing.assert_allclose(
            loss.detach().numpy(), self.fixture['output_flux'], rtol=1e-5)

    def test_center(self):
        from src.physics_model import Loss_center
        loss_fn = Loss_center(torch.device("cpu"), center=15.5, dim=32)
        loss = loss_fn(self.img)
        np.testing.assert_allclose(
            loss.detach().numpy(), self.fixture['output_center'], rtol=1e-5)

    def test_cross_entropy(self):
        from src.physics_model import Loss_cross_entropy
        prior_im = torch.tensor(self.fixture['input_prior_im'], dtype=torch.float32)
        loss = Loss_cross_entropy(prior_im, self.img)
        np.testing.assert_allclose(
            loss.detach().numpy(), self.fixture['output_cross_entropy'], rtol=1e-5)


class TestLossVisDiff(unittest.TestCase):
    """Test complex visibility loss function."""

    def setUp(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.physics_model import Loss_vis_diff
        self.fixture = np.load(os.path.join(FIXTURE_DIR, "loss_vis_diff.npz"),
                               allow_pickle=False)
        self.loss_fn = Loss_vis_diff(self.fixture['input_sigma'], torch.device("cpu"))

    def test_output_value(self):
        y_true = torch.tensor(self.fixture['input_true'], dtype=torch.float32)
        y_pred = torch.tensor(self.fixture['input_pred'], dtype=torch.float32)
        loss = self.loss_fn(y_true, y_pred)
        np.testing.assert_allclose(
            loss.detach().numpy(), self.fixture['output_loss'], rtol=1e-5)


class TestLossLogampDiff(unittest.TestCase):
    """Test log visibility amplitude loss function."""

    def setUp(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.physics_model import Loss_logamp_diff
        self.fixture = np.load(os.path.join(FIXTURE_DIR, "loss_logamp_diff.npz"),
                               allow_pickle=False)
        self.loss_fn = Loss_logamp_diff(self.fixture['input_sigma'], torch.device("cpu"))

    def test_output_value(self):
        y_true = torch.tensor(self.fixture['input_true'], dtype=torch.float32)
        y_pred = torch.tensor(self.fixture['input_pred'], dtype=torch.float32)
        loss = self.loss_fn(y_true, y_pred)
        np.testing.assert_allclose(
            loss.detach().numpy(), self.fixture['output_loss'], rtol=1e-5)


class TestLossVisampDiff(unittest.TestCase):
    """Test visibility amplitude squared difference loss function."""

    def setUp(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.physics_model import Loss_visamp_diff
        self.fixture = np.load(os.path.join(FIXTURE_DIR, "loss_visamp_diff.npz"),
                               allow_pickle=False)
        self.loss_fn = Loss_visamp_diff(self.fixture['input_sigma'], torch.device("cpu"))

    def test_output_value(self):
        y_true = torch.tensor(self.fixture['input_true'], dtype=torch.float32)
        y_pred = torch.tensor(self.fixture['input_pred'], dtype=torch.float32)
        loss = self.loss_fn(y_true, y_pred)
        np.testing.assert_allclose(
            loss.detach().numpy(), self.fixture['output_loss'], rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
