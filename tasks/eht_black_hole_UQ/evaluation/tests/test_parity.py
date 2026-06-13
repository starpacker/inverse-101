"""
Parity tests: original DPI code vs cleaned code.

Verifies that our cleaned implementation produces numerically identical
outputs to the original DPI code at every stage.

Because pynfft segfaults when NFFTInfo is called twice in one process,
preprocessing/forward-model parity is tested via pre-saved reference files.
Generate them with:
    cd tasks/eht_black_hole_UQ
    conda run -n dpi python evaluation/tests/generate_parity_refs.py
"""

import os
import sys
import unittest
import numpy as np
import torch

TASK_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
ORIGINAL_DIR = os.path.join(os.path.dirname(__file__), "..", "reference_code")
DATA_DIR = os.path.join(TASK_DIR, "data")
REFERENCE_DIR = os.path.join(TASK_DIR, "evaluation", "reference_outputs")
PARITY_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "parity")


class TestPreprocessingParity(unittest.TestCase):
    """Verify preprocessing outputs match original code (via saved fixtures)."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(os.path.join(PARITY_DIR, "orig_preproc.npz")):
            raise unittest.SkipTest("Parity fixtures not generated yet")
        cls.orig = np.load(os.path.join(PARITY_DIR, "orig_preproc.npz"))

        sys.path.insert(0, TASK_DIR)
        import ehtim as eh
        from src.preprocessing import (
            extract_closure_indices, compute_nufft_params, build_prior_image)

        obs = eh.obsdata.load_uvfits(os.path.join(DATA_DIR, "obs.uvfits"))
        cls.closure = extract_closure_indices(obs)
        cls.nufft = compute_nufft_params(obs, 32, 160.0)
        obs2 = eh.obsdata.load_uvfits(os.path.join(DATA_DIR, "obs.uvfits"))
        cls.prior_image, cls.flux_const = build_prior_image(obs2, 32, 160.0, 50.0)

    def test_cphase_ind_match(self):
        for i in range(3):
            np.testing.assert_array_equal(
                self.orig[f'cp_ind{i}'], self.closure['cphase_ind_list'][i])

    def test_cphase_sign_match(self):
        for i in range(3):
            np.testing.assert_array_equal(
                self.orig[f'cp_sign{i}'], self.closure['cphase_sign_list'][i])

    def test_camp_ind_match(self):
        for i in range(4):
            np.testing.assert_array_equal(
                self.orig[f'ca_ind{i}'], self.closure['camp_ind_list'][i])

    def test_ktraj_match(self):
        np.testing.assert_allclose(
            self.orig['ktraj'], self.nufft['ktraj_vis'].numpy(), rtol=1e-6)

    def test_pulsefac_match(self):
        np.testing.assert_allclose(
            self.orig['pulsefac'], self.nufft['pulsefac_vis'].numpy(), rtol=1e-6)

    def test_flux_const_match(self):
        np.testing.assert_allclose(float(self.orig['flux_const']), self.flux_const, rtol=1e-10)

    def test_prior_image_match(self):
        np.testing.assert_allclose(self.orig['prior_image'], self.prior_image, rtol=1e-6)


class TestForwardModelParity(unittest.TestCase):
    """Verify NUFFT forward model produces identical outputs (via saved fixtures)."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(os.path.join(PARITY_DIR, "orig_forward.npz")):
            raise unittest.SkipTest("Parity fixtures not generated yet")
        cls.orig = np.load(os.path.join(PARITY_DIR, "orig_forward.npz"))

        sys.path.insert(0, TASK_DIR)
        import ehtim as eh
        from src.preprocessing import extract_closure_indices, compute_nufft_params
        from src.physics_model import NUFFTForwardModel

        obs = eh.obsdata.load_uvfits(os.path.join(DATA_DIR, "obs.uvfits"))
        closure = extract_closure_indices(obs)
        nufft = compute_nufft_params(obs, 32, 160.0)

        device = torch.device("cpu")
        cphase_ind_t = [torch.tensor(a, dtype=torch.long) for a in closure['cphase_ind_list']]
        cphase_sign_t = [torch.tensor(a, dtype=torch.float32) for a in closure['cphase_sign_list']]
        camp_ind_t = [torch.tensor(a, dtype=torch.long) for a in closure['camp_ind_list']]
        cls.fwd = NUFFTForwardModel(32, nufft['ktraj_vis'], nufft['pulsefac_vis'],
                                     cphase_ind_t, cphase_sign_t, camp_ind_t, device)

        np.random.seed(42)
        cls.test_images = torch.tensor(
            np.abs(np.random.randn(4, 32, 32)).astype(np.float32))

    def test_vis_match(self):
        vis, _, _, _ = self.fwd(self.test_images)
        np.testing.assert_allclose(
            self.orig['vis'], vis.detach().numpy(), rtol=1e-4, atol=1e-4)

    def test_visamp_match(self):
        _, visamp, _, _ = self.fwd(self.test_images)
        np.testing.assert_allclose(
            self.orig['visamp'], visamp.detach().numpy(), rtol=1e-4, atol=1e-4)

    def test_cphase_match(self):
        _, _, cphase, _ = self.fwd(self.test_images)
        np.testing.assert_allclose(
            self.orig['cphase'], cphase.detach().numpy(), rtol=1e-3, atol=1e-3)

    def test_logcamp_match(self):
        _, _, _, logcamp = self.fwd(self.test_images)
        np.testing.assert_allclose(
            self.orig['logcamp'], logcamp.detach().numpy(), rtol=1e-4, atol=1e-5)


class TestLossFunctionParity(unittest.TestCase):
    """Verify all loss functions produce identical outputs."""

    @classmethod
    def setUpClass(cls):
        if ORIGINAL_DIR not in sys.path:
            sys.path.insert(0, ORIGINAL_DIR)
        if TASK_DIR not in sys.path:
            sys.path.insert(0, TASK_DIR)

        from interferometry_helpers import (
            Loss_angle_diff as O_LAD, Loss_logca_diff2 as O_LCD,
            Loss_logamp_diff as O_LLD, Loss_l1 as O_L1,
            Loss_TSV as O_TSV, Loss_flux as O_F,
            Loss_center as O_C, Loss_cross_entropy as O_CE)
        from src.physics_model import (
            Loss_angle_diff, Loss_logca_diff2, Loss_logamp_diff,
            Loss_l1, Loss_TSV, Loss_flux, Loss_center, Loss_cross_entropy)

        cls.device = torch.device("cpu")
        cls.orig = {'LAD': O_LAD, 'LCD': O_LCD, 'LLD': O_LLD,
                     'L1': O_L1, 'TSV': O_TSV, 'F': O_F, 'C': O_C, 'CE': O_CE}
        cls.clean = {'LAD': Loss_angle_diff, 'LCD': Loss_logca_diff2,
                      'LLD': Loss_logamp_diff, 'L1': Loss_l1, 'TSV': Loss_TSV,
                      'F': Loss_flux, 'C': Loss_center, 'CE': Loss_cross_entropy}
        np.random.seed(42)

    def test_loss_angle_diff(self):
        sigma = np.random.rand(20).astype(np.float32) * 10 + 1
        t, p = torch.randn(4, 20) * 30, torch.randn(4, 20) * 30
        np.testing.assert_allclose(
            self.orig['LAD'](sigma, self.device)(t, p).numpy(),
            self.clean['LAD'](sigma, self.device)(t, p).numpy(), rtol=1e-6)

    def test_loss_logca_diff2(self):
        sigma = np.abs(np.random.randn(20).astype(np.float32)) + 0.1
        t, p = torch.randn(4, 20), torch.randn(4, 20)
        np.testing.assert_allclose(
            self.orig['LCD'](sigma, self.device)(t, p).numpy(),
            self.clean['LCD'](sigma, self.device)(t, p).numpy(), rtol=1e-6)

    def test_loss_logamp_diff(self):
        sigma = np.abs(np.random.randn(20).astype(np.float32)) + 0.1
        t = torch.abs(torch.randn(4, 20)) + 0.1
        p = torch.abs(t + torch.randn(4, 20) * 0.01) + 0.01
        np.testing.assert_allclose(
            self.orig['LLD'](sigma, self.device)(t, p).numpy(),
            self.clean['LLD'](sigma, self.device)(t, p).numpy(), rtol=1e-6)

    def test_loss_l1(self):
        img = torch.abs(torch.randn(4, 32, 32))
        np.testing.assert_allclose(
            self.orig['L1'](img).numpy(), self.clean['L1'](img).numpy(), rtol=1e-10)

    def test_loss_tsv(self):
        img = torch.abs(torch.randn(4, 32, 32))
        np.testing.assert_allclose(
            self.orig['TSV'](img).numpy(), self.clean['TSV'](img).numpy(), rtol=1e-10)

    def test_loss_flux(self):
        img = torch.abs(torch.randn(4, 32, 32))
        np.testing.assert_allclose(
            self.orig['F'](2.0)(img).numpy(), self.clean['F'](2.0)(img).numpy(), rtol=1e-10)

    def test_loss_center(self):
        img = torch.abs(torch.randn(4, 32, 32))
        np.testing.assert_allclose(
            self.orig['C'](self.device, 15.5, 32)(img).numpy(),
            self.clean['C'](self.device, 15.5, 32)(img).numpy(), rtol=1e-6)

    def test_loss_cross_entropy(self):
        prior = torch.abs(torch.randn(32, 32)) + 0.01
        img = torch.abs(torch.randn(4, 32, 32)) + 0.01
        np.testing.assert_allclose(
            self.orig['CE'](prior, img).numpy(),
            self.clean['CE'](prior, img).numpy(), rtol=1e-6)


class TestFlowArchitectureParity(unittest.TestCase):
    """Verify RealNVP produces identical outputs with same state_dict."""

    @classmethod
    def setUpClass(cls):
        if ORIGINAL_DIR not in sys.path:
            sys.path.insert(0, ORIGINAL_DIR)
        if TASK_DIR not in sys.path:
            sys.path.insert(0, TASK_DIR)
        from generative_model.realnvpfc_model import RealNVP as Orig_RealNVP
        from src.solvers import RealNVP
        cls.Orig_RealNVP = Orig_RealNVP
        cls.RealNVP = RealNVP

    def test_permutation_orders_match(self):
        o = self.Orig_RealNVP(64, 4, affine=True, seqfrac=4)
        c = self.RealNVP(64, 4, affine=True, seqfrac=4)
        for i in range(4):
            np.testing.assert_array_equal(o.orders[i], c.orders[i])
            np.testing.assert_array_equal(o.inverse_orders[i], c.inverse_orders[i])

    def test_reverse_output_match(self):
        o = self.Orig_RealNVP(64, 4, affine=True, seqfrac=4)
        c = self.RealNVP(64, 4, affine=True, seqfrac=4)
        c.load_state_dict(o.state_dict())
        o.eval(); c.eval()
        torch.manual_seed(42)
        z = torch.randn(8, 64)
        with torch.no_grad():
            xo, lo = o.reverse(z)
            xc, lc = c.reverse(z)
        np.testing.assert_allclose(xo.numpy(), xc.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(lo.numpy(), lc.numpy(), rtol=1e-5, atol=1e-6)

    def test_forward_output_match(self):
        o = self.Orig_RealNVP(64, 4, affine=True, seqfrac=4)
        c = self.RealNVP(64, 4, affine=True, seqfrac=4)
        c.load_state_dict(o.state_dict())
        o.eval(); c.eval()
        torch.manual_seed(42)
        x = torch.randn(8, 64)
        with torch.no_grad():
            zo, lo = o.forward(x)
            zc, lc = c.forward(x)
        np.testing.assert_allclose(zo.numpy(), zc.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(lo.numpy(), lc.numpy(), rtol=1e-5, atol=1e-6)

    def test_full_size_reverse_match(self):
        o = self.Orig_RealNVP(1024, 4, affine=True, seqfrac=4)
        c = self.RealNVP(1024, 4, affine=True, seqfrac=4)
        c.load_state_dict(o.state_dict())
        o.eval(); c.eval()
        torch.manual_seed(0)
        z = torch.randn(4, 1024)
        with torch.no_grad():
            xo, lo = o.reverse(z)
            xc, lc = c.reverse(z)
        np.testing.assert_allclose(xo.numpy(), xc.numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(lo.numpy(), lc.numpy(), rtol=1e-4, atol=1e-5)


@unittest.skipUnless(
    os.path.exists(os.path.join(REFERENCE_DIR, "model_state_dict.pt")),
    "Reference outputs not generated")
class TestModelLoadParity(unittest.TestCase):
    """Verify reference checkpoint produces identical samples in both codebases."""

    @classmethod
    def setUpClass(cls):
        if ORIGINAL_DIR not in sys.path:
            sys.path.insert(0, ORIGINAL_DIR)
        if TASK_DIR not in sys.path:
            sys.path.insert(0, TASK_DIR)
        from generative_model.realnvpfc_model import RealNVP as Orig_RealNVP
        from DPI_interferometry import Img_logscale as Orig_Img_logscale
        from src.solvers import RealNVP, Img_logscale

        npix, n_flow = 32, 16
        device = torch.device("cpu")
        model_path = os.path.join(REFERENCE_DIR, "model_state_dict.pt")
        scale_path = os.path.join(REFERENCE_DIR, "logscale_state_dict.pt")

        cls.orig_model = Orig_RealNVP(npix * npix, n_flow, affine=True).to(device)
        cls.orig_model.load_state_dict(torch.load(model_path, map_location=device))
        cls.orig_model.eval()
        cls.orig_scale = Orig_Img_logscale(scale=1.0).to(device)
        cls.orig_scale.load_state_dict(torch.load(scale_path, map_location=device))

        cls.clean_model = RealNVP(npix * npix, n_flow, affine=True).to(device)
        cls.clean_model.load_state_dict(torch.load(model_path, map_location=device))
        cls.clean_model.eval()
        cls.clean_scale = Img_logscale(scale=1.0).to(device)
        cls.clean_scale.load_state_dict(torch.load(scale_path, map_location=device))
        cls.npix = npix

    def test_samples_match(self):
        torch.manual_seed(0)
        z = torch.randn(100, self.npix * self.npix)
        with torch.no_grad():
            xo, _ = self.orig_model.reverse(z)
            so = (torch.nn.Softplus()(xo.reshape(-1, 32, 32)) *
                  torch.exp(self.orig_scale.forward())).numpy()

        torch.manual_seed(0)
        z = torch.randn(100, self.npix * self.npix)
        with torch.no_grad():
            xc, _ = self.clean_model.reverse(z)
            sc = (torch.nn.Softplus()(xc.reshape(-1, 32, 32)) *
                  torch.exp(self.clean_scale.forward())).numpy()

        np.testing.assert_allclose(so, sc, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
