"""
Tests for physics_model.py — Decoupled per-function tests
===========================================================

Each test class loads its own fixture from evaluation/fixtures/physics_model/
and tests exactly one method of VLBIForwardModel.

Fixture naming convention:
  param_*  : constructor parameters (to build the model object)
  input_*  : function arguments
  output_* : expected return values

Tested functions:
  - VLBIForwardModel.__init__
  - VLBIForwardModel.forward
  - VLBIForwardModel.adjoint
  - VLBIForwardModel.dirty_image
  - VLBIForwardModel.psf
  - VLBIForwardModel.add_noise
"""

import os
import sys
import unittest
import numpy as np

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, TASK_DIR)

FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures", "physics_model")


def _build_model(fixture):
    """Build VLBIForwardModel from fixture params."""
    from src.physics_model import VLBIForwardModel
    return VLBIForwardModel(
        uv_coords=fixture["param_uv_coords"],
        image_size=int(fixture["param_image_size"]),
        pixel_size_rad=float(fixture["param_pixel_size_rad"]),
    )


# ═══════════════════════════════════════════════════════════════════════════
# __init__
# ═══════════════════════════════════════════════════════════════════════════

class TestInit(unittest.TestCase):
    """
    Fixture: init.npz
      input_uv_coords, input_image_size, input_pixel_size_rad
      output_A_shape, output_A_row0, output_A_col0, output_A_diagonal
    """

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "init.npz"))
        from src.physics_model import VLBIForwardModel
        self.model = VLBIForwardModel(
            self.f["input_uv_coords"],
            int(self.f["input_image_size"]),
            float(self.f["input_pixel_size_rad"]),
        )

    def test_A_shape(self):
        expected = tuple(self.f["output_A_shape"])
        self.assertEqual(self.model.A.shape, expected)

    def test_A_dtype(self):
        self.assertTrue(np.iscomplexobj(self.model.A))

    def test_A_first_row(self):
        np.testing.assert_allclose(self.model.A[0, :], self.f["output_A_row0"], rtol=1e-12)

    def test_A_first_col(self):
        np.testing.assert_allclose(self.model.A[:, 0], self.f["output_A_col0"], rtol=1e-12)

    def test_A_diagonal(self):
        n = len(self.f["output_A_diagonal"])
        actual = np.diag(self.model.A[:n, :n])
        np.testing.assert_allclose(actual, self.f["output_A_diagonal"], rtol=1e-12)


# ═══════════════════════════════════════════════════════════════════════════
# forward
# ═══════════════════════════════════════════════════════════════════════════

class TestForward(unittest.TestCase):
    """
    Fixture: forward.npz
      param_uv_coords, param_image_size, param_pixel_size_rad
      input_image   : (N, N) ground truth image
      output_vis    : (M,) complex visibilities
    """

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "forward.npz"))
        self.model = _build_model(self.f)

    def test_output_shape(self):
        vis = self.model.forward(self.f["input_image"])
        self.assertEqual(vis.shape, self.f["output_vis"].shape)

    def test_output_dtype(self):
        vis = self.model.forward(self.f["input_image"])
        self.assertTrue(np.iscomplexobj(vis))

    def test_output_values(self):
        vis = self.model.forward(self.f["input_image"])
        np.testing.assert_allclose(vis, self.f["output_vis"], rtol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# adjoint
# ═══════════════════════════════════════════════════════════════════════════

class TestAdjoint(unittest.TestCase):
    """
    Fixture: adjoint.npz
      param_uv_coords, param_image_size, param_pixel_size_rad
      input_vis     : (M,) complex visibilities
      output_image  : (N, N) real image
    """

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "adjoint.npz"))
        self.model = _build_model(self.f)

    def test_output_shape(self):
        img = self.model.adjoint(self.f["input_vis"])
        self.assertEqual(img.shape, self.f["output_image"].shape)

    def test_output_dtype(self):
        img = self.model.adjoint(self.f["input_vis"])
        self.assertTrue(np.isrealobj(img))

    def test_output_values(self):
        img = self.model.adjoint(self.f["input_vis"])
        np.testing.assert_allclose(img, self.f["output_image"], rtol=1e-10)

    def test_adjoint_identity(self):
        """Mathematical property: <Ax, y> == <x, A^H y> for random x, y."""
        rng = np.random.default_rng(123)
        N = int(self.f["param_image_size"])
        M = len(self.f["input_vis"])
        x = rng.standard_normal((N, N))
        y = rng.standard_normal(M) + 1j * rng.standard_normal(M)

        lhs = np.real(np.vdot(self.model.forward(x), y))
        rhs = np.real(np.vdot(x.ravel(), self.model.adjoint(y).ravel()))
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# dirty_image
# ═══════════════════════════════════════════════════════════════════════════

class TestDirtyImage(unittest.TestCase):
    """
    Fixture: dirty_image.npz
      param_uv_coords, param_image_size, param_pixel_size_rad
      input_vis     : (M,) complex visibilities
      output_image  : (N, N) normalized dirty image
    """

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "dirty_image.npz"))
        self.model = _build_model(self.f)

    def test_output_shape(self):
        img = self.model.dirty_image(self.f["input_vis"])
        self.assertEqual(img.shape, self.f["output_image"].shape)

    def test_output_dtype(self):
        img = self.model.dirty_image(self.f["input_vis"])
        self.assertTrue(np.isrealobj(img))

    def test_output_values(self):
        img = self.model.dirty_image(self.f["input_vis"])
        np.testing.assert_allclose(img, self.f["output_image"], rtol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# psf
# ═══════════════════════════════════════════════════════════════════════════

class TestPSF(unittest.TestCase):
    """
    Fixture: psf.npz
      param_uv_coords, param_image_size, param_pixel_size_rad
      output_psf  : (N, N) point spread function
    """

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "psf.npz"))
        self.model = _build_model(self.f)

    def test_output_shape(self):
        psf = self.model.psf()
        self.assertEqual(psf.shape, self.f["output_psf"].shape)

    def test_peak_is_one(self):
        psf = self.model.psf()
        self.assertAlmostEqual(float(psf.max()), 1.0, places=12)

    def test_output_values(self):
        psf = self.model.psf()
        np.testing.assert_allclose(psf, self.f["output_psf"], rtol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# add_noise
# ═══════════════════════════════════════════════════════════════════════════

class TestAddNoise(unittest.TestCase):
    """
    Fixture: add_noise.npz
      param_uv_coords, param_image_size, param_pixel_size_rad
      input_vis_clean  : (M,) complex — noiseless visibilities
      input_snr        : scalar — target SNR
      output_noise_std : scalar — deterministic: signal_rms / snr

    Note: vis_noisy values are random, so we test statistical properties
    rather than exact values. Only noise_std is deterministic.
    """

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "add_noise.npz"))
        self.model = _build_model(self.f)
        self.vis_clean = self.f["input_vis_clean"]
        self.snr = float(self.f["input_snr"])

    def test_returns_tuple_of_two(self):
        result = self.model.add_noise(self.vis_clean, snr=self.snr)
        self.assertEqual(len(result), 2)

    def test_noise_std_deterministic(self):
        """noise_std = signal_rms / snr, independent of random seed."""
        _, noise_std = self.model.add_noise(self.vis_clean, snr=self.snr)
        np.testing.assert_allclose(
            noise_std, float(self.f["output_noise_std"]), rtol=1e-10
        )

    def test_output_shape(self):
        vis_noisy, _ = self.model.add_noise(self.vis_clean, snr=self.snr)
        self.assertEqual(vis_noisy.shape, self.vis_clean.shape)

    def test_output_is_complex(self):
        vis_noisy, _ = self.model.add_noise(self.vis_clean, snr=self.snr)
        self.assertTrue(np.iscomplexobj(vis_noisy))

    def test_noise_mean_near_zero(self):
        """Noise should have approximately zero mean (statistical test)."""
        vis_noisy, _ = self.model.add_noise(self.vis_clean, snr=self.snr)
        noise = vis_noisy - self.vis_clean
        # With M=540 samples, |mean| should be well within 5σ/√M
        _, noise_std = self.model.add_noise(self.vis_clean, snr=self.snr)
        threshold = 5 * noise_std / np.sqrt(len(noise))
        self.assertLess(abs(noise.mean()), threshold)

    def test_noise_std_matches_expected(self):
        """Empirical noise std should be close to the reported noise_std."""
        vis_noisy, noise_std = self.model.add_noise(self.vis_clean, snr=self.snr)
        noise = vis_noisy - self.vis_clean
        empirical_std = np.std(noise) / np.sqrt(2)  # each of real/imag has std=noise_std
        np.testing.assert_allclose(empirical_std, noise_std, rtol=0.3)  # 30% for M=540

    def test_different_calls_give_different_noise(self):
        """Two calls without fixed seed should produce different noise."""
        v1, _ = self.model.add_noise(self.vis_clean, snr=self.snr)
        v2, _ = self.model.add_noise(self.vis_clean, snr=self.snr)
        self.assertFalse(np.allclose(v1, v2))


if __name__ == "__main__":
    unittest.main()
