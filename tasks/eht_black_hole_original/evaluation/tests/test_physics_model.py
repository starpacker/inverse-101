"""Unit tests for physics_model module."""
import os
import sys
import unittest
import numpy as np
from numpy.testing import assert_allclose

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, TASK_DIR)

FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures", "physics_model")


def _build_model(f):
    """Reconstruct ClosureForwardModel from fixture params."""
    from src.physics_model import ClosureForwardModel
    return ClosureForwardModel(
        uv_coords=f["param_uv_coords"],
        image_size=int(f["param_image_size"]),
        pixel_size_rad=float(f["param_pixel_size_rad"]),
        station_ids=f["param_station_ids"],
        triangles=f["param_triangles"],
        quadrangles=f["param_quadrangles"],
    )


class TestInit(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "init.npz"))
        self.model = _build_model(self.f)

    def test_A_shape(self):
        expected_shape = tuple(self.f["output_A_shape"])
        self.assertEqual(self.model.A.shape, expected_shape)

    def test_A_row0(self):
        assert_allclose(self.model.A[0, :], self.f["output_A_row0"], rtol=1e-12)

    def test_A_col0(self):
        assert_allclose(self.model.A[:, 0], self.f["output_A_col0"], rtol=1e-12)

    def test_A_is_complex(self):
        self.assertTrue(np.iscomplexobj(self.model.A))


class TestForward(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "forward.npz"))
        self.model = _build_model(self.f)
        self.result = self.model.forward(self.f["input_image"])

    def test_shape(self):
        self.assertEqual(self.result.shape, self.f["output_vis"].shape)

    def test_dtype(self):
        self.assertTrue(np.iscomplexobj(self.result))

    def test_values(self):
        assert_allclose(self.result, self.f["output_vis"], rtol=1e-10)


class TestAdjoint(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "adjoint.npz"))
        self.model = _build_model(self.f)
        self.result = self.model.adjoint(self.f["input_vis"])

    def test_shape(self):
        N = int(self.f["param_image_size"])
        self.assertEqual(self.result.shape, (N, N))

    def test_real_valued(self):
        self.assertTrue(np.isrealobj(self.result))

    def test_values(self):
        assert_allclose(self.result, self.f["output_image"], rtol=1e-10)

    def test_adjoint_identity(self):
        """Test <Ax, y> == <x, A^H y> (mathematical adjoint property)."""
        rng = np.random.default_rng(0)
        N = int(self.f["param_image_size"])
        x = rng.standard_normal((N, N))
        y = rng.standard_normal(self.model.M) + 1j * rng.standard_normal(self.model.M)
        lhs = np.real(np.vdot(self.model.forward(x), y))
        rhs = np.real(np.vdot(x.ravel(), self.model.adjoint(y).ravel()))
        assert_allclose(lhs, rhs, rtol=1e-10)


class TestDirtyImage(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "dirty_image.npz"))
        self.model = _build_model(self.f)
        self.result = self.model.dirty_image(self.f["input_vis"])

    def test_shape(self):
        N = int(self.f["param_image_size"])
        self.assertEqual(self.result.shape, (N, N))

    def test_values(self):
        assert_allclose(self.result, self.f["output_image"], rtol=1e-10)


class TestPSF(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "psf.npz"))
        self.model = _build_model(self.f)
        self.result = self.model.psf()

    def test_shape(self):
        N = int(self.f["param_image_size"])
        self.assertEqual(self.result.shape, (N, N))

    def test_peak_normalised(self):
        self.assertAlmostEqual(float(self.result.max()), 1.0, places=12)

    def test_values(self):
        assert_allclose(self.result, self.f["output_psf"], rtol=1e-10)


class TestModelClosurePhases(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "model_closure_phases.npz"))
        self.model = _build_model(self.f)
        self.result = self.model.model_closure_phases(self.f["input_image"])

    def test_shape(self):
        self.assertEqual(self.result.shape, self.f["output_cphases"].shape)

    def test_range(self):
        self.assertTrue(np.all(self.result >= -np.pi))
        self.assertTrue(np.all(self.result <= np.pi))

    def test_values(self):
        assert_allclose(self.result, self.f["output_cphases"], rtol=1e-10)


class TestClosurePhaseChisq(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "closure_phase_chisq.npz"))
        self.model = _build_model(self.f)
        self.result = self.model.closure_phase_chisq(
            self.f["input_image"],
            self.f["input_cphases_obs"],
            self.f["input_sigma_cp"],
        )

    def test_scalar(self):
        self.assertIsInstance(self.result, float)

    def test_non_negative(self):
        self.assertGreaterEqual(self.result, 0.0)

    def test_value(self):
        assert_allclose(self.result, float(self.f["output_chisq"]), rtol=1e-10)


class TestClosurePhaseChisqGrad(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR,
                                      "closure_phase_chisq_grad.npz"))
        self.model = _build_model(self.f)
        self.image = self.f["input_image"]
        self.cp_obs = self.f["input_cphases_obs"]
        self.sigma = self.f["input_sigma_cp"]
        self.result = self.model.closure_phase_chisq_grad(
            self.image, self.cp_obs, self.sigma,
        )

    def test_shape(self):
        N = int(self.f["param_image_size"])
        self.assertEqual(self.result.shape, (N, N))

    def test_values(self):
        assert_allclose(self.result, self.f["output_grad"], rtol=1e-10)

    def test_gradient_nonzero(self):
        """Gradient should be non-trivial for a non-matching observation."""
        self.assertGreater(np.abs(self.result).max(), 0.0)


class TestModelLogClosureAmplitudes(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR,
                                      "model_log_closure_amplitudes.npz"))
        self.model = _build_model(self.f)
        self.result = self.model.model_log_closure_amplitudes(self.f["input_image"])

    def test_shape(self):
        self.assertEqual(self.result.shape, self.f["output_log_camps"].shape)

    def test_values(self):
        assert_allclose(self.result, self.f["output_log_camps"], rtol=1e-10)


class TestLogClosureAmpChisq(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR,
                                      "log_closure_amp_chisq.npz"))
        self.model = _build_model(self.f)
        self.result = self.model.log_closure_amp_chisq(
            self.f["input_image"],
            self.f["input_log_camps_obs"],
            self.f["input_sigma_logca"],
        )

    def test_scalar(self):
        self.assertIsInstance(self.result, float)

    def test_non_negative(self):
        self.assertGreaterEqual(self.result, 0.0)

    def test_value(self):
        assert_allclose(self.result, float(self.f["output_chisq"]), rtol=1e-10)


class TestLogClosureAmpChisqGrad(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR,
                                      "log_closure_amp_chisq_grad.npz"))
        self.model = _build_model(self.f)
        self.image = self.f["input_image"]
        self.lca_obs = self.f["input_log_camps_obs"]
        self.sigma = self.f["input_sigma_logca"]
        self.result = self.model.log_closure_amp_chisq_grad(
            self.image, self.lca_obs, self.sigma,
        )

    def test_shape(self):
        N = int(self.f["param_image_size"])
        self.assertEqual(self.result.shape, (N, N))

    def test_values(self):
        assert_allclose(self.result, self.f["output_grad"], rtol=1e-10)

    def test_gradient_nonzero(self):
        """Gradient should be non-trivial for a non-matching observation."""
        self.assertGreater(np.abs(self.result).max(), 0.0)


class TestVisibilityChisq(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "visibility_chisq.npz"))
        self.model = _build_model(self.f)
        self.result = self.model.visibility_chisq(
            self.f["input_image"],
            self.f["input_vis_obs"],
            float(self.f["input_noise_std"]),
        )

    def test_scalar(self):
        self.assertIsInstance(self.result, float)

    def test_value(self):
        assert_allclose(self.result, float(self.f["output_chisq"]), rtol=1e-10)


class TestVisibilityChisqGrad(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR,
                                      "visibility_chisq_grad.npz"))
        self.model = _build_model(self.f)
        self.result = self.model.visibility_chisq_grad(
            self.f["input_image"],
            self.f["input_vis_obs"],
            float(self.f["input_noise_std"]),
        )

    def test_shape(self):
        N = int(self.f["param_image_size"])
        self.assertEqual(self.result.shape, (N, N))

    def test_values(self):
        assert_allclose(self.result, self.f["output_grad"], rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
