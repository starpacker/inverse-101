"""Unit tests for solvers module."""
import os
import sys
import unittest
import numpy as np
from numpy.testing import assert_allclose

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, TASK_DIR)

FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures", "solvers")


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


# ═══════════════════════════════════════════════════════════════════════════
# Regularizer tests (deterministic — exact comparison)
# ═══════════════════════════════════════════════════════════════════════════

class TestTVRegularizer(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "tv_regularizer.npz"))
        from src.solvers import TVRegularizer
        reg = TVRegularizer(epsilon=float(self.f["config_epsilon"]))
        self.val, self.grad = reg.value_and_grad(self.f["input_image"])

    def test_val_is_float(self):
        self.assertIsInstance(self.val, float)

    def test_val_non_negative(self):
        self.assertGreaterEqual(self.val, 0.0)

    def test_val_match(self):
        assert_allclose(self.val, float(self.f["output_val"]), rtol=1e-10)

    def test_grad_shape(self):
        self.assertEqual(self.grad.shape, self.f["input_image"].shape)

    def test_grad_match(self):
        assert_allclose(self.grad, self.f["output_grad"], rtol=1e-10)

    def test_zero_image(self):
        """TV of a constant image should be approximately zero."""
        from src.solvers import TVRegularizer
        reg = TVRegularizer(epsilon=1e-6)
        v, g = reg.value_and_grad(np.ones((8, 8)))
        self.assertAlmostEqual(v, 0.0, places=5)


class TestMaxEntropyRegularizer(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR,
                                      "max_entropy_regularizer.npz"))
        from src.solvers import MaxEntropyRegularizer
        reg = MaxEntropyRegularizer(epsilon=float(self.f["config_epsilon"]))
        self.val, self.grad = reg.value_and_grad(self.f["input_image"])

    def test_val_is_float(self):
        self.assertIsInstance(self.val, float)

    def test_val_match(self):
        assert_allclose(self.val, float(self.f["output_val"]), rtol=1e-10)

    def test_grad_match(self):
        assert_allclose(self.grad, self.f["output_grad"], rtol=1e-10)


class TestL1SparsityRegularizer(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR,
                                      "l1_sparsity_regularizer.npz"))
        from src.solvers import L1SparsityRegularizer
        reg = L1SparsityRegularizer(epsilon=float(self.f["config_epsilon"]))
        self.val, self.grad = reg.value_and_grad(self.f["input_image"])

    def test_val_is_float(self):
        self.assertIsInstance(self.val, float)

    def test_val_non_negative(self):
        self.assertGreaterEqual(self.val, 0.0)

    def test_val_match(self):
        assert_allclose(self.val, float(self.f["output_val"]), rtol=1e-10)

    def test_grad_match(self):
        assert_allclose(self.grad, self.f["output_grad"], rtol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# Solver tests (iterative — relaxed tolerance)
# ═══════════════════════════════════════════════════════════════════════════

class TestClosurePhaseOnlySolver(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR,
                                      "closure_phase_only_solver.npz"))
        model = _build_model(self.f)

        from src.solvers import ClosurePhaseOnlySolver, TVRegularizer
        closure_data = dict(
            cphases=self.f["input_cphases"],
            sigma_cp=self.f["input_sigma_cp"],
            triangles=self.f["param_triangles"],
            quadrangles=self.f["param_quadrangles"],
        )
        solver = ClosurePhaseOnlySolver(
            regularizers=[(float(self.f["config_lambda_tv"]), TVRegularizer())],
            alpha_cp=float(self.f["config_alpha_cp"]),
            n_iter=int(self.f["config_n_iter"]),
        )
        self.result = solver.reconstruct(model, closure_data)

    def test_shape(self):
        N = int(self.f["param_image_size"])
        self.assertEqual(self.result.shape, (N, N))

    def test_non_negative(self):
        self.assertGreaterEqual(float(self.result.min()), -1e-10)

    def test_values(self):
        assert_allclose(self.result, self.f["output_image"],
                        rtol=1e-3, atol=1e-6)


class TestClosurePhasePlusAmpSolver(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR,
                                      "closure_phase_plus_amp_solver.npz"))
        model = _build_model(self.f)

        from src.solvers import ClosurePhasePlusAmpSolver, TVRegularizer
        closure_data = dict(
            cphases=self.f["input_cphases"],
            log_camps=self.f["input_log_camps"],
            sigma_cp=self.f["input_sigma_cp"],
            sigma_logca=self.f["input_sigma_logca"],
            triangles=self.f["param_triangles"],
            quadrangles=self.f["param_quadrangles"],
        )
        solver = ClosurePhasePlusAmpSolver(
            regularizers=[(float(self.f["config_lambda_tv"]), TVRegularizer())],
            alpha_cp=float(self.f["config_alpha_cp"]),
            alpha_ca=float(self.f["config_alpha_ca"]),
            n_iter=int(self.f["config_n_iter"]),
        )
        self.result = solver.reconstruct(model, closure_data)

    def test_shape(self):
        N = int(self.f["param_image_size"])
        self.assertEqual(self.result.shape, (N, N))

    def test_non_negative(self):
        self.assertGreaterEqual(float(self.result.min()), -1e-10)

    def test_values(self):
        assert_allclose(self.result, self.f["output_image"],
                        rtol=1e-3, atol=1e-6)


class TestVisibilityRMLSolver(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR,
                                      "visibility_rml_solver.npz"))
        model = _build_model(self.f)

        from src.solvers import VisibilityRMLSolver, TVRegularizer
        solver = VisibilityRMLSolver(
            regularizers=[(float(self.f["config_lambda_tv"]), TVRegularizer())],
            n_iter=int(self.f["config_n_iter"]),
        )
        self.result = solver.reconstruct(
            model, self.f["input_vis"],
            noise_std=float(self.f["input_noise_std"]),
        )

    def test_shape(self):
        N = int(self.f["param_image_size"])
        self.assertEqual(self.result.shape, (N, N))

    def test_non_negative(self):
        self.assertGreaterEqual(float(self.result.min()), -1e-10)

    def test_values(self):
        assert_allclose(self.result, self.f["output_image"],
                        rtol=1e-3, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
