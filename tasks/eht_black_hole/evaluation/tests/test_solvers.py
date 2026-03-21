"""
Tests for solvers.py — Decoupled per-function tests
=====================================================

Each test class loads its own fixture from evaluation/fixtures/solvers/
and tests exactly one solver or regularizer.

Fixture naming convention:
  param_*  : model constructor parameters
  input_*  : function arguments
  config_* : solver/regularizer hyperparameters
  output_* : expected return values

Tested functions:
  - DirtyImageReconstructor.reconstruct
  - CLEANReconstructor.reconstruct
  - RMLSolver.reconstruct  (TV variant)
  - RMLSolver.reconstruct  (MEM variant)
  - TVRegularizer.value_and_grad
  - MaxEntropyRegularizer.value_and_grad
  - L1SparsityRegularizer.value_and_grad
"""

import os
import sys
import unittest
import numpy as np

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, TASK_DIR)

FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures", "solvers")


def _build_model(fixture):
    """Build VLBIForwardModel from fixture params."""
    from src.physics_model import VLBIForwardModel
    return VLBIForwardModel(
        uv_coords=fixture["param_uv_coords"],
        image_size=int(fixture["param_image_size"]),
        pixel_size_rad=float(fixture["param_pixel_size_rad"]),
    )


# ═══════════════════════════════════════════════════════════════════════════
# DirtyImageReconstructor
# ═══════════════════════════════════════════════════════════════════════════

class TestDirtyImageReconstructor(unittest.TestCase):
    """
    Fixture: dirty_image_reconstructor.npz
      param_*       : model constructor params
      input_vis     : (M,) complex
      input_noise_std : scalar
      output_image  : (N, N) reconstruction
    """

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "dirty_image_reconstructor.npz"))
        self.model = _build_model(self.f)
        from src.solvers import DirtyImageReconstructor
        self.rec = DirtyImageReconstructor()

    def test_output_shape(self):
        x = self.rec.reconstruct(self.model, self.f["input_vis"], float(self.f["input_noise_std"]))
        self.assertEqual(x.shape, self.f["output_image"].shape)

    def test_output_dtype(self):
        x = self.rec.reconstruct(self.model, self.f["input_vis"], float(self.f["input_noise_std"]))
        self.assertTrue(np.isrealobj(x))

    def test_output_values(self):
        x = self.rec.reconstruct(self.model, self.f["input_vis"], float(self.f["input_noise_std"]))
        np.testing.assert_allclose(x, self.f["output_image"], rtol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# CLEANReconstructor
# ═══════════════════════════════════════════════════════════════════════════

class TestCLEANReconstructor(unittest.TestCase):
    """
    Fixture: clean_reconstructor.npz
      param_*          : model constructor params
      input_vis        : (M,) complex
      input_noise_std  : scalar
      config_gain, config_n_iter, config_threshold, config_support_radius
      output_image     : (N, N) reconstruction
    """

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "clean_reconstructor.npz"))
        self.model = _build_model(self.f)
        from src.solvers import CLEANReconstructor
        self.rec = CLEANReconstructor(
            gain=float(self.f["config_gain"]),
            n_iter=int(self.f["config_n_iter"]),
            threshold=float(self.f["config_threshold"]),
            support_radius=float(self.f["config_support_radius"]),
        )

    def test_output_shape(self):
        x = self.rec.reconstruct(self.model, self.f["input_vis"], float(self.f["input_noise_std"]))
        self.assertEqual(x.shape, self.f["output_image"].shape)

    def test_output_dtype(self):
        x = self.rec.reconstruct(self.model, self.f["input_vis"], float(self.f["input_noise_std"]))
        self.assertTrue(np.isrealobj(x))

    def test_output_values(self):
        x = self.rec.reconstruct(self.model, self.f["input_vis"], float(self.f["input_noise_std"]))
        np.testing.assert_allclose(x, self.f["output_image"], rtol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# RMLSolver (TV)
# ═══════════════════════════════════════════════════════════════════════════

class TestRMLSolverTV(unittest.TestCase):
    """
    Fixture: rml_solver_tv.npz
      param_*          : model constructor params
      input_vis        : (M,) complex
      input_noise_std  : scalar
      config_lambda_tv, config_n_iter
      output_image     : (N, N) reconstruction
    """

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "rml_solver_tv.npz"))
        self.model = _build_model(self.f)
        from src.solvers import RMLSolver, TVRegularizer
        self.rec = RMLSolver(
            regularizers=[(float(self.f["config_lambda_tv"]), TVRegularizer())],
            n_iter=int(self.f["config_n_iter"]),
        )

    def test_output_shape(self):
        x = self.rec.reconstruct(self.model, self.f["input_vis"], float(self.f["input_noise_std"]))
        self.assertEqual(x.shape, self.f["output_image"].shape)

    def test_output_nonnegative(self):
        x = self.rec.reconstruct(self.model, self.f["input_vis"], float(self.f["input_noise_std"]))
        self.assertGreaterEqual(float(x.min()), -1e-10)

    def test_output_values(self):
        x = self.rec.reconstruct(self.model, self.f["input_vis"], float(self.f["input_noise_std"]))
        np.testing.assert_allclose(x, self.f["output_image"], rtol=1e-3, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# RMLSolver (MEM)
# ═══════════════════════════════════════════════════════════════════════════

class TestRMLSolverMEM(unittest.TestCase):
    """
    Fixture: rml_solver_mem.npz
      param_*          : model constructor params
      input_vis        : (M,) complex
      input_noise_std  : scalar
      config_lambda_mem, config_n_iter
      output_image     : (N, N) reconstruction
    """

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "rml_solver_mem.npz"))
        self.model = _build_model(self.f)
        from src.solvers import RMLSolver, MaxEntropyRegularizer
        self.rec = RMLSolver(
            regularizers=[(float(self.f["config_lambda_mem"]), MaxEntropyRegularizer())],
            n_iter=int(self.f["config_n_iter"]),
        )

    def test_output_shape(self):
        x = self.rec.reconstruct(self.model, self.f["input_vis"], float(self.f["input_noise_std"]))
        self.assertEqual(x.shape, self.f["output_image"].shape)

    def test_output_nonnegative(self):
        x = self.rec.reconstruct(self.model, self.f["input_vis"], float(self.f["input_noise_std"]))
        self.assertGreaterEqual(float(x.min()), -1e-10)

    def test_output_values(self):
        x = self.rec.reconstruct(self.model, self.f["input_vis"], float(self.f["input_noise_std"]))
        np.testing.assert_allclose(x, self.f["output_image"], rtol=1e-3, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# TVRegularizer
# ═══════════════════════════════════════════════════════════════════════════

class TestTVRegularizer(unittest.TestCase):
    """
    Fixture: tv_regularizer.npz
      input_image     : (N, N) test image
      config_epsilon  : scalar
      output_val      : scalar — TV(x)
      output_grad     : (N, N) — ∂TV/∂x
    """

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "tv_regularizer.npz"))
        from src.solvers import TVRegularizer
        self.reg = TVRegularizer(epsilon=float(self.f["config_epsilon"]))

    def test_val_type(self):
        val, _ = self.reg.value_and_grad(self.f["input_image"])
        self.assertIsInstance(val, float)

    def test_grad_shape(self):
        _, grad = self.reg.value_and_grad(self.f["input_image"])
        self.assertEqual(grad.shape, self.f["input_image"].shape)

    def test_val_value(self):
        val, _ = self.reg.value_and_grad(self.f["input_image"])
        np.testing.assert_allclose(val, float(self.f["output_val"]), rtol=1e-10)

    def test_grad_values(self):
        _, grad = self.reg.value_and_grad(self.f["input_image"])
        np.testing.assert_allclose(grad, self.f["output_grad"], rtol=1e-10)

    def test_zero_image_gives_zero_val(self):
        x_zero = np.zeros_like(self.f["input_image"])
        val, _ = self.reg.value_and_grad(x_zero)
        self.assertAlmostEqual(val, 0.0, places=5)


# ═══════════════════════════════════════════════════════════════════════════
# MaxEntropyRegularizer
# ═══════════════════════════════════════════════════════════════════════════

class TestMaxEntropyRegularizer(unittest.TestCase):
    """
    Fixture: max_entropy_regularizer.npz
      input_image     : (N, N) test image
      config_epsilon  : scalar
      output_val      : scalar — R(x)
      output_grad     : (N, N) — ∂R/∂x
    """

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "max_entropy_regularizer.npz"))
        from src.solvers import MaxEntropyRegularizer
        self.reg = MaxEntropyRegularizer(epsilon=float(self.f["config_epsilon"]))

    def test_val_type(self):
        val, _ = self.reg.value_and_grad(self.f["input_image"])
        self.assertIsInstance(val, float)

    def test_grad_shape(self):
        _, grad = self.reg.value_and_grad(self.f["input_image"])
        self.assertEqual(grad.shape, self.f["input_image"].shape)

    def test_val_value(self):
        val, _ = self.reg.value_and_grad(self.f["input_image"])
        np.testing.assert_allclose(val, float(self.f["output_val"]), rtol=1e-10)

    def test_grad_values(self):
        _, grad = self.reg.value_and_grad(self.f["input_image"])
        np.testing.assert_allclose(grad, self.f["output_grad"], rtol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# L1SparsityRegularizer
# ═══════════════════════════════════════════════════════════════════════════

class TestL1SparsityRegularizer(unittest.TestCase):
    """
    Fixture: l1_sparsity_regularizer.npz
      input_image     : (N, N) test image
      config_epsilon  : scalar
      output_val      : scalar — R(x)
      output_grad     : (N, N) — ∂R/∂x
    """

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "l1_sparsity_regularizer.npz"))
        from src.solvers import L1SparsityRegularizer
        self.reg = L1SparsityRegularizer(epsilon=float(self.f["config_epsilon"]))

    def test_val_type(self):
        val, _ = self.reg.value_and_grad(self.f["input_image"])
        self.assertIsInstance(val, float)

    def test_grad_shape(self):
        _, grad = self.reg.value_and_grad(self.f["input_image"])
        self.assertEqual(grad.shape, self.f["input_image"].shape)

    def test_val_value(self):
        val, _ = self.reg.value_and_grad(self.f["input_image"])
        np.testing.assert_allclose(val, float(self.f["output_val"]), rtol=1e-10)

    def test_grad_values(self):
        _, grad = self.reg.value_and_grad(self.f["input_image"])
        np.testing.assert_allclose(grad, self.f["output_grad"], rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
