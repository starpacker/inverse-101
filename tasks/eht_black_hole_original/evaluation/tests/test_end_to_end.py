"""End-to-end integration test for the closure-only EHT reconstruction pipeline."""
import os
import sys
import json
import unittest
import numpy as np
from numpy.testing import assert_allclose

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, TASK_DIR)

REF_DIR = os.path.join(TASK_DIR, "evaluation", "reference_outputs")


class TestEndToEnd(unittest.TestCase):
    """
    Integration test: run the full pipeline and validate outputs against
    reference results.
    """

    @classmethod
    def setUpClass(cls):
        from src.preprocessing import prepare_data
        from src.physics_model import ClosureForwardModel
        from src.solvers import (
            ClosurePhaseOnlySolver, ClosurePhasePlusAmpSolver,
            VisibilityRMLSolver, TVRegularizer,
        )
        from src.visualization import compute_metrics

        data_dir = os.path.join(TASK_DIR, "data")
        obs, closure_data, metadata = prepare_data(data_dir)

        N = metadata["N"]
        pixel_size_rad = metadata["pixel_size_rad"]
        noise_std = metadata["noise_std"]

        model = ClosureForwardModel(
            uv_coords=obs["uv_coords"],
            image_size=N,
            pixel_size_rad=pixel_size_rad,
            station_ids=obs["station_ids"],
            triangles=closure_data["triangles"],
            quadrangles=closure_data["quadrangles"],
        )

        # Visibility init (used as starting point for closure methods)
        cls.results = {}
        cls.results["vis_corrupted"] = VisibilityRMLSolver(
            regularizers=[(5e3, TVRegularizer())], n_iter=500,
        ).reconstruct(model, obs["vis_corrupted"], noise_std)

        cls.results["vis_true"] = VisibilityRMLSolver(
            regularizers=[(5e3, TVRegularizer())], n_iter=500,
        ).reconstruct(model, obs["vis_true"], noise_std)

        x0 = cls.results["vis_corrupted"]

        cls.results["cp_only_tv"] = ClosurePhaseOnlySolver(
            regularizers=[(1e2, TVRegularizer())],
            alpha_cp=100.0, n_iter=2000,
        ).reconstruct(model, closure_data, x0=x0)

        cls.results["cpca_tv"] = ClosurePhasePlusAmpSolver(
            regularizers=[(1e2, TVRegularizer())],
            alpha_cp=100.0, alpha_ca=100.0, n_iter=2000,
        ).reconstruct(model, closure_data, x0=x0)

        # Load ground truth and compute metrics
        cls.ground_truth = np.load(os.path.join(REF_DIR, "ground_truth.npy"))
        cls.metrics = {}
        for name, recon in cls.results.items():
            cls.metrics[name] = compute_metrics(recon, cls.ground_truth)

    def test_ground_truth_shape(self):
        self.assertEqual(self.ground_truth.shape, (64, 64))

    def test_reconstruction_shapes(self):
        for name, img in self.results.items():
            self.assertEqual(img.shape, (64, 64), msg=f"{name} shape")

    def test_vis_true_beats_vis_corrupted(self):
        """True-visibility RML should beat corrupted-visibility RML."""
        self.assertLess(
            self.metrics["vis_true"]["nrmse"],
            self.metrics["vis_corrupted"]["nrmse"],
        )

    def test_closure_methods_ncc_above_threshold(self):
        """Closure-only methods should achieve NCC > 0.3."""
        for name in ["cp_only_tv", "cpca_tv"]:
            self.assertGreater(
                self.metrics[name]["ncc"], 0.3,
                msg=f"{name} NCC should be > 0.3",
            )

    def test_cpca_tv_reconstruction_reproducible(self):
        """CP+CA (TV) reconstruction should match reference within tolerance."""
        ref_path = os.path.join(REF_DIR, "cpca_tv.npy")
        if os.path.exists(ref_path):
            ref = np.load(ref_path)
            assert_allclose(self.results["cpca_tv"], ref,
                            rtol=1e-3, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
