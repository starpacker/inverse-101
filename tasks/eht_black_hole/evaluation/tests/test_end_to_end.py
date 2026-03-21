"""
End-to-End Integration Test
=============================

Runs the full pipeline and verifies the final reconstruction
matches reference outputs. This is NOT decoupled — it deliberately
tests the entire chain from data loading through reconstruction.

Uses evaluation/reference_outputs/ (not fixtures/).
"""

import os
import sys
import json
import unittest
import numpy as np

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, TASK_DIR)

DATA_DIR = os.path.join(TASK_DIR, "data")
REF_DIR = os.path.join(TASK_DIR, "evaluation", "reference_outputs")


class TestEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.preprocessing import prepare_data
        from src.physics_model import VLBIForwardModel
        from src.solvers import (
            DirtyImageReconstructor, CLEANReconstructor,
            RMLSolver, TVRegularizer, MaxEntropyRegularizer,
        )
        from src.visualization import compute_metrics

        vis, uv, meta = prepare_data(DATA_DIR)
        model = VLBIForwardModel(uv, meta["N"], meta["pixel_size_rad"])
        noise_std = meta["noise_std"]
        cls.gt = np.load(os.path.join(REF_DIR, "ground_truth.npy"))

        cls.reconstructions = {
            "dirty_image": DirtyImageReconstructor().reconstruct(model, vis, noise_std),
            "clean": CLEANReconstructor(
                gain=0.1, n_iter=500, threshold=1e-4, support_radius=15
            ).reconstruct(model, vis, noise_std),
            "rml_tv": RMLSolver(
                regularizers=[(5e3, TVRegularizer())], n_iter=500
            ).reconstruct(model, vis, noise_std),
            "rml_mem": RMLSolver(
                regularizers=[(1e4, MaxEntropyRegularizer())], n_iter=500
            ).reconstruct(model, vis, noise_std),
        }

        cls.metrics = {
            name: compute_metrics(recon, cls.gt)
            for name, recon in cls.reconstructions.items()
        }

        with open(os.path.join(REF_DIR, "metrics.json")) as f:
            cls.ref_metrics = json.load(f)

    def test_rml_tv_nrmse(self):
        self.assertAlmostEqual(
            self.metrics["rml_tv"]["nrmse"],
            self.ref_metrics["rml_tv"]["nrmse"], places=2
        )

    def test_rml_tv_ncc(self):
        self.assertAlmostEqual(
            self.metrics["rml_tv"]["ncc"],
            self.ref_metrics["rml_tv"]["ncc"], places=2
        )

    def test_rml_mem_nrmse(self):
        self.assertAlmostEqual(
            self.metrics["rml_mem"]["nrmse"],
            self.ref_metrics["rml_mem"]["nrmse"], places=2
        )

    def test_rml_beats_dirty(self):
        self.assertLess(
            self.metrics["rml_tv"]["nrmse"],
            self.metrics["dirty_image"]["nrmse"]
        )

    def test_rml_tv_pixel_agreement(self):
        ref = np.load(os.path.join(REF_DIR, "rml_tv.npy"))
        np.testing.assert_allclose(
            self.reconstructions["rml_tv"], ref, rtol=1e-3, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
