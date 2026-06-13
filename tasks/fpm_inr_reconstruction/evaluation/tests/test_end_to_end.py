"""End-to-end test: load trained model, evaluate, check metrics match reference."""
import os
import json
import unittest
import numpy as np
import torch
import torch.nn.functional as F

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF_DIR = os.path.join(TASK_DIR, "evaluation", "reference_outputs")

import sys
sys.path.insert(0, TASK_DIR)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestEndToEnd(unittest.TestCase):
    """Verify that trained model reproduces reference metrics."""

    @classmethod
    def setUpClass(cls):
        from src.solvers import FullModel, load_model_with_required_grad, FPMINRSolver
        from src.preprocessing import load_ground_truth
        from src.visualization import compute_metrics, compute_ssim_per_slice

        device = "cuda:0"
        data_dir = os.path.join(TASK_DIR, "data")

        # Load model
        model = FullModel(
            w=2048, h=2048, num_feats=32, x_mode=512, y_mode=512,
            z_min=-20.0, z_max=20.0, ds_factor=1, use_layernorm=False,
        ).to(device)
        load_model_with_required_grad(
            model, os.path.join(REF_DIR, "model_weights.pth")
        )

        # Load GT
        gt_data = load_ground_truth(data_dir)
        gt_stack = gt_data["I_stack"]
        gt_zvec = gt_data["zvec"]

        # Evaluate
        solver = FPMINRSolver()
        pred_ampli, _ = solver.evaluate(model, gt_zvec, device=device)

        # Resize to GT resolution
        pred_tensor = torch.from_numpy(pred_ampli).unsqueeze(1)
        pred_tensor = F.interpolate(
            pred_tensor, size=(gt_stack.shape[0], gt_stack.shape[0]),
            mode="bilinear", align_corners=False,
        )
        pred_ampli_resized = pred_tensor.squeeze(1).numpy()

        gt_ampli = np.transpose(gt_stack, (2, 0, 1))
        cls.metrics = compute_metrics(pred_ampli_resized, gt_ampli)
        cls.ssim = compute_ssim_per_slice(
            cls.metrics["pred_norm"], cls.metrics["gt_norm"]
        )

        with open(os.path.join(REF_DIR, "metrics.json")) as f:
            cls.ref = json.load(f)

    def test_psnr_overall(self):
        np.testing.assert_allclose(
            self.metrics["psnr_overall"], self.ref["overall_psnr"], rtol=0.01
        )

    def test_psnr_per_slice_mean(self):
        np.testing.assert_allclose(
            self.metrics["psnr_per_slice"].mean(),
            self.ref["per_slice_psnr_mean"],
            rtol=0.01,
        )

    def test_ssim_per_slice_mean(self):
        np.testing.assert_allclose(
            self.ssim.mean(), self.ref["per_slice_ssim_mean"], rtol=0.01
        )

    def test_mse_overall(self):
        np.testing.assert_allclose(
            self.metrics["mse_overall"], self.ref["overall_mse"], rtol=0.01
        )


if __name__ == "__main__":
    unittest.main()
