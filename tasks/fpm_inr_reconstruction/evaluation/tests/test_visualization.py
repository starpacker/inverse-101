"""Tests for src/visualization.py (includes all-in-focus, metrics, and plotting)"""
import os
import unittest
import numpy as np

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DIR_VIS = os.path.join(TASK_DIR, "evaluation", "fixtures", "visualization")
FIXTURE_DIR_AIF = os.path.join(TASK_DIR, "evaluation", "fixtures", "all_in_focus")

import sys
sys.path.insert(0, TASK_DIR)


# ──────────────────────────────────────────────────────────────────────────────
# All-in-Focus Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestCreateBalanceMap(unittest.TestCase):
    def test_shape(self):
        from src.visualization import create_balance_map
        n_patches, bmap = create_balance_map(256, patch_size=64, patch_pace=16)
        self.assertEqual(bmap.shape, (256, 256))
        self.assertIsInstance(n_patches, int)

    def test_positive_values(self):
        from src.visualization import create_balance_map
        _, bmap = create_balance_map(256, patch_size=64, patch_pace=16)
        self.assertTrue(np.all(bmap > 0))

    def test_center_value(self):
        from src.visualization import create_balance_map
        _, bmap = create_balance_map(256, patch_size=64, patch_pace=16)
        # Center should have highest overlap (lowest balance value)
        center = bmap[128, 128]
        corner = bmap[0, 0]
        self.assertLessEqual(center, corner)


class TestAllInFocusNormalVariance(unittest.TestCase):
    def test_output_shape(self):
        from src.visualization import all_in_focus_normal_variance
        # Create synthetic z-stack where one plane is sharper
        n_z = 5
        size = 256
        z_stack = np.random.rand(size, size, n_z).astype("float32") * 0.1 + 0.5
        result = all_in_focus_normal_variance(z_stack, patch_size=64, patch_pace=16)
        self.assertEqual(result.shape, (size, size))

    def test_does_not_mutate_input(self):
        from src.visualization import all_in_focus_normal_variance
        z_stack = np.random.rand(256, 256, 3).astype("float32") * 0.1 + 0.5
        original = z_stack.copy()
        all_in_focus_normal_variance(z_stack, patch_size=64, patch_pace=16)
        np.testing.assert_array_equal(z_stack, original)


class TestCreateBalanceMapFixture(unittest.TestCase):
    def test_values(self):
        from src.visualization import create_balance_map
        fixture = np.load(
            os.path.join(FIXTURE_DIR_AIF, "create_balance_map.npz"), allow_pickle=True
        )
        n_patches, bmap = create_balance_map(
            int(fixture["input_image_size"]),
            patch_size=int(fixture["input_patch_size"]),
            patch_pace=int(fixture["input_patch_pace"]),
        )
        self.assertEqual(n_patches, int(fixture["output_n_patches"]))
        np.testing.assert_allclose(bmap, fixture["output_balance_map"], rtol=1e-6)


class TestAllInFocusFixture(unittest.TestCase):
    def test_values(self):
        from src.visualization import all_in_focus_normal_variance
        fixture = np.load(
            os.path.join(FIXTURE_DIR_AIF, "all_in_focus.npz"), allow_pickle=True
        )
        result = all_in_focus_normal_variance(
            fixture["input_z_stack"], patch_size=64, patch_pace=16
        )
        np.testing.assert_allclose(result, fixture["output_aif"], rtol=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestComputeMetrics(unittest.TestCase):
    def test_returns_expected_keys(self):
        from src.visualization import compute_metrics
        pred = np.random.rand(10, 32, 32).astype("float32")
        gt = np.random.rand(10, 32, 32).astype("float32")
        result = compute_metrics(pred, gt)
        for key in ["pred_norm", "gt_norm", "l2_per_slice", "mse_per_slice",
                     "psnr_per_slice", "l2_overall", "mse_overall", "psnr_overall"]:
            self.assertIn(key, result)

    def test_shapes(self):
        from src.visualization import compute_metrics
        n_z = 10
        pred = np.random.rand(n_z, 32, 32).astype("float32")
        gt = np.random.rand(n_z, 32, 32).astype("float32")
        result = compute_metrics(pred, gt)
        self.assertEqual(result["l2_per_slice"].shape, (n_z,))
        self.assertEqual(result["psnr_per_slice"].shape, (n_z,))
        self.assertEqual(result["pred_norm"].shape, (n_z, 32, 32))

    def test_perfect_match(self):
        from src.visualization import compute_metrics
        data = np.random.rand(5, 32, 32).astype("float32")
        result = compute_metrics(data, data)
        self.assertAlmostEqual(result["mse_overall"], 0.0, places=10)
        self.assertTrue(result["psnr_overall"] > 50)  # very high PSNR


class TestComputeAllfocusL2(unittest.TestCase):
    def test_returns_expected_keys(self):
        from src.visualization import compute_allfocus_l2
        pred = np.random.rand(64, 64).astype("float32")
        gt = np.random.rand(64, 64).astype("float32")
        result = compute_allfocus_l2(pred, gt)
        self.assertIn("mse", result)
        self.assertIn("psnr", result)

    def test_mean_subtracted(self):
        from src.visualization import compute_allfocus_l2
        # If both images differ only by a constant, MSE should be ~0
        base = np.random.rand(64, 64).astype("float32")
        shifted = base + 5.0
        result = compute_allfocus_l2(base, shifted)
        self.assertAlmostEqual(result["mse"], 0.0, places=6)


class TestComputeSsimPerSlice(unittest.TestCase):
    def test_shape(self):
        from src.visualization import compute_ssim_per_slice
        n_z = 5
        pred = np.random.rand(n_z, 64, 64).astype("float32")
        gt = np.random.rand(n_z, 64, 64).astype("float32")
        result = compute_ssim_per_slice(pred, gt)
        self.assertEqual(result.shape, (n_z,))

    def test_perfect_match(self):
        from src.visualization import compute_ssim_per_slice
        data = np.random.rand(3, 64, 64).astype("float32")
        result = compute_ssim_per_slice(data, data)
        np.testing.assert_allclose(result, 1.0, atol=1e-6)


class TestComputeSsimFixture(unittest.TestCase):
    def test_values(self):
        from src.visualization import compute_ssim_per_slice
        fixture = np.load(
            os.path.join(FIXTURE_DIR_VIS, "compute_ssim_per_slice.npz"), allow_pickle=True
        )
        result = compute_ssim_per_slice(fixture["input_pred"], fixture["input_gt"])
        np.testing.assert_allclose(result, fixture["output_ssim"], rtol=1e-5)


class TestComputeAllfocusL2Fixture(unittest.TestCase):
    def test_values(self):
        from src.visualization import compute_allfocus_l2
        fixture = np.load(
            os.path.join(FIXTURE_DIR_VIS, "compute_allfocus_l2.npz"), allow_pickle=True
        )
        result = compute_allfocus_l2(fixture["input_pred"], fixture["input_gt"])
        np.testing.assert_allclose(result["mse"], float(fixture["output_mse"]), rtol=1e-6)
        np.testing.assert_allclose(result["psnr"], float(fixture["output_psnr"]), rtol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestPlotAmplitudePhase(unittest.TestCase):
    def test_runs_without_error(self):
        import matplotlib
        matplotlib.use("Agg")
        from src.visualization import plot_amplitude_phase
        fig = plot_amplitude_phase(
            np.random.rand(32, 32).astype("float32"),
            np.random.rand(32, 32).astype("float32"),
            epoch=1,
        )
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotPerSliceMetrics(unittest.TestCase):
    def test_runs_without_error(self):
        import matplotlib
        matplotlib.use("Agg")
        from src.visualization import plot_per_slice_metrics
        z = np.linspace(-20, 20, 10)
        fig = plot_per_slice_metrics(
            z, np.random.rand(10), np.random.rand(10), ssim=np.random.rand(10),
        )
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotGtComparison(unittest.TestCase):
    def test_runs_without_error(self):
        import matplotlib
        matplotlib.use("Agg")
        from src.visualization import plot_gt_comparison
        n_z = 5
        fig = plot_gt_comparison(
            np.random.rand(n_z, 32, 32).astype("float32"),
            np.random.rand(n_z, 32, 32).astype("float32"),
            np.linspace(-20, 20, n_z),
            np.random.rand(n_z) * 20 + 10,
            np.random.rand(n_z) * 0.1,
        )
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotAllfocusComparison(unittest.TestCase):
    def test_runs_without_error(self):
        import matplotlib
        matplotlib.use("Agg")
        from src.visualization import plot_allfocus_comparison
        fig = plot_allfocus_comparison(
            np.random.rand(64, 64).astype("float32"),
            np.random.rand(64, 64).astype("float32"),
            0.001,
        )
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
