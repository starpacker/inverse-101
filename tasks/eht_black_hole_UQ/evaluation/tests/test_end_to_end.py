"""
End-to-end integration tests for DPI pipeline.

Two test classes:
1. TestEndToEnd — quick smoke test with reduced epochs (verifies pipeline runs)
2. TestPosteriorQuality — evaluates pretrained model's posterior against reference
   thresholds, including multi-modal detection, per-mode quality, and data fidelity.
"""

import os
import unittest
import numpy as np

TASK_DIR = os.path.join(os.path.dirname(__file__), "..", "..")


class TestEndToEnd(unittest.TestCase):
    """Integration test: full DPI pipeline with minimal training."""

    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, TASK_DIR)
        import torch
        from src.preprocessing import prepare_data, load_ground_truth
        from src.solvers import DPISolver

        (obs, obs_data, closure_indices, nufft_params,
         prior_image, flux_const, metadata) = prepare_data(
            os.path.join(TASK_DIR, "data"))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        solver = DPISolver(
            npix=metadata["npix"],
            n_flow=4,
            n_epoch=200,
            batch_size=8,
            device=device,
        )

        cls.result = solver.reconstruct(
            obs_data, closure_indices, nufft_params, prior_image, flux_const)
        cls.posterior = solver.posterior_statistics(n_samples=20)
        cls.gt = load_ground_truth(
            os.path.join(TASK_DIR, "data"),
            metadata["npix"], metadata["fov_uas"])
        cls.metadata = metadata

    def test_loss_history_exists(self):
        self.assertIn('loss_history', self.result)
        self.assertEqual(len(self.result['loss_history']['total']), 200)

    def test_loss_decreases(self):
        losses = self.result['loss_history']['total']
        initial_mean = np.mean(losses[:20])
        final_mean = np.mean(losses[-20:])
        self.assertLess(final_mean, initial_mean)

    def test_posterior_mean_shape(self):
        npix = self.metadata["npix"]
        self.assertEqual(self.posterior['mean'].shape, (npix, npix))

    def test_posterior_std_shape(self):
        npix = self.metadata["npix"]
        self.assertEqual(self.posterior['std'].shape, (npix, npix))

    def test_posterior_samples_shape(self):
        npix = self.metadata["npix"]
        self.assertEqual(self.posterior['samples'].shape, (20, npix, npix))

    def test_posterior_positivity(self):
        self.assertTrue(np.all(self.posterior['samples'] >= 0))

    def test_posterior_mean_positive(self):
        self.assertTrue(np.all(self.posterior['mean'] >= 0))

    def test_metrics_computable(self):
        import sys
        sys.path.insert(0, TASK_DIR)
        from src.visualization import compute_metrics
        metrics = compute_metrics(self.posterior['mean'], self.gt)
        self.assertIn('nrmse', metrics)
        self.assertIn('ncc', metrics)
        self.assertTrue(0 <= metrics['nrmse'] <= 10)


# ── Posterior Quality Tests (pretrained model) ────────────────────────────

class TestPosteriorQuality(unittest.TestCase):
    """
    Evaluate the quality of the trained posterior from reference_outputs/.

    These tests verify that an LLM-produced DPI implementation:
    1. Produces a reasonable posterior mean (NRMSE, NCC vs ground truth)
    2. Captures multi-modal structure (silhouette score, mode balance)
    3. Each mode is physically plausible (per-mode NRMSE/NCC)
    4. Fits the observed data (closure phase/amplitude RMSE)
    5. Provides non-degenerate uncertainty estimates
    """

    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, TASK_DIR)
        import torch

        ref_dir = os.path.join(TASK_DIR, "evaluation", "reference_outputs")
        samples_path = os.path.join(ref_dir, "posterior_samples.npy")
        if not os.path.exists(samples_path):
            raise unittest.SkipTest("Reference outputs not generated yet")

        cls.samples = np.load(samples_path)
        cls.mean = np.load(os.path.join(ref_dir, "posterior_mean.npy"))
        cls.std = np.load(os.path.join(ref_dir, "posterior_std.npy"))
        cls.gt = np.load(os.path.join(ref_dir, "ground_truth.npy"))

        from src.visualization import compute_metrics
        cls.metrics_all = compute_metrics(cls.mean, cls.gt)

        # Multi-modal analysis
        from sklearn.manifold import TSNE
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        flat = cls.samples.reshape(len(cls.samples), -1)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        emb = tsne.fit_transform(flat)
        km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(emb)
        cls.labels = km.labels_
        cls.silhouette = silhouette_score(emb, cls.labels)

        cls.mean_mode0 = cls.samples[cls.labels == 0].mean(0)
        cls.mean_mode1 = cls.samples[cls.labels == 1].mean(0)
        cls.metrics_mode0 = compute_metrics(cls.mean_mode0, cls.gt)
        cls.metrics_mode1 = compute_metrics(cls.mean_mode1, cls.gt)

        n0 = (cls.labels == 0).sum()
        n1 = (cls.labels == 1).sum()
        cls.mode_fraction_min = min(n0, n1) / len(cls.labels)

        # Data fidelity
        from src.preprocessing import prepare_data
        from src.physics_model import NUFFTForwardModel
        (obs, obs_data, closure_indices, nufft_params,
         prior_image, flux_const, metadata) = prepare_data(
            os.path.join(TASK_DIR, "data"))
        cls.flux_const = flux_const
        cls.npix = metadata["npix"]

        cphase_ind_torch = [torch.tensor(a, dtype=torch.long)
                            for a in closure_indices["cphase_ind_list"]]
        cphase_sign_torch = [torch.tensor(a, dtype=torch.float32)
                             for a in closure_indices["cphase_sign_list"]]
        camp_ind_torch = [torch.tensor(a, dtype=torch.long)
                          for a in closure_indices["camp_ind_list"]]

        fwd = NUFFTForwardModel(
            cls.npix, nufft_params["ktraj_vis"], nufft_params["pulsefac_vis"],
            cphase_ind_torch, cphase_sign_torch, camp_ind_torch,
            torch.device("cpu"))

        cphase_obs = np.array(closure_indices["cphase_data"]["cphase"])
        logcamp_obs = np.array(closure_indices["logcamp_data"]["camp"])

        def _data_fidelity(img):
            t = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
            _, _, cp, lca = fwd(t)
            cp_np = cp[0].detach().numpy()
            lca_np = lca[0].detach().numpy()
            diff_rad = (cphase_obs - cp_np) * np.pi / 180
            cp_rmse = float(np.sqrt(np.mean(2 * (1 - np.cos(diff_rad))))) * 180 / np.pi
            lca_rmse = float(np.sqrt(np.mean((logcamp_obs - lca_np) ** 2)))
            return cp_rmse, lca_rmse

        cls.cp_rmse_all, cls.lca_rmse_all = _data_fidelity(cls.mean)
        cls.cp_rmse_m0, cls.lca_rmse_m0 = _data_fidelity(cls.mean_mode0)
        cls.cp_rmse_m1, cls.lca_rmse_m1 = _data_fidelity(cls.mean_mode1)

    # ── Layer 1: Basic posterior statistics ────────────────────────────

    def test_posterior_mean_nrmse(self):
        """Posterior mean NRMSE vs GT should be < 0.25."""
        self.assertLess(self.metrics_all['nrmse'], 0.25)

    def test_posterior_mean_ncc_positive(self):
        """Posterior mean should be positively correlated with GT."""
        self.assertGreater(self.metrics_all['ncc'], 0.3)

    def test_samples_non_negative(self):
        """All posterior samples must be non-negative (softplus ensures this)."""
        self.assertTrue(np.all(self.samples >= 0))

    def test_flux_conservation(self):
        """Posterior mean flux should be within 20% of observed flux."""
        flux_err = abs(self.mean.sum() - self.flux_const) / self.flux_const
        self.assertLess(flux_err, 0.20)

    # ── Layer 2: Multi-modal posterior ─────────────────────────────────

    def test_silhouette_score(self):
        """t-SNE + K-Means(k=2) silhouette > 0.3 — posterior is multi-modal."""
        self.assertGreater(self.silhouette, 0.3)

    def test_mode_balance(self):
        """Smaller mode must contain > 10% of samples (not degenerate)."""
        self.assertGreater(self.mode_fraction_min, 0.10)

    def test_mode0_nrmse(self):
        """Mode 0 mean NRMSE vs GT should be < 0.30."""
        self.assertLess(self.metrics_mode0['nrmse'], 0.30)

    def test_mode1_nrmse(self):
        """Mode 1 mean NRMSE vs GT should be < 0.30."""
        self.assertLess(self.metrics_mode1['nrmse'], 0.30)

    def test_mode0_ncc_positive(self):
        """Mode 0 mean should be positively correlated with GT."""
        self.assertGreater(self.metrics_mode0['ncc'], 0.0)

    def test_mode1_ncc_positive(self):
        """Mode 1 mean should be positively correlated with GT."""
        self.assertGreater(self.metrics_mode1['ncc'], 0.0)

    # ── Layer 3: Data fidelity ─────────────────────────────────────────

    def test_per_mode_cphase_rmse(self):
        """At least one mode should fit closure phases reasonably (< 45 deg)."""
        best = min(self.cp_rmse_m0, self.cp_rmse_m1)
        self.assertLess(best, 45.0)

    def test_per_mode_logcamp_rmse(self):
        """At least one mode should fit log closure amps reasonably (< 0.6)."""
        best = min(self.lca_rmse_m0, self.lca_rmse_m1)
        self.assertLess(best, 0.6)

    # ── Layer 4: Uncertainty quality ───────────────────────────────────

    def test_uncertainty_nonzero(self):
        """Posterior std must be non-zero (not degenerate)."""
        self.assertGreater(self.std.mean(), 0.0)

    def test_uncertainty_spatial_structure(self):
        """Uncertainty should be higher near the ring than at image corners."""
        npix = self.npix
        c = npix // 2
        # Ring region: annulus 5-12 pixels from center
        y, x = np.mgrid[0:npix, 0:npix]
        r = np.sqrt((x - c + 0.5)**2 + (y - c + 0.5)**2)
        ring_mask = (r >= 5) & (r <= 12)
        corner_mask = r > 14
        ring_std = self.std[ring_mask].mean()
        corner_std = self.std[corner_mask].mean()
        self.assertGreater(ring_std, corner_std)


if __name__ == "__main__":
    unittest.main()
