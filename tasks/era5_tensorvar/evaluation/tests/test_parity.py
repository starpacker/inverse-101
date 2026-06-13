"""Parity tests: cleaned `src/` outputs must match the captured originals.

Fixtures in `evaluation/fixtures/parity/` were produced by running the
upstream Tensor-Var ERA5 notebook with num_mc=1, ass_w=ass_T=5, seed=0,
start_index=750. Every cleaned function is checked against those captured
intermediates so any silent regression in the port is caught immediately.

Run with `python -m pytest evaluation/tests/test_parity.py -v`.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

TASK_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK_DIR))

from src.preprocessing import (
    default_covariances,
    download_pretrained_weights,
    load_observation,
    load_pretrained_models,
    select_sample,
)
from src.solvers import qp_solver_latent, tensor_var_4dvar

PARITY_DIR = TASK_DIR / "evaluation" / "fixtures" / "parity"
WEIGHTS_DIR = TASK_DIR / "evaluation" / "checkpoints"
DEVICE = "cpu"
RTOL = 5e-5  # convolutional surrogates use float32; loose float-equality bound
ATOL = 5e-6


def _have_parity_fixtures() -> bool:
    return (PARITY_DIR / "decoded_traj.npy").exists()


@unittest.skipUnless(_have_parity_fixtures(), "parity fixtures missing")
class TestPretrainedParity(unittest.TestCase):
    """Run the cleaned 4D-Var loop and compare every intermediate to the originals."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        np.random.seed(0)
        download_pretrained_weights(WEIGHTS_DIR)
        cls.forward_model, cls.inverse_model, cls.z_b = load_pretrained_models(
            WEIGHTS_DIR, DEVICE
        )
        cls.forward_model.eval()
        cls.inverse_model.eval()

        raw = select_sample(load_observation(TASK_DIR / "data"), 0)
        cls.obs_history = torch.tensor(raw["obs_history"], dtype=torch.float32, device=DEVICE)

        cls.expected_inv = np.load(PARITY_DIR / "inv_obs_seq_z.npy")
        cls.expected_K_S = np.load(PARITY_DIR / "K_S_seq_z.npy")
        cls.expected_qp = np.load(PARITY_DIR / "qp_result.npy")
        cls.expected_decoded = np.load(PARITY_DIR / "decoded_traj.npy")
        cls.expected_C_forward = np.load(PARITY_DIR / "C_forward.npy")
        cls.expected_z_b = np.load(PARITY_DIR / "z_b.npy")

        cls.B, cls.R, cls.Q = default_covariances(cls.forward_model.hidden_dim)

        # Run the assimilation once and reuse for every test below.
        cls.trajectory, cls.diagnostics = tensor_var_4dvar(
            obs_history=cls.obs_history,
            forward_model=cls.forward_model,
            inverse_model=cls.inverse_model,
            z_b=cls.z_b,
            B=cls.B,
            R=cls.R,
            Q=cls.Q,
            assimilation_window=5,
            total_steps=5,
        )

    def test_C_forward_matches_checkpoint(self):
        np.testing.assert_allclose(
            self.forward_model.C_forward.detach().cpu().numpy(),
            self.expected_C_forward,
            rtol=0,
            atol=0,
        )

    def test_z_b_matches_checkpoint(self):
        np.testing.assert_allclose(self.z_b.numpy(), self.expected_z_b, rtol=0, atol=0)

    def test_inverse_obs_features_match(self):
        np.testing.assert_allclose(
            self.diagnostics["inv_obs_seq_z"], self.expected_inv, rtol=RTOL, atol=ATOL
        )

    def test_K_S_features_match(self):
        np.testing.assert_allclose(
            self.diagnostics["K_S_seq_z"], self.expected_K_S, rtol=RTOL, atol=ATOL
        )

    def test_qp_result_matches(self):
        # CVXPY/SCS may have small per-platform numerical drift; relax slightly.
        np.testing.assert_allclose(
            self.diagnostics["qp_result"], self.expected_qp, rtol=1e-4, atol=1e-5
        )

    def test_decoded_trajectory_matches(self):
        np.testing.assert_allclose(
            self.trajectory, self.expected_decoded, rtol=2e-4, atol=2e-5
        )


@unittest.skipUnless(_have_parity_fixtures(), "parity fixtures missing")
class TestQPSolverIsolated(unittest.TestCase):
    """`qp_solver_latent` must reproduce the captured QP output for the recorded inputs."""

    def test_qp_solver_matches_capture(self):
        seq_z = np.load(PARITY_DIR / "K_S_seq_z.npy")
        z_b = np.load(PARITY_DIR / "z_b.npy")
        F = np.load(PARITY_DIR / "C_forward.npy")
        expected = np.load(PARITY_DIR / "qp_result.npy")

        B, R, Q = default_covariances(z_b.shape[0])
        result = qp_solver_latent(z_b=z_b, seq_z=seq_z, F=F, B=B, R=R, Q=Q, T=seq_z.shape[0])
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)


@unittest.skipUnless(_have_parity_fixtures(), "parity fixtures missing")
class TestFinalMetricsParity(unittest.TestCase):
    """Per-channel weighted NRMSE must match the captured value."""

    def test_weighted_nrmse_matches(self):
        from src.preprocessing import load_ground_truth
        from src.visualization import compute_weighted_nrmse_per_channel

        torch.manual_seed(0)
        np.random.seed(0)
        download_pretrained_weights(WEIGHTS_DIR)
        forward_model, inverse_model, z_b = load_pretrained_models(WEIGHTS_DIR, DEVICE)
        forward_model.eval()
        inverse_model.eval()

        raw = select_sample(load_observation(TASK_DIR / "data"), 0)
        gt = select_sample(load_ground_truth(TASK_DIR / "data"), 0)
        obs = torch.tensor(raw["obs_history"], dtype=torch.float32, device=DEVICE)
        weight = raw["lat_weight_matrix"]
        state_true = gt["state"]

        B, R, Q = default_covariances(forward_model.hidden_dim)
        traj, _ = tensor_var_4dvar(obs, forward_model, inverse_model, z_b, B, R, Q, 5, 5)
        per_ch = compute_weighted_nrmse_per_channel(traj, state_true, weight)

        with (PARITY_DIR / "final_metrics.json").open("r") as fh:
            expected = json.load(fh)["weighted_nrmse_per_channel"]
        np.testing.assert_allclose(per_ch, expected, rtol=2e-4, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
