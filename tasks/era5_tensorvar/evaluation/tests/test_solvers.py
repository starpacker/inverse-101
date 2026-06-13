"""Unit tests for src/solvers.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

TASK_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK_DIR))

from src.preprocessing import default_covariances
from src.solvers import qp_solver_latent, tensor_var_4dvar
from src.physics_model import ERA5ForwardModel, ERA5InverseModel


class TestQPSolverIdentity(unittest.TestCase):
    """When B=R=I, Q=0 and the dynamics term vanishes, the QP minimum is the
    midpoint between background and observation features."""

    def test_single_step_midpoint(self):
        D = 8
        z_b = np.ones(D, dtype=np.float64)
        seq_z = np.full((1, D), 3.0, dtype=np.float64)
        F = np.eye(D)
        B = np.eye(D)
        R = np.eye(D)
        Q = np.zeros((D, D))

        result = qp_solver_latent(z_b=z_b, seq_z=seq_z, F=F, B=B, R=R, Q=Q, T=1)
        np.testing.assert_allclose(result[0], np.full(D, 2.0), atol=1e-4)

    def test_zero_R_returns_background(self):
        D = 4
        z_b = np.zeros(D, dtype=np.float64)
        seq_z = np.full((1, D), 5.0, dtype=np.float64)
        F = np.eye(D)
        B = np.eye(D)
        R = np.zeros((D, D))
        Q = np.zeros((D, D))

        result = qp_solver_latent(z_b=z_b, seq_z=seq_z, F=F, B=B, R=R, Q=Q, T=1)
        np.testing.assert_allclose(result[0], z_b, atol=1e-4)


class TestQPSolverShape(unittest.TestCase):
    def test_output_shape(self):
        D = 16
        T = 5
        z_b = np.random.RandomState(0).randn(D)
        seq_z = np.random.RandomState(1).randn(T, D)
        F = np.eye(D)
        B, R, Q = default_covariances(D)
        result = qp_solver_latent(z_b, seq_z, F, B, R, Q, T)
        self.assertEqual(result.shape, (T, D))


class TestTensorVarShapes(unittest.TestCase):
    """Smoke test on randomly initialised modules — checks plumbing only."""

    def test_runs_end_to_end_with_random_weights(self):
        torch.manual_seed(0)
        forward_model = ERA5ForwardModel()
        forward_model.C_forward = torch.eye(forward_model.hidden_dim)
        inverse_model = ERA5InverseModel()
        forward_model.eval()
        inverse_model.eval()

        obs = torch.randn(5, 50, 64, 32)
        z_b = torch.zeros(forward_model.hidden_dim)
        B, R, Q = default_covariances(forward_model.hidden_dim)

        traj, diag = tensor_var_4dvar(
            obs_history=obs,
            forward_model=forward_model,
            inverse_model=inverse_model,
            z_b=z_b,
            B=B,
            R=R,
            Q=Q,
            assimilation_window=5,
            total_steps=5,
        )
        self.assertEqual(traj.shape, (5, 5, 64, 32))
        self.assertEqual(diag["inv_obs_seq_z"].shape, (5, 5, 64, 32))
        self.assertEqual(diag["K_S_seq_z"].shape, (5, 512))
        self.assertEqual(diag["qp_result"].shape, (5, 512))


if __name__ == "__main__":
    unittest.main()
