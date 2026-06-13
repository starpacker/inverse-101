"""Tests for solvers module."""

import os
import sys
import numpy as np
import torch
import pytest

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, TASK_DIR)
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")


class TestStochasticInterpolant:
    """Test interpolant coefficient computations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "solvers_interpolant.npz"))
        self.t_vals = fix["input_t"]
        self.expected_alpha = fix["output_alpha"]
        self.expected_beta = fix["output_beta"]
        self.expected_sigma = fix["output_sigma"]

    def test_alpha(self):
        from src.solvers import StochasticInterpolant
        interp = StochasticInterpolant(beta_fn="t^2", sigma_coef=1.0)
        for i, t in enumerate(self.t_vals):
            t_tensor = torch.tensor([t])
            result = interp.alpha(t_tensor).item()
            np.testing.assert_allclose(result, self.expected_alpha[i], rtol=1e-5)

    def test_beta_t_squared(self):
        from src.solvers import StochasticInterpolant
        interp = StochasticInterpolant(beta_fn="t^2", sigma_coef=1.0)
        for i, t in enumerate(self.t_vals):
            t_tensor = torch.tensor([t])
            result = interp.beta(t_tensor).item()
            np.testing.assert_allclose(result, self.expected_beta[i], rtol=1e-5)

    def test_sigma(self):
        from src.solvers import StochasticInterpolant
        interp = StochasticInterpolant(beta_fn="t^2", sigma_coef=1.0)
        for i, t in enumerate(self.t_vals):
            t_tensor = torch.tensor([t])
            result = interp.sigma(t_tensor).item()
            np.testing.assert_allclose(result, self.expected_sigma[i], rtol=1e-5)

    def test_boundary_conditions(self):
        """At t=0: alpha=1, beta=0, sigma=1. At t=1: alpha=0, beta=1, sigma=0."""
        from src.solvers import StochasticInterpolant
        interp = StochasticInterpolant(beta_fn="t^2", sigma_coef=1.0)
        t0 = torch.tensor([0.0])
        t1 = torch.tensor([1.0])
        assert abs(interp.alpha(t0).item() - 1.0) < 1e-6
        assert abs(interp.beta(t0).item() - 0.0) < 1e-6
        assert abs(interp.sigma(t0).item() - 1.0) < 1e-6
        assert abs(interp.alpha(t1).item() - 0.0) < 1e-6
        assert abs(interp.beta(t1).item() - 1.0) < 1e-6
        assert abs(interp.sigma(t1).item() - 0.0) < 1e-6


class TestDriftModel:
    """Test drift model construction and forward pass shapes."""

    @pytest.fixture(autouse=True)
    def setup(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "solvers_drift_model.npz"))
        self.input_shape = tuple(fix["input_shape"])
        self.output_shape = tuple(fix["output_shape"])

    def test_model_construction(self):
        from src.solvers import DriftModel
        model = DriftModel(in_channels=7, out_channels=1, unet_channels=128)
        assert model is not None

    def test_model_param_count(self):
        from src.solvers import DriftModel
        model = DriftModel(in_channels=7, out_channels=1, unet_channels=128)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == 30_020_113, f"Expected 30,020,113 params, got {n_params:,}"

    def test_forward_pass_shape(self):
        from src.solvers import DriftModel
        # in_channels=7: DriftModel.forward concatenates zt (1ch) + cond (6ch) internally
        model = DriftModel(in_channels=7, out_channels=1, unet_channels=128)
        model.eval()
        zt = torch.randn(1, 1, 128, 128)
        cond = torch.randn(1, 6, 128, 128)
        t = torch.tensor([0.5])
        with torch.no_grad():
            out = model(zt, t, None, cond=cond)
        assert out.shape == self.output_shape


class TestLoadDriftModel:
    """Test loading pretrained checkpoint."""

    def test_load_checkpoint(self):
        ckpt_path = os.path.join(TASK_DIR, "model", "latest.pt")
        if not os.path.exists(ckpt_path):
            pytest.skip("Checkpoint not available")
        from src.solvers import load_drift_model
        model = load_drift_model(ckpt_path, device="cpu")
        assert model is not None
        assert not model.training  # should be in eval mode
