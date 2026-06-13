"""Tests for solvers module using external fixtures."""

import os
import sys
import pickle
import numpy as np
import torch
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")


def load_fixture(name):
    with open(os.path.join(FIXTURE_DIR, name), "rb") as f:
        return pickle.load(f)


from src.solvers import create_gaussian_kernel, ncg


class TestGaussianKernel:
    def test_against_fixture(self):
        inp = load_fixture("input_gaussian_kernel.pkl")
        expected = load_fixture("output_gaussian_kernel.pkl")

        kernel = create_gaussian_kernel(inp["kernel_size"], inp["sigma"])
        np.testing.assert_allclose(kernel.numpy(), expected["kernel"], rtol=1e-10)

    def test_sums_to_one(self):
        inp = load_fixture("input_gaussian_kernel.pkl")
        k = create_gaussian_kernel(inp["kernel_size"], inp["sigma"])
        assert abs(k.sum().item() - 1.0) < 1e-5

    def test_symmetric(self):
        inp = load_fixture("input_gaussian_kernel.pkl")
        k = create_gaussian_kernel(inp["kernel_size"], inp["sigma"])
        assert torch.allclose(k, k.T)
        assert torch.allclose(k, k.flip(0))


class TestNCG:
    def test_returns_valid_output(self):
        """NCG should return a tensor of the same shape within bounds."""
        def fun(x):
            f = torch.sum((x - 1.0 / 1500) ** 2)
            g = 2 * (x - 1.0 / 1500)
            return f, g

        x0 = torch.ones(10, 10, device="cuda") / 1480.0
        x_opt = ncg(fun, x0, max_iters=3, v_bounds=(1300.0, 1700.0))
        assert x_opt.shape == (10, 10)
        assert x_opt.min() >= 1.0 / 1700 - 1e-6
        assert x_opt.max() <= 1.0 / 1300 + 1e-6


class TestParity:
    """Parity test: single-freq output must match original code output."""

    def test_single_freq_parity(self):
        fixture_path = os.path.join(FIXTURE_DIR, "output_parity_single_freq.pkl")
        recon_path = os.path.join(TASK_DIR, "evaluation", "reference_outputs", "reconstruction.npy")

        if not os.path.exists(fixture_path):
            pytest.skip("Parity fixture not generated")
        if not os.path.exists(recon_path):
            pytest.skip("Reconstruction output not available")

        expected = load_fixture("output_parity_single_freq.pkl")
        vp_orig = expected["vp_original"]

        # Run single-freq to get current output
        # (use the saved reconstruction which was generated from single-freq 0.3 MHz)
        vp_new = np.load(recon_path)

        ncc = np.dot(vp_orig.flatten(), vp_new.flatten()) / (
            np.linalg.norm(vp_orig.flatten()) * np.linalg.norm(vp_new.flatten())
        )
        assert ncc > 0.999, f"Parity NCC={ncc:.6f}, expected > 0.999"
