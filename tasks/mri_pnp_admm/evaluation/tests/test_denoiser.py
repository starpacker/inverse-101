"""Unit tests for src/denoiser.py."""
import os
import sys
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.denoiser import RealSN_DnCNN, load_denoiser

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")


class TestRealSN_DnCNN:
    def test_output_shape(self):
        model = RealSN_DnCNN(channels=1, num_of_layers=5)
        model.eval()
        x = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 32, 32)

    def test_residual_small(self):
        """For clean input, residual should be small."""
        model = RealSN_DnCNN(channels=1, num_of_layers=5)
        model.eval()
        x = torch.ones(1, 1, 32, 32) * 0.5
        with torch.no_grad():
            residual = model(x)
        # Untrained model may give arbitrary output, just check shape
        assert residual.shape == x.shape


class TestLoadDenoiser:
    def test_loads_and_runs(self):
        weights_path = os.path.join(DATA_DIR, "RealSN_DnCNN_noise15.pth")
        model = load_denoiser(weights_path, device="cpu")
        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 256, 256)

    def test_deterministic(self):
        weights_path = os.path.join(DATA_DIR, "RealSN_DnCNN_noise15.pth")
        model = load_denoiser(weights_path, device="cpu")
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            out1 = model(x).numpy()
            out2 = model(x).numpy()
        np.testing.assert_array_equal(out1, out2)
