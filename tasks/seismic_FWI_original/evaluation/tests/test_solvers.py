"""Unit tests for seismic_FWI_original src/solvers.py."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.solvers import cosine_taper, smooth_gradient

FIXTURES = Path(__file__).parents[1] / "fixtures"


# ---------------------------------------------------------------------------
# Cosine taper
# ---------------------------------------------------------------------------

class TestCosineTaper:
    def test_shape_preserved(self):
        x = torch.randn(3, 5, 100)
        assert cosine_taper(x, n_taper=10).shape == (3, 5, 100)

    def test_untapered_region_unchanged(self):
        x = torch.ones(2, 50)
        y = cosine_taper(x, n_taper=5)
        np.testing.assert_array_equal(y[:, :-5].numpy(), x[:, :-5].numpy())

    def test_taper_values(self):
        """Check the taper window values at the boundary."""
        x = torch.ones(1, 10)
        y = cosine_taper(x, n_taper=5)
        # i=5: cos(π*5/5)=-1 → taper=0; i=1: cos(π/5)≈0.809 → taper≈0.9045
        assert abs(y[0, -1].item()) < 1e-5   # last sample should be ~0
        assert y[0, -5].item() < 1.0          # first tapered sample < 1

    def test_zero_taper(self):
        x = torch.randn(2, 30)
        y = cosine_taper(x, n_taper=0)
        np.testing.assert_array_equal(y.numpy(), x.numpy())

    def test_numerics(self):
        x = torch.ones(2, 3, 20)
        y = cosine_taper(x, n_taper=5)
        ref = np.load(FIXTURES / "output_cosine_taper.npy")
        np.testing.assert_allclose(y.numpy(), ref, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# Smooth gradient
# ---------------------------------------------------------------------------

class TestSmoothGradient:
    def test_output_shape(self):
        g = np.ones((30, 40), dtype=np.float32)
        out = smooth_gradient(g, sigma=1.0)
        assert out.shape == (30, 40)

    def test_uniform_field_unchanged(self):
        """Gaussian filter of a constant field returns the same constant."""
        g = np.full((20, 20), 3.14, dtype=np.float32)
        out = smooth_gradient(g, sigma=2.0)
        np.testing.assert_allclose(out, g, rtol=1e-4, atol=1e-4)

    def test_smoothing_reduces_peak(self):
        """A delta function should be spread by the Gaussian filter."""
        g = np.zeros((30, 30), dtype=np.float32)
        g[15, 15] = 1.0
        out = smooth_gradient(g, sigma=2.0)
        assert out[15, 15] < 1.0
        assert out.sum() > 0.5   # energy roughly conserved

    def test_numerics(self):
        g = np.load(FIXTURES / "input_gradient.npy")
        ref = np.load(FIXTURES / "output_smooth_gradient.npy")
        out = smooth_gradient(g, sigma=1.0)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-7)
