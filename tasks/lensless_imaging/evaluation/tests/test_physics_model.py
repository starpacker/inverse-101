"""Tests for the lensless imaging physics model (RealFFTConvolve2D)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.physics_model import RealFFTConvolve2D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_psf(H=16, W=16, C=3, seed=0):
    """Small normalised PSF for testing."""
    rng = np.random.default_rng(seed)
    psf = rng.uniform(0.0, 1.0, (H, W, C)).astype(np.float64)
    psf /= psf.sum(axis=(0, 1), keepdims=True)
    return psf


def _make_scene(H=16, W=16, C=3, seed=1):
    """Small random scene in [0, 1]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, (H, W, C)).astype(np.float64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRealFFTConvolve2DInit:
    """Constructor and precomputed attributes."""

    def test_padded_shape_at_least_2N_minus_1(self):
        """Padded shape must be >= 2*H-1, 2*W-1 for linear convolution."""
        H, W, C = 16, 16, 3
        psf = _make_psf(H, W, C)
        model = RealFFTConvolve2D(psf)
        ph, pw = model.padded_shape
        assert ph >= 2 * H - 1
        assert pw >= 2 * W - 1

    def test_psf_shape_accessor(self):
        psf = _make_psf(8, 12, 3)
        model = RealFFTConvolve2D(psf)
        assert model.psf_shape == (8, 12)

    def test_H_and_Hadj_conjugate(self):
        """Hadj should be the complex conjugate of H."""
        psf = _make_psf(16, 16, 3)
        model = RealFFTConvolve2D(psf)
        np.testing.assert_allclose(model.Hadj, np.conj(model.H), atol=1e-15)

    def test_H_dtype_is_complex(self):
        psf = _make_psf(16, 16, 3)
        model = RealFFTConvolve2D(psf)
        assert np.iscomplexobj(model.H)


class TestPadCrop:
    """_pad and _crop are inverse operations on the signal window."""

    def test_pad_output_shape(self):
        psf = _make_psf(16, 16, 3)
        model = RealFFTConvolve2D(psf)
        v = _make_scene(16, 16, 3)
        padded = model._pad(v)
        ph, pw = model.padded_shape
        assert padded.shape == (ph, pw, 3)

    def test_crop_pad_roundtrip(self):
        """crop(pad(v)) == v for any (H, W, C) signal."""
        psf = _make_psf(16, 16, 3)
        model = RealFFTConvolve2D(psf)
        v = _make_scene(16, 16, 3)
        np.testing.assert_allclose(model._crop(model._pad(v)), v, atol=1e-15)

    def test_pad_zeros_outside_window(self):
        """Values outside the center window should be zero after padding."""
        psf = _make_psf(8, 8, 1)
        model = RealFFTConvolve2D(psf)
        v = np.ones((8, 8, 1), dtype=np.float64)
        padded = model._pad(v)
        # Zero out the center window and check the rest is all zeros
        s0, s1 = model._start
        e0, e1 = model._end
        padded[s0:e0, s1:e1, :] = 0.0
        assert np.all(padded == 0.0)


class TestConvolveDeconvolve:
    """Forward and adjoint convolution operators."""

    def test_convolve_output_shape(self):
        psf = _make_psf(16, 16, 3)
        model = RealFFTConvolve2D(psf)
        x_padded = model._pad(_make_scene(16, 16, 3))
        out = model.convolve(x_padded)
        assert out.shape == x_padded.shape

    def test_deconvolve_output_shape(self):
        psf = _make_psf(16, 16, 3)
        model = RealFFTConvolve2D(psf)
        y_padded = model._pad(_make_scene(16, 16, 3))
        out = model.deconvolve(y_padded)
        assert out.shape == y_padded.shape

    def test_adjoint_identity(self):
        """<Mx, y> == <x, M^H y>  (adjoint dot-product test).

        For the convolution operator M and its adjoint M^H, the inner
        products must match up to numerical precision.
        """
        psf = _make_psf(16, 16, 3)
        model = RealFFTConvolve2D(psf)
        rng = np.random.default_rng(42)
        x = rng.standard_normal(model.convolve(model._pad(_make_scene())).shape)
        y = rng.standard_normal(x.shape)
        Mx = model.convolve(x)
        Mhy = model.deconvolve(y)
        lhs = np.sum(Mx * y)
        rhs = np.sum(x * Mhy)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-8)


class TestForwardConvenience:
    """forward() = crop(convolve(pad(v)))."""

    def test_forward_output_shape(self):
        H, W, C = 16, 16, 3
        psf = _make_psf(H, W, C)
        model = RealFFTConvolve2D(psf)
        v = _make_scene(H, W, C)
        out = model.forward(v)
        assert out.shape == (H, W, C)

    def test_forward_dtype(self):
        psf = _make_psf(16, 16, 3)
        model = RealFFTConvolve2D(psf)
        out = model.forward(_make_scene())
        assert out.dtype in (np.float64, np.float32)

    def test_forward_energy_conservation(self):
        """Convolution with a unit-sum PSF should roughly preserve total energy.

        sum(forward(v)) should be close to sum(v) * sum(psf_cropped), where
        the PSF is normalised so that its per-channel sums are 1.
        """
        H, W, C = 16, 16, 1
        psf = _make_psf(H, W, C, seed=10)
        # psf is already normalised so sum over spatial dims per channel = 1
        model = RealFFTConvolve2D(psf)
        v = _make_scene(H, W, C, seed=7)
        out = model.forward(v)
        # The forward model includes a crop, so exact energy match is not
        # expected, but the output should have a plausible magnitude (not
        # zero or wildly inflated).
        assert out.sum() > 0, "Forward output should have positive total energy"
        ratio = out.sum() / v.sum()
        assert 0.01 < ratio < 100, f"Energy ratio {ratio} is unreasonable"
