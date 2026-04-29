"""Unit tests for src/physics_model.py."""
import os, sys, pytest, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.physics_model import centered_fft2, centered_ifft2, sos_combine, zero_filled_recon

class TestFFT:
    def test_roundtrip(self):
        x = np.random.randn(32, 32, 4) + 1j * np.random.randn(32, 32, 4)
        np.testing.assert_allclose(centered_ifft2(centered_fft2(x)), x, atol=1e-10)

    def test_parseval(self):
        x = np.random.randn(32, 32) + 1j * np.random.randn(32, 32)
        k = centered_fft2(x)
        np.testing.assert_allclose(np.sum(np.abs(k)**2), np.sum(np.abs(x)**2), rtol=1e-10)

class TestSosCombine:
    def test_shape(self):
        x = np.random.randn(32, 32, 4) + 1j * np.random.randn(32, 32, 4)
        assert sos_combine(x).shape == (32, 32)

    def test_non_negative(self):
        x = np.random.randn(32, 32, 4) + 1j * np.random.randn(32, 32, 4)
        assert np.all(sos_combine(x) >= 0)

class TestZeroFilledRecon:
    def test_shape(self):
        k = np.random.randn(32, 32, 4) + 1j * np.random.randn(32, 32, 4)
        assert zero_filled_recon(k).shape == (32, 32)
