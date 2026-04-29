"""Unit tests for src/physics_model.py."""
import os, sys, pytest, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.physics_model import *

class TestFFT:
    def test_roundtrip(self):
        x = np.random.randn(32, 32, 4) + 1j * np.random.randn(32, 32, 4)
        np.testing.assert_allclose(centered_ifft2(centered_fft2(x)), x, atol=1e-10)

class TestSenseOperators:
    def test_forward_shape(self):
        x = np.random.randn(32, 32) + 1j * np.random.randn(32, 32)
        sens = np.random.randn(32, 32, 4) + 1j * np.random.randn(32, 32, 4)
        mask = np.ones(32, dtype=bool)
        mask[::2] = False
        y = sense_forward(x, sens, mask)
        assert y.shape == (32, 32, 4)

    def test_adjoint_shape(self):
        y = np.random.randn(32, 32, 4) + 1j * np.random.randn(32, 32, 4)
        sens = np.random.randn(32, 32, 4) + 1j * np.random.randn(32, 32, 4)
        x = sense_adjoint(y, sens)
        assert x.shape == (32, 32)

    def test_forward_adjoint_roundtrip(self):
        """A^H A should produce a valid image (smoke test)."""
        np.random.seed(42)
        N, nc = 16, 4
        x = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        sens = np.random.randn(N, N, nc) + 1j * np.random.randn(N, N, nc)
        mask = np.ones(N, dtype=bool)

        Ax = sense_forward(x, sens, mask)
        AHAx = sense_adjoint(Ax, sens)
        assert AHAx.shape == x.shape
        # With full sampling and identity-ish sensitivities, should correlate with x
        ncc = np.abs(np.vdot(AHAx.flatten(), x.flatten())) / (
            np.linalg.norm(AHAx) * np.linalg.norm(x))
        assert ncc > 0.5

class TestZeroFill:
    def test_shape(self):
        k = np.random.randn(32, 32, 4) + 1j * np.random.randn(32, 32, 4)
        assert zero_filled_recon(k).shape == (32, 32)
