"""Tests for the lensless imaging ADMM solver and helper functions."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.solvers import soft_thresh, finite_diff, finite_diff_adj, finite_diff_gram, ADMM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_psf(H=16, W=16, C=3, seed=0):
    rng = np.random.default_rng(seed)
    psf = rng.uniform(0.0, 1.0, (H, W, C)).astype(np.float64)
    psf /= psf.sum(axis=(0, 1), keepdims=True)
    return psf


def _make_scene(H=16, W=16, C=3, seed=1):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, (H, W, C)).astype(np.float64)


# ---------------------------------------------------------------------------
# Tests for soft_thresh
# ---------------------------------------------------------------------------

class TestSoftThresh:
    def test_zero_below_threshold(self):
        """Values with |x| < thresh should be zeroed."""
        x = np.array([-0.3, -0.1, 0.0, 0.1, 0.3])
        result = soft_thresh(x, 0.2)
        np.testing.assert_allclose(result[1], 0.0, atol=1e-15)
        np.testing.assert_allclose(result[2], 0.0, atol=1e-15)
        np.testing.assert_allclose(result[3], 0.0, atol=1e-15)

    def test_shrinks_above_threshold(self):
        """Values with |x| > thresh should shrink by thresh."""
        x = np.array([1.0, -1.0, 0.5, -0.5])
        result = soft_thresh(x, 0.3)
        expected = np.array([0.7, -0.7, 0.2, -0.2])
        np.testing.assert_allclose(result, expected, atol=1e-15)

    def test_preserves_sign(self):
        x = np.array([-2.0, 2.0])
        result = soft_thresh(x, 0.5)
        assert result[0] < 0
        assert result[1] > 0

    def test_output_shape_matches_input(self):
        x = np.random.randn(8, 8, 3)
        result = soft_thresh(x, 0.1)
        assert result.shape == x.shape


# ---------------------------------------------------------------------------
# Tests for finite_diff and finite_diff_adj
# ---------------------------------------------------------------------------

class TestFiniteDiff:
    def test_output_shape(self):
        v = np.random.randn(16, 16, 3)
        u = finite_diff(v)
        assert u.shape == (2, 16, 16, 3)

    def test_constant_input_gives_zero(self):
        """Gradient of a constant field should be zero."""
        v = np.ones((8, 8, 1), dtype=np.float64) * 3.14
        u = finite_diff(v)
        np.testing.assert_allclose(u, 0.0, atol=1e-15)

    def test_adjoint_dot_product(self):
        """<Psi v, u> == <v, Psi^T u> (adjoint dot-product test)."""
        rng = np.random.default_rng(42)
        v = rng.standard_normal((16, 16, 3))
        u_rand = rng.standard_normal((2, 16, 16, 3))
        Psi_v = finite_diff(v)
        PsiT_u = finite_diff_adj(u_rand)
        lhs = np.sum(Psi_v * u_rand)
        rhs = np.sum(v * PsiT_u)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)

    def test_finite_diff_adj_output_shape(self):
        u = np.random.randn(2, 8, 8, 3)
        v = finite_diff_adj(u)
        assert v.shape == (8, 8, 3)


# ---------------------------------------------------------------------------
# Tests for finite_diff_gram
# ---------------------------------------------------------------------------

class TestFiniteDiffGram:
    def test_output_shape(self):
        gram = finite_diff_gram((16, 16))
        # rfft2 of (16,16) -> (16, 9), plus trailing dim 1
        assert gram.shape == (16, 9, 1)

    def test_dc_component(self):
        """DC component (k=0) of the Laplacian should be 0.

        The Laplacian kernel sums to 4 - 1 - 1 - 1 - 1 = 0, so its DFT
        at (0,0) should be 0.
        """
        gram = finite_diff_gram((32, 32))
        np.testing.assert_allclose(gram[0, 0, 0], 0.0, atol=1e-10)

    def test_nonneg_eigenvalues(self):
        """The Gram matrix Psi^T Psi should have non-negative eigenvalues."""
        gram = finite_diff_gram((16, 16))
        assert np.all(np.real(gram) >= -1e-10)


# ---------------------------------------------------------------------------
# Tests for ADMM
# ---------------------------------------------------------------------------

class TestADMM:
    def test_get_image_shape_and_dtype(self):
        """After set_data + zero iterations, get_image returns (H,W,C) float32."""
        H, W, C = 16, 16, 3
        psf = _make_psf(H, W, C).astype(np.float32)
        meas = _make_scene(H, W, C, seed=5).astype(np.float32)
        solver = ADMM(psf)
        solver.set_data(meas)
        img = solver.get_image()
        assert img.shape == (H, W, C)
        assert img.dtype == np.float32

    def test_reconstruction_nonnegative(self):
        """ADMM output should be non-negative (clipped by get_image)."""
        H, W, C = 16, 16, 3
        psf = _make_psf(H, W, C).astype(np.float32)
        meas = _make_scene(H, W, C, seed=5).astype(np.float32)
        solver = ADMM(psf, mu1=1e-4, mu2=1e-3, mu3=1e-3, tau=1e-3)
        solver.set_data(meas)
        rec = solver.apply(n_iter=5, verbose=False)
        assert np.all(rec >= 0.0)

    def test_apply_reduces_residual(self):
        """Running more iterations should reduce the data-fidelity residual.

        We generate a synthetic measurement b = forward(v) and check that
        ||b - forward(v_hat)|| decreases from 0 to 5 to 20 iterations.
        """
        H, W, C = 16, 16, 3
        psf = _make_psf(H, W, C).astype(np.float32)
        scene = _make_scene(H, W, C, seed=3).astype(np.float32)

        from src.physics_model import RealFFTConvolve2D
        conv = RealFFTConvolve2D(psf)
        meas = conv.forward(scene).astype(np.float32)

        solver = ADMM(psf, mu1=1e-4, mu2=1e-3, mu3=1e-3, tau=1e-3)
        solver.set_data(meas)

        # 0 iterations -> all zeros
        rec0 = solver.get_image()
        res0 = np.sqrt(np.mean((conv.forward(rec0) - meas) ** 2))

        # 5 iterations
        rec5 = solver.apply(n_iter=5, verbose=False)
        res5 = np.sqrt(np.mean((conv.forward(rec5) - meas) ** 2))

        # 20 more iterations (total 25)
        rec25 = solver.apply(n_iter=20, verbose=False)
        res25 = np.sqrt(np.mean((conv.forward(rec25) - meas) ** 2))

        assert res5 < res0, f"5-iter residual {res5} >= 0-iter residual {res0}"
        assert res25 < res5, f"25-iter residual {res25} >= 5-iter residual {res5}"

    def test_reset_clears_state(self):
        """After reset(), the image estimate should be back to zero."""
        H, W, C = 8, 8, 3
        psf = _make_psf(H, W, C).astype(np.float32)
        meas = _make_scene(H, W, C, seed=5).astype(np.float32)
        solver = ADMM(psf)
        solver.set_data(meas)
        _ = solver.apply(n_iter=3, verbose=False)

        solver.reset()
        img = solver.get_image()
        np.testing.assert_allclose(img, 0.0, atol=1e-7)

    def test_set_data_shape_mismatch_raises(self):
        """Passing a measurement with wrong shape should raise AssertionError."""
        psf = _make_psf(16, 16, 3).astype(np.float32)
        solver = ADMM(psf)
        wrong_shape = np.zeros((8, 8, 3), dtype=np.float32)
        with pytest.raises(AssertionError):
            solver.set_data(wrong_shape)
