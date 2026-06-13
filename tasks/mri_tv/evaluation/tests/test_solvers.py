"""Unit tests for solvers.py."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.solvers import (
    tv_reconstruct_single, tv_reconstruct_batch,
    sense_forward, sense_adjoint,
    finite_difference, finite_difference_adjoint,
    soft_thresh, prox_l2_reg, prox_l1_conj,
    estimate_max_eigenvalue,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "../fixtures/solvers")


class TestSenseOperator:
    """Test SENSE forward/adjoint pair."""

    def setup_method(self):
        rng = np.random.default_rng(42)
        self.H, self.W, self.C = 16, 16, 4
        self.smaps = rng.standard_normal((self.C, self.H, self.W)) + \
                     1j * rng.standard_normal((self.C, self.H, self.W))
        self.mask_w = np.ones((1, self.H, self.W))

    def test_forward_shape(self):
        x = np.ones((self.H, self.W), dtype=np.complex128)
        y = sense_forward(x, self.smaps, self.mask_w)
        assert y.shape == (self.C, self.H, self.W)

    def test_adjoint_shape(self):
        y = np.ones((self.C, self.H, self.W), dtype=np.complex128)
        x = sense_adjoint(y, self.smaps, self.mask_w)
        assert x.shape == (self.H, self.W)

    def test_adjoint_consistency(self):
        """<Ax, y> ~ <x, A^H y> (dot product test)."""
        rng = np.random.default_rng(99)
        x = rng.standard_normal((self.H, self.W)) + 1j * rng.standard_normal((self.H, self.W))
        y = rng.standard_normal((self.C, self.H, self.W)) + 1j * rng.standard_normal((self.C, self.H, self.W))
        Ax = sense_forward(x, self.smaps, self.mask_w)
        AHy = sense_adjoint(y, self.smaps, self.mask_w)
        lhs = np.vdot(Ax, y)
        rhs = np.vdot(x, AHy)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


class TestFiniteDifference:
    def test_output_shape(self):
        x = np.ones((8, 8), dtype=np.complex128)
        g = finite_difference(x)
        assert g.shape == (2, 8, 8)

    def test_constant_image_zero_gradient(self):
        x = np.ones((8, 8), dtype=np.complex128) * 5.0
        g = finite_difference(x)
        np.testing.assert_allclose(g, 0, atol=1e-12)

    def test_adjoint_consistency(self):
        """<Gx, p> ~ <x, G^H p>"""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
        p = rng.standard_normal((2, 8, 8)) + 1j * rng.standard_normal((2, 8, 8))
        Gx = finite_difference(x)
        GHp = finite_difference_adjoint(p)
        lhs = np.vdot(Gx, p)
        rhs = np.vdot(x, GHp)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


class TestProximalOperators:
    def test_soft_thresh_zero_below(self):
        x = np.array([0.5, -0.3, 0.1])
        result = soft_thresh(0.5, x)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-12)

    def test_soft_thresh_shrinks(self):
        x = np.array([2.0, -1.5])
        result = soft_thresh(0.5, x)
        np.testing.assert_allclose(result, [1.5, -1.0], atol=1e-12)

    def test_soft_thresh_complex(self):
        x = np.array([3.0 + 4.0j])  # |x| = 5
        result = soft_thresh(1.0, x)
        expected = (4.0 / 5.0) * x  # shrink magnitude by 1
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_prox_l2_reg(self):
        u = np.array([2.0, 4.0])
        y = np.array([1.0, 1.0])
        sigma = 1.0
        result = prox_l2_reg(sigma, u, y)
        # (u + sigma * y) / (1 + sigma) = (2+1, 4+1) / 2 = (1.5, 2.5)
        np.testing.assert_allclose(result, [1.5, 2.5], atol=1e-12)


class TestPowerIteration:
    def test_positive_eigenvalue(self):
        rng = np.random.default_rng(42)
        C, H, W = 2, 8, 8
        smaps = rng.standard_normal((C, H, W)) + 1j * rng.standard_normal((C, H, W))
        mask_w = np.ones((1, H, W))
        eig = estimate_max_eigenvalue(smaps, mask_w, (H, W), max_iter=20)
        assert eig > 0


class TestTVReconstructSingle:
    def setup_method(self):
        data = np.load(os.path.join(DATA_DIR, "raw_data.npz"))
        self.masked_kspace_0 = data["masked_kspace"][0]
        self.smaps_0 = data["sensitivity_maps"][0]
        fix = np.load(os.path.join(FIXTURE_DIR, "output_tv_recon_sample0.npz"))
        self.expected = fix["reconstruction"]

    def test_output_shape(self):
        recon = tv_reconstruct_single(self.masked_kspace_0, self.smaps_0, lamda=1e-4)
        assert recon.shape == (320, 320)

    def test_output_complex(self):
        recon = tv_reconstruct_single(self.masked_kspace_0, self.smaps_0, lamda=1e-4)
        assert np.iscomplexobj(recon)

    def test_parity_with_reference(self):
        """Iterative solver has platform-dependent numerics; check NCC > 0.999."""
        recon = tv_reconstruct_single(self.masked_kspace_0, self.smaps_0, lamda=1e-4)
        recon_mag = np.abs(recon).flatten()
        ref_mag = np.abs(self.expected).flatten()
        ncc = np.dot(recon_mag, ref_mag) / (np.linalg.norm(recon_mag) * np.linalg.norm(ref_mag))
        assert ncc > 0.999, f"NCC too low: {ncc}"

    def test_ncc_with_reference(self):
        recon = tv_reconstruct_single(self.masked_kspace_0, self.smaps_0, lamda=1e-4)
        recon_mag = np.abs(recon).flatten()
        ref_mag = np.abs(self.expected).flatten()
        ncc = np.dot(recon_mag, ref_mag) / (np.linalg.norm(recon_mag) * np.linalg.norm(ref_mag))
        assert ncc > 0.999


class TestTVReconstructBatch:
    def setup_method(self):
        data = np.load(os.path.join(DATA_DIR, "raw_data.npz"))
        self.masked_kspace = data["masked_kspace"][:1]
        self.smaps = data["sensitivity_maps"][:1]

    def test_output_shape(self):
        recons = tv_reconstruct_batch(self.masked_kspace, self.smaps, lamda=1e-4)
        assert recons.shape == (1, 320, 320)

    def test_output_complex(self):
        recons = tv_reconstruct_batch(self.masked_kspace, self.smaps, lamda=1e-4)
        assert np.iscomplexobj(recons)

    def test_single_batch_consistency(self):
        """Single-sample batch should match single call (NCC > 0.999)."""
        single = tv_reconstruct_single(self.masked_kspace[0], self.smaps[0], lamda=1e-4)
        batch = tv_reconstruct_batch(self.masked_kspace[:1], self.smaps[:1], lamda=1e-4)
        s = np.abs(single).flatten()
        b = np.abs(batch[0]).flatten()
        ncc = np.dot(s, b) / (np.linalg.norm(s) * np.linalg.norm(b))
        assert ncc > 0.999, f"NCC too low: {ncc}"
