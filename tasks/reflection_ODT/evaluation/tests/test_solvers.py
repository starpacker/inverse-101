"""Tests for the solvers module (FISTA momentum, TV proximal, ReflectionBPMReconstructor)."""

import os
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.solvers import (
    _fista_q,
    _gradient_2d,
    _divergence_2d,
    tv_2d_proximal_single,
    tv_2d_proximal,
    ReflectionBPMReconstructor,
)
from src.physics_model import ReflectionBPMConfig, ReflectionBPMForwardModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_metadata():
    return {
        "volume_shape": [2, 16, 16],
        "n0": 1.5,
        "NA_obj": 0.55,
        "wavelength_um": 0.532,
        "res_um": [0.1, 0.1, 0.5],
        "ri_contrast": 0.02,
        "illumination_rings": [
            {"NA": 0.3, "n_angles": 2, "type": "BF"},
        ],
        "dz_layer": 0.5,
        "dz_gap": 10.0,
    }


def _build_model(metadata=None):
    if metadata is None:
        metadata = _small_metadata()
    config = ReflectionBPMConfig.from_metadata(metadata)
    return ReflectionBPMForwardModel(config, device="cpu"), config


# ---------------------------------------------------------------------------
# FISTA momentum tests
# ---------------------------------------------------------------------------


class TestFistaQ:
    def test_q_zero(self):
        """q(0) should be 1.0."""
        assert _fista_q(0) == 1.0

    def test_q_negative(self):
        """Negative index should also return 1.0."""
        assert _fista_q(-5) == 1.0

    def test_q_one(self):
        """q(1) = (1 + sqrt(1 + 4)) / 2 = (1 + sqrt(5)) / 2."""
        expected = (1.0 + np.sqrt(5.0)) / 2.0
        np.testing.assert_allclose(_fista_q(1), expected, rtol=1e-12)

    def test_q_monotonically_increasing(self):
        """FISTA momentum coefficients should be strictly increasing."""
        vals = [_fista_q(k) for k in range(8)]
        for i in range(1, len(vals)):
            assert vals[i] > vals[i - 1]

    def test_q_recurrence_relation(self):
        """Verify the recurrence q(k) = (1 + sqrt(1 + 4*q(k-1)^2)) / 2."""
        for k in range(1, 6):
            q_prev = _fista_q(k - 1)
            expected = (1.0 + np.sqrt(1.0 + 4.0 * q_prev ** 2)) / 2.0
            np.testing.assert_allclose(_fista_q(k), expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2D gradient / divergence tests
# ---------------------------------------------------------------------------


class TestGradientDivergence:
    def test_gradient_2d_shape(self):
        img = torch.randn(8, 8, dtype=torch.float64)
        dx, dy = _gradient_2d(img)
        assert dx.shape == (8, 8)
        assert dy.shape == (8, 8)

    def test_gradient_constant_image_is_zero(self):
        """Gradient of a constant image should be zero everywhere."""
        img = torch.full((8, 8), 3.14, dtype=torch.float64)
        dx, dy = _gradient_2d(img)
        np.testing.assert_allclose(dx.numpy(), 0.0, atol=1e-15)
        np.testing.assert_allclose(dy.numpy(), 0.0, atol=1e-15)

    def test_gradient_boundary_padding(self):
        """Last column of dx and last row of dy should be zero (zero padding)."""
        rng = np.random.default_rng(42)
        img = torch.tensor(rng.standard_normal((8, 8)), dtype=torch.float64)
        dx, dy = _gradient_2d(img)
        np.testing.assert_allclose(dx[:, -1].numpy(), 0.0, atol=1e-15)
        np.testing.assert_allclose(dy[-1, :].numpy(), 0.0, atol=1e-15)

    def test_divergence_shape(self):
        px = torch.randn(8, 8, dtype=torch.float64)
        py = torch.randn(8, 8, dtype=torch.float64)
        div = _divergence_2d(px, py)
        assert div.shape == (8, 8)

    def test_divergence_adjoint_property(self):
        """
        <grad(u), p> should equal <u, -div(p)> (adjoint relation).
        This is the fundamental property that Chambolle's algorithm relies on.
        """
        rng = np.random.default_rng(99)
        u = torch.tensor(rng.standard_normal((8, 8)), dtype=torch.float64)
        px = torch.tensor(rng.standard_normal((8, 8)), dtype=torch.float64)
        py = torch.tensor(rng.standard_normal((8, 8)), dtype=torch.float64)

        dx, dy = _gradient_2d(u)
        lhs = torch.sum(dx * px + dy * py).item()

        div = _divergence_2d(px, py)
        rhs = torch.sum(u * (-div)).item()

        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


# ---------------------------------------------------------------------------
# TV proximal operator tests
# ---------------------------------------------------------------------------


class TestTVProximal:
    def test_single_output_shape(self):
        img = torch.randn(16, 16, dtype=torch.float64)
        out = tv_2d_proximal_single(img, tau=0.1, n_iter=10)
        assert out.shape == (16, 16)

    def test_single_zero_tau_returns_clone(self):
        """With tau=0, the proximal operator is the identity."""
        rng = np.random.default_rng(7)
        img = torch.tensor(rng.standard_normal((8, 8)), dtype=torch.float64)
        out = tv_2d_proximal_single(img, tau=0.0)
        np.testing.assert_allclose(out.numpy(), img.numpy(), atol=1e-15)

    def test_single_constant_image_unchanged(self):
        """A constant image has zero TV, so the proximal should return it as-is."""
        img = torch.full((16, 16), 2.5, dtype=torch.float64)
        out = tv_2d_proximal_single(img, tau=0.5, n_iter=30)
        np.testing.assert_allclose(out.numpy(), img.numpy(), atol=1e-10)

    def test_single_small_tau_close_to_input(self):
        """
        With very small tau, the TV proximal output should stay close
        to the input (proximal operator approaches identity as tau -> 0).
        """
        rng = np.random.default_rng(123)
        img = torch.tensor(rng.standard_normal((16, 16)), dtype=torch.float64)
        tau = 1e-4
        out = tv_2d_proximal_single(img, tau=tau, n_iter=20)
        diff = torch.max(torch.abs(out - img)).item()
        assert diff < 0.1  # output should be very close to input

    def test_single_stronger_tau_more_shrinkage(self):
        """Larger tau should pull the result further from the input (more denoising)."""
        rng = np.random.default_rng(77)
        img = torch.tensor(rng.standard_normal((16, 16)), dtype=torch.float64)

        out_small = tv_2d_proximal_single(img, tau=0.01, n_iter=50)
        out_large = tv_2d_proximal_single(img, tau=0.05, n_iter=50)

        dist_small = torch.sum((out_small - img) ** 2).item()
        dist_large = torch.sum((out_large - img) ** 2).item()
        assert dist_large > dist_small

    def test_volume_output_shape(self):
        vol = torch.randn(4, 8, 8, dtype=torch.float64)
        out = tv_2d_proximal(vol, tau=0.1, n_iter=5)
        assert out.shape == (4, 8, 8)

    def test_volume_zero_tau_returns_clone(self):
        """Volume version with tau=0 should be identity."""
        rng = np.random.default_rng(11)
        vol = torch.tensor(rng.standard_normal((3, 8, 8)), dtype=torch.float64)
        out = tv_2d_proximal(vol, tau=0.0)
        np.testing.assert_allclose(out.numpy(), vol.numpy(), atol=1e-15)

    def test_volume_slicewise_consistency(self):
        """Volume TV should give the same result as applying single-slice TV to each slice."""
        rng = np.random.default_rng(55)
        vol = torch.tensor(rng.standard_normal((3, 8, 8)), dtype=torch.float64)
        tau = 0.3
        n_iter = 15

        vol_out = tv_2d_proximal(vol, tau, n_iter)

        for iz in range(vol.shape[0]):
            single_out = tv_2d_proximal_single(vol[iz], tau, n_iter)
            np.testing.assert_allclose(
                vol_out[iz].numpy(), single_out.numpy(), atol=1e-14
            )


# ---------------------------------------------------------------------------
# ReflectionBPMReconstructor tests
# ---------------------------------------------------------------------------


class TestReconstructor:
    def test_init_defaults(self):
        rec = ReflectionBPMReconstructor()
        assert rec.n_iter == 50
        assert rec.lr == 5.0
        assert rec.tv_weight == 8e-7
        assert rec.positivity is False

    def test_reconstruct_output_shapes(self):
        """Reconstruction should return (Nz, Ny, Nx) ndarray and loss list."""
        model, config = _build_model()
        nz, ny, nx = config.volume_shape
        n_iter = 3

        # Generate synthetic measurements from a zero sample
        dn_true = torch.zeros(*config.volume_shape, dtype=torch.float64)
        with torch.no_grad():
            measurements = model.forward(dn_true)
        meas_amp = torch.sqrt(measurements)

        rec = ReflectionBPMReconstructor(n_iter=n_iter, lr=1.0,
                                         tv_weight=0.0, device="cpu")
        dn_recon, loss_history = rec.reconstruct(meas_amp, model)

        assert isinstance(dn_recon, np.ndarray)
        assert dn_recon.shape == (nz, ny, nx)
        assert len(loss_history) == n_iter

    def test_reconstruct_loss_dtype(self):
        """Each entry in loss_history should be a Python float."""
        model, config = _build_model()
        dn_true = torch.zeros(*config.volume_shape, dtype=torch.float64)
        with torch.no_grad():
            measurements = model.forward(dn_true)
        meas_amp = torch.sqrt(measurements)

        rec = ReflectionBPMReconstructor(n_iter=2, lr=1.0,
                                         tv_weight=0.0, device="cpu")
        _, loss_history = rec.reconstruct(meas_amp, model)

        for val in loss_history:
            assert isinstance(val, float)

    def test_reconstruct_zero_sample_near_zero(self):
        """
        If measurements come from dn=0, reconstruction should stay near zero
        (since loss is already minimal at dn=0).
        """
        model, config = _build_model()
        dn_true = torch.zeros(*config.volume_shape, dtype=torch.float64)
        with torch.no_grad():
            measurements = model.forward(dn_true)
        meas_amp = torch.sqrt(measurements)

        rec = ReflectionBPMReconstructor(n_iter=5, lr=1.0,
                                         tv_weight=0.0, device="cpu")
        dn_recon, loss_history = rec.reconstruct(meas_amp, model)

        # Reconstruction from a blank sample should remain small
        np.testing.assert_allclose(dn_recon, 0.0, atol=1e-4)

    def test_reconstruct_loss_nonnegative(self):
        """All loss values should be non-negative (MSE-based)."""
        model, config = _build_model()
        rng = np.random.default_rng(42)
        dn_true = torch.tensor(
            rng.uniform(-0.01, 0.01, config.volume_shape),
            dtype=torch.float64,
        )
        with torch.no_grad():
            measurements = model.forward(dn_true)
        meas_amp = torch.sqrt(measurements)

        rec = ReflectionBPMReconstructor(n_iter=3, lr=1.0,
                                         tv_weight=0.0, device="cpu")
        _, loss_history = rec.reconstruct(meas_amp, model)

        for val in loss_history:
            assert val >= 0.0

    def test_reconstruct_with_positivity(self):
        """With positivity=True, reconstruction should have no negative values."""
        model, config = _build_model()
        dn_true = torch.zeros(*config.volume_shape, dtype=torch.float64)
        with torch.no_grad():
            measurements = model.forward(dn_true)
        meas_amp = torch.sqrt(measurements)

        rec = ReflectionBPMReconstructor(n_iter=3, lr=1.0,
                                         tv_weight=0.0,
                                         positivity=True, device="cpu")
        dn_recon, _ = rec.reconstruct(meas_amp, model)

        assert np.all(dn_recon >= -1e-15)

    def test_reconstruct_with_tv_regularization(self):
        """
        Reconstruction with TV > 0 should still produce valid output
        with finite values and correct shape.
        """
        model, config = _build_model()
        rng = np.random.default_rng(10)
        dn_true = torch.tensor(
            rng.uniform(0.0, 0.01, config.volume_shape),
            dtype=torch.float64,
        )
        with torch.no_grad():
            measurements = model.forward(dn_true)
        meas_amp = torch.sqrt(measurements)

        rec = ReflectionBPMReconstructor(n_iter=3, lr=1.0,
                                         tv_weight=1e-4, device="cpu")
        dn_recon, loss_history = rec.reconstruct(meas_amp, model)

        assert dn_recon.shape == config.volume_shape
        assert np.all(np.isfinite(dn_recon))
        assert all(np.isfinite(v) for v in loss_history)
