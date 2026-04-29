"""Unit tests for seismic_FWI_original src/physics_model.py.

All functions are deterministic → exact numerical comparison (rtol=1e-5, atol=1e-6).
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.physics_model import (
    _fft_downsample,
    _fft_upsample,
    _loc_to_flat,
    cfl_step_ratio,
    forward_model,
    make_acquisition_geometry,
    make_ricker_wavelet,
    setup_pml_profiles,
    wave_step,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"


# ---------------------------------------------------------------------------
# Ricker wavelet
# ---------------------------------------------------------------------------

class TestMakeRickerWavelet:
    def test_shape(self):
        w = make_ricker_wavelet(5.0, 100, 0.004, 4)
        assert w.shape == (4, 1, 100)

    def test_dtype(self):
        w = make_ricker_wavelet(5.0, 50, 0.004, 2)
        assert w.dtype == torch.float32

    def test_peak_is_positive(self):
        w = make_ricker_wavelet(5.0, 200, 0.004, 1)
        # peak of Ricker at t_p = 1.5/freq
        assert w.max() > 0

    def test_numerics(self):
        w = make_ricker_wavelet(freq=5.0, nt=50, dt=0.004, n_shots=2)
        ref = np.load(FIXTURES / "output_ricker_wavelet.npy")
        np.testing.assert_allclose(w.numpy(), ref, rtol=1e-5, atol=1e-7)

    def test_same_across_shots(self):
        w = make_ricker_wavelet(5.0, 50, 0.004, 3)
        np.testing.assert_array_equal(w[0].numpy(), w[1].numpy())
        np.testing.assert_array_equal(w[0].numpy(), w[2].numpy())


# ---------------------------------------------------------------------------
# Acquisition geometry
# ---------------------------------------------------------------------------

class TestMakeAcquisitionGeometry:
    def test_shapes(self):
        src, rec = make_acquisition_geometry(50, n_shots=3, n_receivers=7)
        assert src.shape == (3, 1, 2)
        assert rec.shape == (3, 7, 2)

    def test_source_depth(self):
        src, _ = make_acquisition_geometry(50, n_shots=4, source_depth=2)
        assert (src[..., 1] == 2).all()

    def test_receiver_depth(self):
        _, rec = make_acquisition_geometry(50, n_receivers=5, receiver_depth=3)
        assert (rec[..., 1] == 3).all()

    def test_numerics(self):
        src, rec = make_acquisition_geometry(nx=50, n_shots=3, n_receivers=7)
        ref_src = np.load(FIXTURES / "output_src_loc.npy")
        ref_rec = np.load(FIXTURES / "output_rec_loc.npy")
        np.testing.assert_array_equal(src.numpy(), ref_src)
        np.testing.assert_array_equal(rec.numpy(), ref_rec)


# ---------------------------------------------------------------------------
# CFL step ratio
# ---------------------------------------------------------------------------

class TestCflStepRatio:
    def test_step_ratio_positive(self):
        idt, sr = cfl_step_ratio(20., 20., 0.004, 5000.)
        assert sr >= 1

    def test_inner_dt_satisfies_cfl(self):
        dy, dx, dt, v_max = 20., 20., 0.004, 5000.
        idt, sr = cfl_step_ratio(dy, dx, dt, v_max)
        max_allowed = 0.6 / math.sqrt(1/dy**2 + 1/dx**2) / v_max
        assert abs(idt) <= max_allowed + 1e-10

    def test_inner_dt_times_ratio_eq_dt(self):
        idt, sr = cfl_step_ratio(20., 20., 0.004, 5000.)
        assert abs(idt * sr - 0.004) < 1e-10

    def test_numerics(self):
        idt, sr = cfl_step_ratio(dy=20., dx=20., dt=0.004, v_max=5000.)
        ref = np.load(FIXTURES / "output_cfl.npy")
        assert abs(idt - ref[0]) < 1e-7
        assert sr == int(ref[1])


# ---------------------------------------------------------------------------
# FFT upsample / downsample
# ---------------------------------------------------------------------------

class TestFftResample:
    def test_upsample_shape(self):
        sig = torch.randn(3, 40)
        up = _fft_upsample(sig, step_ratio=3)
        assert up.shape == (3, 120)

    def test_downsample_shape(self):
        sig = torch.randn(3, 120)
        dn = _fft_downsample(sig, step_ratio=3)
        assert dn.shape == (3, 40)

    def test_identity_step_ratio_1(self):
        sig = torch.randn(2, 50)
        assert torch.equal(_fft_upsample(sig, 1), sig)
        assert torch.equal(_fft_downsample(sig, 1), sig)

    def test_roundtrip(self):
        """Upsample then downsample should recover original with zeroed Nyquist.

        Both _fft_upsample and _fft_downsample zero the Nyquist component,
        so the roundtrip is exact for any signal after removing its Nyquist.
        """
        sig = np.load(FIXTURES / "input_signal.npy")
        sig_t = torch.from_numpy(sig)
        # Expected output: original with Nyquist zeroed (both ends of the pipeline zero it)
        sig_f = torch.fft.rfft(sig_t, norm="ortho")
        sig_f = sig_f.clone()
        sig_f[..., -1] = 0
        sig_no_nyq = torch.fft.irfft(sig_f, n=sig_t.shape[-1], norm="ortho")
        dn = _fft_downsample(_fft_upsample(sig_t, 3), 3)
        np.testing.assert_allclose(dn.numpy(), sig_no_nyq.numpy(), rtol=1e-5, atol=1e-6)

    def test_upsample_numerics(self):
        sig = torch.from_numpy(np.load(FIXTURES / "input_signal.npy"))
        ref = np.load(FIXTURES / "output_upsample_3.npy")
        out = _fft_upsample(sig, 3)
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-6)

    def test_downsample_numerics(self):
        sig = torch.from_numpy(np.load(FIXTURES / "input_signal.npy"))
        up = _fft_upsample(sig, 3)
        ref = np.load(FIXTURES / "output_downsample_3.npy")
        dn = _fft_downsample(up, 3)
        np.testing.assert_allclose(dn.numpy(), ref, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Location helper
# ---------------------------------------------------------------------------

class TestLocToFlat:
    def test_single(self):
        loc = torch.tensor([[2, 3]])  # dim0=2, dim1=3
        idx = _loc_to_flat(loc, pad_y=5, pad_x=5, nx_p=20)
        expected = (2 + 5) * 20 + (3 + 5)   # (dim0+pad_y)*nx_p + (dim1+pad_x)
        assert idx.item() == expected

    def test_batch(self):
        loc = torch.tensor([[0, 0], [1, 1]])
        idx = _loc_to_flat(loc, pad_y=2, pad_x=2, nx_p=10)
        assert idx[0].item() == (0 + 2) * 10 + (0 + 2)
        assert idx[1].item() == (1 + 2) * 10 + (1 + 2)


# ---------------------------------------------------------------------------
# PML profiles
# ---------------------------------------------------------------------------

class TestSetupPmlProfiles:
    def test_shapes(self):
        ny_p, nx_p, pml_width, fd_pad = 50, 60, 10, 2
        profiles = setup_pml_profiles(
            ny_p, nx_p, pml_width, fd_pad,
            20., 20., 0.001, 5000.,
            torch.float32, torch.device("cpu"), 5.0,
        )
        ay, by, dbydy, ax, bx, dbxdx = profiles
        assert ay.shape == (ny_p, 1)
        assert ax.shape == (1, nx_p)

    def test_interior_is_zero(self):
        ny_p, nx_p, pml_width, fd_pad = 50, 60, 10, 2
        interior_start = fd_pad + pml_width
        profiles = setup_pml_profiles(
            ny_p, nx_p, pml_width, fd_pad,
            20., 20., 0.001, 5000.,
            torch.float32, torch.device("cpu"), 5.0,
        )
        ay = profiles[0].squeeze()
        # Interior y-range
        assert ay[interior_start : ny_p - fd_pad - pml_width].abs().max() == 0


# ---------------------------------------------------------------------------
# Wave step
# ---------------------------------------------------------------------------

class TestWaveStep:
    def _tiny_setup(self):
        ny_p, nx_p = 20, 25
        pml_width, fd_pad = 5, 2
        dy, dx, inner_dt = 20., 20., 0.001
        v_p = torch.ones(ny_p, nx_p) * 2000.
        wfc = torch.zeros(ny_p, nx_p)
        wfp = torch.zeros(ny_p, nx_p)
        psi_y = torch.zeros_like(wfc)
        psi_x = torch.zeros_like(wfc)
        zeta_y = torch.zeros_like(wfc)
        zeta_x = torch.zeros_like(wfc)
        from src.physics_model import setup_pml_profiles
        pml = setup_pml_profiles(
            ny_p, nx_p, pml_width, fd_pad,
            dy, dx, inner_dt, 2000., torch.float32, torch.device("cpu"), 5.0,
        )
        return v_p, wfc, wfp, psi_y, psi_x, zeta_y, zeta_x, pml, dy, dx, inner_dt

    def test_output_shapes(self):
        args = self._tiny_setup()
        result = wave_step(*args)
        assert len(result) == 5
        ny_p, nx_p = 20, 25
        for t in result:
            assert t.shape == (ny_p, nx_p)

    def test_zero_field_stays_zero(self):
        """Zero wavefield should produce zero update."""
        args = self._tiny_setup()
        new_wfc, *_ = wave_step(*args)
        assert new_wfc.abs().max().item() == 0.0

    def test_differentiable(self):
        v_p, wfc, wfp, psi_y, psi_x, zeta_y, zeta_x, pml, dy, dx, idt = self._tiny_setup()
        v_p.requires_grad_(True)
        # Inject a tiny source so wavefield is nonzero
        wfc = torch.zeros_like(wfc)
        wfc[10, 12] = 1e-3
        new_wfc, *_ = wave_step(v_p, wfc, wfp, psi_y, psi_x, zeta_y, zeta_x, pml, dy, dx, idt)
        new_wfc.sum().backward()
        assert v_p.grad is not None
        assert v_p.grad.abs().max() > 0


# ---------------------------------------------------------------------------
# Forward model (end-to-end, tiny model)
# ---------------------------------------------------------------------------

class TestForwardModel:
    def test_shape(self):
        v = torch.ones(20, 30) * 2000.
        src, rec = make_acquisition_geometry(30, n_shots=2, n_receivers=5)
        amp = make_ricker_wavelet(5.0, 50, 0.004, 2)
        out = forward_model(v, (20., 20.), 0.004, amp, src, rec, 5.0, pml_width=10)
        assert out.shape == (2, 5, 50)

    def test_dtype(self):
        v = torch.ones(20, 30) * 2000.
        src, rec = make_acquisition_geometry(30, n_shots=2, n_receivers=5)
        amp = make_ricker_wavelet(5.0, 50, 0.004, 2)
        out = forward_model(v, (20., 20.), 0.004, amp, src, rec, 5.0, pml_width=10)
        assert out.dtype == torch.float32

    def test_numerics(self):
        v = torch.ones(20, 30) * 2000.
        src, rec = make_acquisition_geometry(30, n_shots=2, n_receivers=5)
        amp = make_ricker_wavelet(5.0, 50, 0.004, 2)
        out = forward_model(v, (20., 20.), 0.004, amp, src, rec, 5.0, pml_width=10)
        ref = np.load(FIXTURES / "output_forward_tiny.npy")
        np.testing.assert_allclose(out.detach().numpy(), ref, rtol=1e-5, atol=1e-7)

    def test_gradient_flows(self):
        v = torch.ones(20, 30) * 2000.
        v.requires_grad_(True)
        src, rec = make_acquisition_geometry(30, n_shots=2, n_receivers=5)
        amp = make_ricker_wavelet(5.0, 50, 0.004, 2)
        out = forward_model(v, (20., 20.), 0.004, amp, src, rec, 5.0, pml_width=10)
        out.sum().backward()
        assert v.grad is not None
        assert v.grad.abs().max() > 0

    def test_no_grad_matches_grad(self):
        """Forward pass with and without gradient should give identical output."""
        v = torch.ones(20, 30) * 2000.
        src, rec = make_acquisition_geometry(30, n_shots=2, n_receivers=5)
        amp = make_ricker_wavelet(5.0, 50, 0.004, 2)
        out_ng = forward_model(v, (20., 20.), 0.004, amp, src, rec, 5.0, pml_width=10)
        v2 = v.clone().requires_grad_(True)
        out_g = forward_model(v2, (20., 20.), 0.004, amp, src, rec, 5.0, pml_width=10)
        np.testing.assert_allclose(out_ng.numpy(), out_g.detach().numpy(), rtol=1e-5, atol=1e-7)
