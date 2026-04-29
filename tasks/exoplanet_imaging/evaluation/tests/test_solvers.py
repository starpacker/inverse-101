"""Tests for solvers module (KLIP PSF subtraction, derotation, combination, full pipeline)."""

import os
import sys
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.solvers import (
    compute_psf_residuals,
    derotate_cube,
    combine_frames,
    klip_adi,
)


# ---------------------------------------------------------------------------
# Helper: small synthetic ADI cube
# ---------------------------------------------------------------------------

def _make_cube(N=8, H=16, W=16, seed=42):
    """Create a small synthetic ADI cube and parallactic angles."""
    rng = np.random.default_rng(seed)
    cube = rng.standard_normal((N, H, W)).astype(np.float32)
    angles = np.linspace(-10, 10, N).astype(np.float32)
    return cube, angles


# ---------------------------------------------------------------------------
# compute_psf_residuals
# ---------------------------------------------------------------------------

class TestComputePsfResiduals:
    def test_output_shape_single_k(self):
        cube, _ = _make_cube()
        res = compute_psf_residuals(cube, K_klip=3)
        assert res.shape == (1, 8, 16, 16)

    def test_output_shape_multiple_k(self):
        cube, _ = _make_cube()
        res = compute_psf_residuals(cube, K_klip=[2, 4])
        assert res.shape == (2, 8, 16, 16)

    def test_residual_variance_decreases_with_k(self):
        """More KL modes removed should yield smaller residual variance."""
        cube, _ = _make_cube()
        res = compute_psf_residuals(cube, K_klip=[1, 5])
        var_k1 = torch.nanmean(res[0] ** 2).item()
        var_k5 = torch.nanmean(res[1] ** 2).item()
        assert var_k5 < var_k1

    def test_residual_dtype(self):
        cube, _ = _make_cube()
        res = compute_psf_residuals(cube, K_klip=2)
        assert res.dtype == torch.float32


# ---------------------------------------------------------------------------
# derotate_cube
# ---------------------------------------------------------------------------

class TestDerotateCube:
    def test_output_shape(self):
        residuals = torch.randn(1, 5, 16, 16)
        angles = np.linspace(-10, 10, 5).astype(np.float32)
        out = derotate_cube(residuals, angles)
        assert out.shape == residuals.shape

    def test_zero_angles_identity(self):
        """Zero parallactic angles should leave residuals nearly unchanged."""
        residuals = torch.randn(1, 4, 16, 16)
        angles = np.zeros(4, dtype=np.float32)
        out = derotate_cube(residuals, angles)
        inner = slice(2, -2)
        np.testing.assert_allclose(
            out[0, :, inner, inner].numpy(),
            residuals[0, :, inner, inner].numpy(),
            atol=0.05,
        )


# ---------------------------------------------------------------------------
# combine_frames
# ---------------------------------------------------------------------------

class TestCombineFrames:
    def test_mean_output_shape(self):
        data = torch.randn(2, 5, 16, 16)
        out = combine_frames(data, statistic='mean')
        assert out.shape == (2, 16, 16)

    def test_median_output_shape(self):
        data = torch.randn(1, 6, 8, 8)
        out = combine_frames(data, statistic='median')
        assert out.shape == (1, 8, 8)

    def test_mean_value(self):
        """Mean combination of identical frames should equal that frame."""
        frame = torch.randn(1, 1, 8, 8)
        data = frame.expand(1, 5, 8, 8).clone()
        out = combine_frames(data, statistic='mean')
        np.testing.assert_allclose(
            out[0].numpy(), frame[0, 0].numpy(), atol=1e-6
        )

    def test_invalid_statistic_raises(self):
        data = torch.randn(1, 3, 8, 8)
        with pytest.raises(ValueError, match="statistic must be"):
            combine_frames(data, statistic='max')


# ---------------------------------------------------------------------------
# klip_adi (full pipeline)
# ---------------------------------------------------------------------------

class TestKlipAdi:
    def test_output_shape_single_k(self):
        cube, angles = _make_cube(N=8, H=16, W=16)
        result = klip_adi(cube, angles, K_klip=3)
        assert result.shape == (16, 16)

    def test_output_shape_multiple_k(self):
        cube, angles = _make_cube(N=8, H=16, W=16)
        result = klip_adi(cube, angles, K_klip=[2, 4])
        assert result.shape == (2, 16, 16)

    def test_iwa_mask_propagated(self):
        """Pixels within IWA should be NaN in the final output."""
        cube, angles = _make_cube(N=8, H=32, W=32)
        center = (16, 16)
        iwa = 4.0
        result = klip_adi(cube, angles, K_klip=3, iwa=iwa, center=center)
        # Center pixel should be NaN
        assert np.isnan(result[16, 16])

    def test_output_dtype(self):
        cube, angles = _make_cube()
        result = klip_adi(cube, angles, K_klip=2)
        assert result.dtype == np.float32
