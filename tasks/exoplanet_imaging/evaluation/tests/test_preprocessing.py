"""Tests for preprocessing module (circular masks, mean subtraction, apply mask)."""

import os
import sys
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.preprocessing import (
    create_circular_mask,
    apply_circular_mask,
    mean_subtract_frames,
)


# ---------------------------------------------------------------------------
# create_circular_mask
# ---------------------------------------------------------------------------

class TestCreateCircularMask:
    def test_output_shape_and_dtype(self):
        """Mask should be a bool tensor of (ny, nx)."""
        mask = create_circular_mask(16, 16, center=(8, 8), iwa=3.0)
        assert mask.shape == (16, 16)
        assert mask.dtype == torch.bool

    def test_center_pixel_masked(self):
        """The center pixel should be masked (inside IWA)."""
        mask = create_circular_mask(32, 32, center=(16, 16), iwa=4.0)
        assert mask[16, 16].item() is True

    def test_far_pixel_not_masked_iwa_only(self):
        """A pixel far from center should not be masked when only IWA is set."""
        mask = create_circular_mask(32, 32, center=(16, 16), iwa=3.0)
        assert mask[0, 0].item() is False

    def test_owa_masks_outer_pixels(self):
        """With OWA, corner pixels beyond the outer radius should be masked."""
        mask = create_circular_mask(32, 32, center=(16, 16), iwa=2.0, owa=10.0)
        # Corner (0,0) is sqrt(16^2+16^2) ~ 22.6 px from center, well beyond OWA=10
        assert mask[0, 0].item() is True
        # A pixel just inside OWA should not be masked
        # pixel (16, 24): distance = 8 < 10
        assert mask[16, 24].item() is False

    def test_mask_is_radially_symmetric(self):
        """Mask should be symmetric under 90-degree rotation about center.

        Uses an odd-sized grid so the center pixel aligns with the
        geometric center used by np.rot90.
        """
        mask = create_circular_mask(31, 31, center=(15, 15), iwa=5.0, owa=13.0)
        m = mask.numpy()
        np.testing.assert_array_equal(m, np.rot90(m))


# ---------------------------------------------------------------------------
# apply_circular_mask
# ---------------------------------------------------------------------------

class TestApplyCircularMask:
    def test_output_shape(self):
        cube = np.ones((5, 16, 16), dtype=np.float32)
        out = apply_circular_mask(cube, center=(8, 8), iwa=3.0)
        assert out.shape == cube.shape

    def test_center_becomes_nan(self):
        """Pixels inside IWA should be NaN after masking."""
        cube = np.ones((3, 16, 16), dtype=np.float32)
        out = apply_circular_mask(cube, center=(8, 8), iwa=3.0)
        assert np.isnan(out[0, 8, 8])

    def test_does_not_modify_original(self):
        """apply_circular_mask should not modify the input array in place."""
        cube = np.ones((2, 16, 16), dtype=np.float32)
        _ = apply_circular_mask(cube, center=(8, 8), iwa=3.0)
        assert not np.any(np.isnan(cube))


# ---------------------------------------------------------------------------
# mean_subtract_frames
# ---------------------------------------------------------------------------

class TestMeanSubtractFrames:
    def test_output_shape(self):
        cube = np.random.randn(5, 16, 16).astype(np.float32)
        out = mean_subtract_frames(cube)
        assert out.shape == cube.shape

    def test_per_frame_mean_zero(self):
        """After subtraction, each frame's spatial mean should be ~0."""
        rng = np.random.default_rng(0)
        cube = rng.standard_normal((4, 16, 16)).astype(np.float32) + 100.0
        out = mean_subtract_frames(cube)
        for i in range(cube.shape[0]):
            np.testing.assert_allclose(np.nanmean(out[i]), 0.0, atol=1e-5)

    def test_nan_tolerance(self):
        """Frames with NaN pixels should still have zero mean over non-NaN pixels."""
        rng = np.random.default_rng(7)
        cube = rng.standard_normal((3, 16, 16)).astype(np.float32)
        cube[:, 8, 8] = np.nan
        out = mean_subtract_frames(cube)
        for i in range(cube.shape[0]):
            np.testing.assert_allclose(np.nanmean(out[i]), 0.0, atol=1e-5)

    def test_dtype_preserved(self):
        cube = np.ones((2, 8, 8), dtype=np.float32)
        out = mean_subtract_frames(cube)
        assert out.dtype == np.float32
