"""Tests for solvers module (lucky_imaging)."""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.preprocessing import prepare_all_frames
from src.solvers import (
    rank_frames,
    find_alignment_rect,
    create_ap_grid,
    one_dim_weight,
    unsharp_mask,
)


# ---------------------------------------------------------------------------
# Shared helper to build a small frames_data dict
# ---------------------------------------------------------------------------
def _make_frames_data(n=5, h=64, w=64, seed=0):
    """Create a small synthetic frames_data dict via prepare_all_frames."""
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 256, (n, h, w, 3), dtype=np.uint8)
    return prepare_all_frames(frames, gauss_width=7, stride=2), frames


# ---------------------------------------------------------------------------
# rank_frames
# ---------------------------------------------------------------------------
class TestRankFrames:
    def test_output_shapes(self):
        fd, _ = _make_frames_data(n=5)
        scores, indices = rank_frames(fd, method="Laplace")
        assert scores.shape == (5,)
        assert indices.shape == (5,)

    def test_scores_normalised(self):
        """Max score should be 1.0 after normalisation."""
        fd, _ = _make_frames_data(n=8, seed=1)
        scores, _ = rank_frames(fd, method="Laplace")
        assert scores.max() == pytest.approx(1.0, abs=1e-10)

    def test_sorted_indices_descending(self):
        """sorted_indices should be in descending quality order."""
        fd, _ = _make_frames_data(n=6, seed=2)
        scores, indices = rank_frames(fd, method="Laplace")
        for i in range(len(indices) - 1):
            assert scores[indices[i]] >= scores[indices[i + 1]]

    def test_all_methods(self):
        """All three ranking methods should run without error."""
        fd, _ = _make_frames_data(n=3, seed=3)
        for method in ("Laplace", "Gradient", "Sobel"):
            scores, indices = rank_frames(fd, method=method)
            assert scores.shape == (3,)
            assert np.all(scores >= 0)

    def test_invalid_method_raises(self):
        fd, _ = _make_frames_data(n=3)
        with pytest.raises(ValueError, match="Unknown ranking method"):
            rank_frames(fd, method="InvalidMethod")


# ---------------------------------------------------------------------------
# find_alignment_rect
# ---------------------------------------------------------------------------
class TestFindAlignmentRect:
    def test_returns_four_tuple(self):
        rng = np.random.default_rng(10)
        frame = rng.integers(100, 60000, (128, 128), dtype=np.uint16)
        rect = find_alignment_rect(frame, search_width=10, border_width=4)
        assert len(rect) == 4

    def test_rect_within_frame_bounds(self):
        rng = np.random.default_rng(11)
        H, W = 128, 128
        frame = rng.integers(100, 60000, (H, W), dtype=np.uint16)
        y_lo, y_hi, x_lo, x_hi = find_alignment_rect(
            frame, search_width=10, border_width=4
        )
        assert 0 <= y_lo < y_hi <= H
        assert 0 <= x_lo < x_hi <= W

    def test_rect_positive_area(self):
        rng = np.random.default_rng(12)
        frame = rng.integers(100, 60000, (64, 64), dtype=np.uint16)
        y_lo, y_hi, x_lo, x_hi = find_alignment_rect(
            frame, search_width=4, border_width=2
        )
        assert (y_hi - y_lo) > 0
        assert (x_hi - x_lo) > 0


# ---------------------------------------------------------------------------
# create_ap_grid
# ---------------------------------------------------------------------------
class TestCreateApGrid:
    def test_returns_list_of_dicts(self):
        """AP grid on a textured mean frame should produce a non-empty list."""
        rng = np.random.default_rng(20)
        mean_frame = rng.integers(5000, 50000, (256, 256), dtype=np.int32)
        aps = create_ap_grid(
            mean_frame,
            half_box_width=16,
            structure_threshold=0.0,  # keep all
            brightness_threshold=1,
            search_width=8,
        )
        assert isinstance(aps, list)
        assert len(aps) > 0
        assert isinstance(aps[0], dict)

    def test_ap_keys(self):
        rng = np.random.default_rng(21)
        mean_frame = rng.integers(5000, 50000, (256, 256), dtype=np.int32)
        aps = create_ap_grid(
            mean_frame,
            half_box_width=16,
            structure_threshold=0.0,
            brightness_threshold=1,
            search_width=8,
        )
        required_keys = {'y', 'x', 'half_box_width', 'box_y_low', 'box_y_high',
                         'box_x_low', 'box_x_high', 'reference_box', 'structure'}
        for ap in aps:
            assert required_keys.issubset(ap.keys())

    def test_high_threshold_filters_all(self):
        """With structure_threshold = 1.0 no AP should survive (unless all
        are exactly at the max, which is vanishingly unlikely)."""
        rng = np.random.default_rng(22)
        mean_frame = rng.integers(5000, 50000, (256, 256), dtype=np.int32)
        aps = create_ap_grid(
            mean_frame,
            half_box_width=16,
            structure_threshold=1.0,
            brightness_threshold=1,
            search_width=8,
        )
        # At most 1 AP could survive (the one with max structure = 1.0)
        assert len(aps) <= 1


# ---------------------------------------------------------------------------
# one_dim_weight
# ---------------------------------------------------------------------------
class TestOneDimWeight:
    def test_output_shape(self):
        w = one_dim_weight(10, 50, 30)
        assert w.shape == (40,)

    def test_dtype_float32(self):
        w = one_dim_weight(0, 20, 10)
        assert w.dtype == np.float32

    def test_peak_at_center(self):
        """Weight should be highest at the center offset."""
        w = one_dim_weight(0, 40, 20)
        center_idx = 20  # box_center - patch_low
        assert w[center_idx] >= w[0]
        assert w[center_idx] >= w[-1]

    def test_all_positive(self):
        """All weights should be positive."""
        w = one_dim_weight(0, 30, 15)
        assert np.all(w > 0)

    def test_extend_low_flat(self):
        """With extend_low=True the lower ramp should be all 1.0."""
        w = one_dim_weight(0, 40, 20, extend_low=True)
        np.testing.assert_allclose(w[:20], 1.0, atol=1e-6)

    def test_extend_high_flat(self):
        """With extend_high=True the upper ramp should be all 1.0."""
        w = one_dim_weight(0, 40, 20, extend_high=True)
        np.testing.assert_allclose(w[20:], 1.0, atol=1e-6)

    def test_triangular_symmetry(self):
        """For a centered AP the weights should be symmetric."""
        w = one_dim_weight(0, 40, 20)
        np.testing.assert_allclose(w[:20], w[39:19:-1], atol=1e-6)


# ---------------------------------------------------------------------------
# unsharp_mask
# ---------------------------------------------------------------------------
class TestUnsharpMask:
    def test_output_shape_and_dtype(self):
        img = np.full((32, 32, 3), 30000, dtype=np.uint16)
        out = unsharp_mask(img, sigma=2.0, alpha=1.5)
        assert out.shape == img.shape
        assert out.dtype == np.uint16

    def test_clipped_to_uint16_range(self):
        """Output values must stay in [0, 65535]."""
        rng = np.random.default_rng(30)
        img = rng.integers(0, 65535, (32, 32, 3), dtype=np.uint16)
        out = unsharp_mask(img, sigma=2.0, alpha=3.0)
        assert out.min() >= 0
        assert out.max() <= 65535

    def test_uniform_image_unchanged(self):
        """A uniform image has no high-frequency content, so unsharp
        masking should leave it unchanged."""
        img = np.full((32, 32), 20000, dtype=np.uint16)
        out = unsharp_mask(img, sigma=2.0, alpha=1.5)
        np.testing.assert_allclose(out.astype(float), 20000.0, atol=1.0)

    def test_alpha_zero_identity(self):
        """With alpha=0 the output should equal the input."""
        rng = np.random.default_rng(31)
        img = rng.integers(0, 65535, (16, 16, 3), dtype=np.uint16)
        out = unsharp_mask(img, sigma=2.0, alpha=0.0)
        np.testing.assert_array_equal(out, img)

    def test_sharpening_increases_laplacian_variance(self):
        """Sharpened image should have higher Laplacian variance than the
        original (more high-frequency content)."""
        rng = np.random.default_rng(32)
        img = rng.integers(5000, 60000, (64, 64), dtype=np.uint16)
        out = unsharp_mask(img, sigma=2.0, alpha=2.0)
        lap_orig = cv2.Laplacian(img.astype(np.float32), cv2.CV_32F).var()
        lap_sharp = cv2.Laplacian(out.astype(np.float32), cv2.CV_32F).var()
        assert lap_sharp > lap_orig
