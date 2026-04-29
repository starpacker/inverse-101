"""Tests for preprocessing module (lucky_imaging)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.preprocessing import (
    to_mono,
    gaussian_blur,
    average_brightness,
    compute_laplacian,
    prepare_all_frames,
)


# ---------------------------------------------------------------------------
# to_mono
# ---------------------------------------------------------------------------
class TestToMono:
    def test_output_shape(self):
        """RGB input (H, W, 3) should produce (H, W) grayscale."""
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        mono = to_mono(frame)
        assert mono.shape == (32, 32)

    def test_output_dtype(self):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        mono = to_mono(frame)
        assert mono.dtype == np.uint8

    def test_already_mono_passthrough(self):
        """A 2-D input should be returned unchanged."""
        frame = np.arange(64, dtype=np.uint8).reshape(8, 8)
        mono = to_mono(frame)
        np.testing.assert_array_equal(mono, frame)

    def test_white_frame_is_255(self):
        """All-white RGB frame should map to 255 in mono."""
        frame = np.full((8, 8, 3), 255, dtype=np.uint8)
        mono = to_mono(frame)
        assert mono.max() == 255

    def test_black_frame_is_zero(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        mono = to_mono(frame)
        assert mono.max() == 0

    def test_value_range(self):
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        mono = to_mono(frame)
        assert mono.min() >= 0
        assert mono.max() <= 255


# ---------------------------------------------------------------------------
# gaussian_blur
# ---------------------------------------------------------------------------
class TestGaussianBlur:
    def test_output_shape(self):
        mono = np.zeros((32, 32), dtype=np.uint8)
        blurred = gaussian_blur(mono, gauss_width=7)
        assert blurred.shape == (32, 32)

    def test_output_dtype_uint16(self):
        mono = np.zeros((16, 16), dtype=np.uint8)
        blurred = gaussian_blur(mono, gauss_width=7)
        assert blurred.dtype == np.uint16

    def test_upscale_factor(self):
        """Output should be in the 16-bit range (0..65535), roughly 256x the
        input's uint8 values for a uniform image."""
        val = 100
        mono = np.full((16, 16), val, dtype=np.uint8)
        blurred = gaussian_blur(mono, gauss_width=7)
        # All pixels should be close to val * 256
        np.testing.assert_allclose(blurred.astype(float), val * 256, atol=1)

    def test_blurring_reduces_variance(self):
        """Gaussian blur should reduce the pixel variance of a noisy image."""
        rng = np.random.default_rng(1)
        mono = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        blurred = gaussian_blur(mono, gauss_width=7)
        raw_var = mono.astype(float).var()
        blur_var = blurred.astype(float).var()
        # After upscaling by 256, variance scales by 256^2, but blurring
        # should reduce high-frequency content. Compare normalised variance.
        assert (blur_var / 256**2) < raw_var


# ---------------------------------------------------------------------------
# average_brightness
# ---------------------------------------------------------------------------
class TestAverageBrightness:
    def test_all_black_returns_epsilon(self):
        """All-zero image should return near-zero (epsilon)."""
        frame = np.zeros((16, 16), dtype=np.uint8)
        b = average_brightness(frame)
        assert b > 0
        assert b < 1.0  # essentially epsilon

    def test_bright_frame_positive(self):
        frame = np.full((16, 16), 128, dtype=np.uint8)
        b = average_brightness(frame)
        assert b > 0.0

    def test_clipped_pixels_excluded(self):
        """Pixels outside [low, high] should be thresholded to 0 and
        therefore reduce the mean."""
        frame = np.full((16, 16), 250, dtype=np.uint8)  # above default high=240
        b = average_brightness(frame, low=16, high=240)
        # All pixels are above 240 so should be zeroed out
        assert b < 1.0

    def test_output_is_positive_float(self):
        rng = np.random.default_rng(2)
        frame = rng.integers(0, 256, (16, 16), dtype=np.uint8)
        b = average_brightness(frame)
        assert isinstance(b, float)
        assert b > 0


# ---------------------------------------------------------------------------
# compute_laplacian
# ---------------------------------------------------------------------------
class TestComputeLaplacian:
    def test_output_shape_with_stride(self):
        """Output should be (H//stride, W//stride)."""
        frame = np.zeros((32, 32), dtype=np.uint16)
        lap = compute_laplacian(frame, stride=2)
        assert lap.shape == (16, 16)

    def test_output_dtype_uint8(self):
        frame = np.zeros((16, 16), dtype=np.uint16)
        lap = compute_laplacian(frame, stride=2)
        assert lap.dtype == np.uint8

    def test_uniform_frame_zero_laplacian(self):
        """A constant image should have zero Laplacian."""
        frame = np.full((32, 32), 10000, dtype=np.uint16)
        lap = compute_laplacian(frame, stride=2)
        assert lap.max() == 0


# ---------------------------------------------------------------------------
# prepare_all_frames
# ---------------------------------------------------------------------------
class TestPrepareAllFrames:
    def test_keys_present(self):
        """Output dict should contain all expected keys."""
        rng = np.random.default_rng(3)
        frames = rng.integers(0, 256, (3, 16, 16, 3), dtype=np.uint8)
        out = prepare_all_frames(frames, gauss_width=7, stride=2)
        expected_keys = {'frames_rgb', 'mono', 'blurred', 'brightness',
                         'laplacian', 'n_frames', 'shape'}
        assert expected_keys == set(out.keys())

    def test_n_frames(self):
        rng = np.random.default_rng(4)
        frames = rng.integers(0, 256, (5, 16, 16, 3), dtype=np.uint8)
        out = prepare_all_frames(frames, gauss_width=7, stride=2)
        assert out['n_frames'] == 5

    def test_mono_shapes_and_dtype(self):
        rng = np.random.default_rng(5)
        frames = rng.integers(0, 256, (2, 16, 16, 3), dtype=np.uint8)
        out = prepare_all_frames(frames, gauss_width=7, stride=2)
        for mono in out['mono']:
            assert mono.shape == (16, 16)
            assert mono.dtype == np.uint8

    def test_blurred_dtype_uint16(self):
        rng = np.random.default_rng(6)
        frames = rng.integers(0, 256, (2, 16, 16, 3), dtype=np.uint8)
        out = prepare_all_frames(frames, gauss_width=7, stride=2)
        for b in out['blurred']:
            assert b.dtype == np.uint16

    def test_brightness_all_positive(self):
        rng = np.random.default_rng(7)
        frames = rng.integers(0, 256, (4, 16, 16, 3), dtype=np.uint8)
        out = prepare_all_frames(frames, gauss_width=7, stride=2)
        assert np.all(out['brightness'] > 0)

    def test_shape_matches_input(self):
        rng = np.random.default_rng(8)
        frames = rng.integers(0, 256, (2, 24, 32, 3), dtype=np.uint8)
        out = prepare_all_frames(frames, gauss_width=7, stride=2)
        assert out['shape'] == (24, 32)
