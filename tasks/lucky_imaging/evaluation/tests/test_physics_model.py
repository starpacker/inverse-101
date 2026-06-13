"""Tests for physics_model module (lucky_imaging)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.physics_model import (
    quality_measure_gradient,
    quality_measure_laplace,
    quality_measure_sobel,
    quality_measure,
    quality_measure_threshold_weighted,
    phase_correlation,
    sub_pixel_solve,
)


# ---------------------------------------------------------------------------
# quality_measure_gradient
# ---------------------------------------------------------------------------
class TestQualityMeasureGradient:
    def test_uniform_frame_zero_quality(self):
        """A constant frame has zero gradient everywhere."""
        frame = np.full((32, 32), 1000, dtype=np.uint16)
        q = quality_measure_gradient(frame, stride=2)
        assert q == pytest.approx(0.0, abs=1e-10)

    def test_positive_quality_for_varying_frame(self):
        """A non-constant frame should have positive gradient quality."""
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 65535, (32, 32), dtype=np.uint16)
        q = quality_measure_gradient(frame, stride=2)
        assert q > 0.0

    def test_sharper_frame_higher_quality(self):
        """A high-contrast edge should score higher than a smooth gradient."""
        smooth = np.tile(np.linspace(100, 200, 32, dtype=np.uint16), (32, 1))
        sharp = np.zeros((32, 32), dtype=np.uint16)
        sharp[:, :16] = 100
        sharp[:, 16:] = 60000
        assert quality_measure_gradient(sharp) > quality_measure_gradient(smooth)

    def test_output_is_scalar_float(self):
        frame = np.ones((16, 16), dtype=np.uint16) * 128
        q = quality_measure_gradient(frame)
        assert isinstance(q, float)


# ---------------------------------------------------------------------------
# quality_measure_laplace
# ---------------------------------------------------------------------------
class TestQualityMeasureLaplace:
    def test_uniform_frame_zero_quality(self):
        """Laplacian of a constant image should be zero."""
        frame = np.full((32, 32), 5000, dtype=np.uint16)
        q = quality_measure_laplace(frame, stride=2)
        assert q == pytest.approx(0.0, abs=1e-5)

    def test_positive_for_textured_frame(self):
        rng = np.random.default_rng(1)
        frame = rng.integers(0, 65535, (32, 32), dtype=np.uint16)
        q = quality_measure_laplace(frame, stride=2)
        assert q > 0.0

    def test_output_is_scalar_float(self):
        frame = np.ones((16, 16), dtype=np.uint16)
        q = quality_measure_laplace(frame)
        assert isinstance(q, float)


# ---------------------------------------------------------------------------
# quality_measure_sobel
# ---------------------------------------------------------------------------
class TestQualityMeasureSobel:
    def test_uniform_frame_zero_quality(self):
        """Sobel of a constant frame should be zero."""
        frame = np.full((32, 32), 128, dtype=np.uint8)
        q = quality_measure_sobel(frame, stride=2)
        assert q == pytest.approx(0.0, abs=1e-10)

    def test_positive_for_varying_frame(self):
        rng = np.random.default_rng(2)
        frame = rng.integers(0, 256, (32, 32), dtype=np.uint8)
        q = quality_measure_sobel(frame, stride=2)
        assert q > 0.0

    def test_output_is_scalar_float(self):
        frame = np.ones((16, 16), dtype=np.uint8)
        q = quality_measure_sobel(frame)
        assert isinstance(q, float)


# ---------------------------------------------------------------------------
# quality_measure  (structure measure used for AP filtering)
# ---------------------------------------------------------------------------
class TestQualityMeasure:
    def test_uniform_patch_zero(self):
        """Flat patch has no structure."""
        patch = np.full((16, 16), 100.0, dtype=np.float64)
        assert quality_measure(patch) == pytest.approx(0.0, abs=1e-10)

    def test_positive_for_textured_patch(self):
        rng = np.random.default_rng(3)
        patch = rng.standard_normal((16, 16))
        assert quality_measure(patch) > 0.0

    def test_returns_min_of_directions(self):
        """Horizontal-only edges: vertical gradient should be near zero, so
        result (min of horiz, vert averages) should be near zero."""
        patch = np.zeros((16, 16), dtype=np.float64)
        for r in range(16):
            patch[r, :] = r * 10.0  # vertical gradient only
        q = quality_measure(patch)
        # x-diff of a row-only ramp is zero => min is 0
        assert q == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# quality_measure_threshold_weighted
# ---------------------------------------------------------------------------
class TestQualityMeasureThresholdWeighted:
    def test_all_black_low_quality(self):
        """An all-black frame should produce near-zero quality."""
        frame = np.zeros((32, 32), dtype=np.float64)
        q = quality_measure_threshold_weighted(frame, stride=2)
        assert q == pytest.approx(0.0, abs=1e-10)

    def test_bright_textured_positive(self):
        rng = np.random.default_rng(4)
        frame = rng.integers(100, 256, (32, 32)).astype(np.float64)
        q = quality_measure_threshold_weighted(frame, stride=2)
        assert q > 0.0

    def test_output_is_scalar_float(self):
        frame = np.ones((16, 16), dtype=np.float64) * 128
        q = quality_measure_threshold_weighted(frame)
        assert isinstance(q, float)


# ---------------------------------------------------------------------------
# phase_correlation
# ---------------------------------------------------------------------------
class TestPhaseCorrelation:
    def test_no_shift(self):
        """Identical frames should yield zero shift."""
        rng = np.random.default_rng(10)
        frame = rng.standard_normal((32, 32))
        dy, dx = phase_correlation(frame, frame, (32, 32))
        assert dy == 0
        assert dx == 0

    def test_known_shift(self):
        """A rolled frame should be detected with the correct shift."""
        rng = np.random.default_rng(11)
        frame0 = rng.standard_normal((32, 32))
        shift_y, shift_x = 3, -2
        frame1 = np.roll(np.roll(frame0, shift_y, axis=0), shift_x, axis=1)
        dy, dx = phase_correlation(frame0, frame1, (32, 32))
        assert dy == shift_y
        assert dx == shift_x

    def test_output_types(self):
        frame = np.ones((16, 16), dtype=np.float64)
        dy, dx = phase_correlation(frame, frame, (16, 16))
        assert isinstance(dy, int)
        assert isinstance(dx, int)


# ---------------------------------------------------------------------------
# sub_pixel_solve
# ---------------------------------------------------------------------------
class TestSubPixelSolve:
    def test_peak_at_center_gives_zero_correction(self):
        """If the maximum is already at the grid center the correction
        should be (approximately) zero."""
        vals = np.array([[1.0, 2.0, 1.0],
                         [2.0, 5.0, 2.0],
                         [1.0, 2.0, 1.0]])
        dy, dx = sub_pixel_solve(vals)
        assert abs(dy) < 0.5
        assert abs(dx) < 0.5

    def test_symmetric_peak(self):
        """A perfectly symmetric peak should give near-zero correction."""
        vals = np.array([[0.0, 1.0, 0.0],
                         [1.0, 4.0, 1.0],
                         [0.0, 1.0, 0.0]])
        dy, dx = sub_pixel_solve(vals)
        assert abs(dy) < 1e-6
        assert abs(dx) < 1e-6

    def test_correction_bounded(self):
        """For a reasonable peak the corrections should be within +/-1 pixel."""
        rng = np.random.default_rng(20)
        vals = rng.standard_normal((3, 3))
        vals[1, 1] = vals.max() + 2.0  # clear peak at center
        dy, dx = sub_pixel_solve(vals)
        assert abs(dy) <= 1.5
        assert abs(dx) <= 1.5
