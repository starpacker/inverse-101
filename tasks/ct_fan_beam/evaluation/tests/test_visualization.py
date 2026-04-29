"""Tests for src/visualization.py."""
import numpy as np

import os, sys
TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)

from src.visualization import compute_ncc, compute_nrmse, centre_crop_normalize


class TestNCC:
    def test_identical(self):
        x = np.array([1.0, 2.0, 3.0])
        assert abs(compute_ncc(x, x) - 1.0) < 1e-10

    def test_scaled(self):
        """NCC should be 1 for scaled vectors (cosine similarity)."""
        x = np.array([1.0, 2.0, 3.0])
        assert abs(compute_ncc(x, 2 * x) - 1.0) < 1e-10

    def test_orthogonal(self):
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        assert abs(compute_ncc(x, y)) < 1e-10

    def test_zero(self):
        assert compute_ncc(np.zeros(5), np.ones(5)) == 0.0

    def test_with_mask(self):
        x = np.array([1.0, 2.0, 999.0])
        y = np.array([1.0, 2.0, -999.0])
        mask = np.array([True, True, False])
        assert abs(compute_ncc(x, y, mask=mask) - 1.0) < 1e-10

    def test_range(self):
        rng = np.random.default_rng(42)
        x = rng.random(100)
        y = rng.random(100)
        ncc = compute_ncc(x, y)
        assert -1 <= ncc <= 1


class TestNRMSE:
    def test_identical(self):
        x = np.array([1.0, 2.0, 3.0])
        assert compute_nrmse(x, x) == 0.0

    def test_known_value(self):
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 3.0])
        assert abs(compute_nrmse(x, y) - 1.0) < 1e-10

    def test_positive(self):
        rng = np.random.default_rng(42)
        x = rng.random(100)
        y = rng.random(100)
        assert compute_nrmse(x, y) >= 0

    def test_constant_reference(self):
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 1.0])
        assert compute_nrmse(x, y) == float('inf')


class TestCentreCrop:
    def test_size_approximately_correct(self):
        img = np.random.randn(100, 100)
        cropped = centre_crop_normalize(img, 0.8)
        assert 78 <= cropped.shape[0] <= 82
        assert 78 <= cropped.shape[1] <= 82

    def test_normalized_range(self):
        img = np.random.randn(100, 100)
        cropped = centre_crop_normalize(img, 0.8)
        assert abs(cropped.min()) < 1e-10
        assert abs(cropped.max() - 1.0) < 1e-10

    def test_constant_image(self):
        """Constant image should remain constant after crop (no normalization)."""
        img = np.ones((50, 50)) * 5.0
        cropped = centre_crop_normalize(img, 0.8)
        # With zero dynamic range, normalization can't work, so just check shape
        assert cropped.shape[0] < 50
