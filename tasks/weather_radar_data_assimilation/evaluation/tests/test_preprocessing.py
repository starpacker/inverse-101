"""Tests for preprocessing module."""

import os
import sys
import numpy as np
import pytest

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, TASK_DIR)
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")
DATA_DIR = os.path.join(TASK_DIR, "data")


class TestScaling:
    """Test pixel value scaling functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "preprocessing_scale.npz"))
        self.input_pixels = fix["input_pixels"]
        self.expected_scaled = fix["output_scaled"]

    def test_scale_to_model(self):
        from src.preprocessing import scale_to_model
        result = scale_to_model(self.input_pixels)
        np.testing.assert_allclose(result, self.expected_scaled, rtol=1e-6)

    def test_scale_from_model(self):
        from src.preprocessing import scale_from_model
        result = scale_from_model(self.expected_scaled)
        np.testing.assert_allclose(result, self.input_pixels, atol=1e-6)

    def test_round_trip(self):
        from src.preprocessing import scale_to_model, scale_from_model
        result = scale_from_model(scale_to_model(self.input_pixels))
        np.testing.assert_allclose(result, self.input_pixels, atol=1e-6)


class TestDataLoading:
    """Test data loading functions."""

    def test_load_raw_data_keys(self):
        from src.preprocessing import load_raw_data
        raw = load_raw_data(DATA_DIR)
        assert "condition_frames" in raw
        assert "observations" in raw
        assert "observation_mask" in raw

    def test_load_raw_data_shapes(self):
        from src.preprocessing import load_raw_data
        raw = load_raw_data(DATA_DIR)
        assert raw["condition_frames"].shape == (1, 6, 128, 128)
        assert raw["observations"].shape == (1, 3, 128, 128)
        assert raw["observation_mask"].shape == (1, 1, 128, 128)

    def test_load_raw_data_dtypes(self):
        from src.preprocessing import load_raw_data
        raw = load_raw_data(DATA_DIR)
        for key in raw:
            assert raw[key].dtype == np.float32

    def test_load_ground_truth_shape(self):
        from src.preprocessing import load_ground_truth
        gt = load_ground_truth(DATA_DIR)
        assert gt.shape == (1, 3, 128, 128)
        assert gt.dtype == np.float32

    def test_load_meta_data_keys(self):
        from src.preprocessing import load_meta_data
        meta = load_meta_data(DATA_DIR)
        assert "image_height" in meta
        assert "image_width" in meta
        assert "observation_mask_ratio" in meta
        assert "noise_sigma" in meta
        assert meta["image_height"] == 128
        assert meta["image_width"] == 128

    def test_pixel_value_ranges(self):
        from src.preprocessing import load_raw_data, load_ground_truth
        raw = load_raw_data(DATA_DIR)
        gt = load_ground_truth(DATA_DIR)
        assert raw["condition_frames"].min() >= 0.0
        assert raw["condition_frames"].max() <= 1.0
        assert gt.min() >= 0.0
        assert gt.max() <= 1.0

    def test_mask_is_binary(self):
        from src.preprocessing import load_raw_data
        raw = load_raw_data(DATA_DIR)
        mask = raw["observation_mask"]
        unique_vals = np.unique(mask)
        assert set(unique_vals).issubset({0.0, 1.0})

    def test_mask_coverage(self):
        from src.preprocessing import load_raw_data
        raw = load_raw_data(DATA_DIR)
        mask = raw["observation_mask"]
        coverage = mask.mean()
        assert 0.05 < coverage < 0.20, f"Mask coverage {coverage:.3f} outside expected range"
