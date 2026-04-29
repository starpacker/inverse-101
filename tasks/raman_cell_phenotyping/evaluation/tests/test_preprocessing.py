"""Unit tests for preprocessing functions."""

import os
import sys
import numpy as np
import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.preprocessing import (
    crop, despike, denoise_savgol, normalise_minmax, preprocess_volume,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "../fixtures")


class TestCrop:
    def setup_method(self):
        f = np.load(f"{FIXTURE_DIR}/preprocessing_crop.npz")
        self.input_data = f["input_data"]
        self.input_axis = f["input_axis"]
        self.expected_data = f["output_data"]
        self.expected_axis = f["output_axis"]

    def test_shape(self):
        data, axis = crop(self.input_data, self.input_axis, (700, 1800))
        assert data.shape[-1] == axis.shape[0]

    def test_values(self):
        data, axis = crop(self.input_data, self.input_axis, (700, 1800))
        np.testing.assert_allclose(data, self.expected_data, rtol=1e-10)
        np.testing.assert_allclose(axis, self.expected_axis, rtol=1e-10)

    def test_axis_bounds(self):
        _, axis = crop(self.input_data, self.input_axis, (700, 1800))
        assert axis[0] >= 700
        assert axis[-1] <= 1800


class TestDespike:
    def setup_method(self):
        f = np.load(f"{FIXTURE_DIR}/preprocessing_despike.npz")
        self.input_data = f["input_data"]
        self.input_axis = f["input_axis"]
        self.expected = f["output_data"]

    def test_spike_removed(self):
        result, _ = despike(self.input_data[np.newaxis], self.input_axis)
        # The artificial spike at index 50 should be reduced
        assert result[0, 50] < self.input_data[50]

    def test_values(self):
        result, _ = despike(self.input_data[np.newaxis], self.input_axis)
        np.testing.assert_allclose(result[0], self.expected, rtol=1e-10)


class TestSavGol:
    def setup_method(self):
        f = np.load(f"{FIXTURE_DIR}/preprocessing_savgol.npz")
        self.input_data = f["input_data"]
        self.input_axis = f["input_axis"]
        self.expected = f["output_data"]

    def test_shape_preserved(self):
        result, _ = denoise_savgol(self.input_data[np.newaxis],
                                   self.input_axis)
        assert result.shape == (1,) + self.input_data.shape

    def test_values(self):
        result, _ = denoise_savgol(self.input_data[np.newaxis],
                                   self.input_axis)
        np.testing.assert_allclose(result[0], self.expected, rtol=1e-10)


class TestMinMax:
    def setup_method(self):
        f = np.load(f"{FIXTURE_DIR}/preprocessing_minmax.npz")
        self.input_data = f["input_data"]
        self.expected = f["output_data"]

    def test_range(self):
        result, _ = normalise_minmax(self.input_data,
                                     np.arange(self.input_data.shape[-1]),
                                     pixelwise=False)
        assert result.min() >= -1e-10
        assert result.max() <= 1 + 1e-10

    def test_values(self):
        result, _ = normalise_minmax(self.input_data,
                                     np.arange(self.input_data.shape[-1]),
                                     pixelwise=False)
        np.testing.assert_allclose(result, self.expected, rtol=1e-10)


class TestPreprocessVolume:
    def test_runs_on_real_data(self):
        from src.preprocessing import load_observation
        obs = load_observation("data")
        vol = obs["spectral_volume"][:4, :4, :2, :]  # small subset
        axis = obs["spectral_axis"]
        result, result_axis = preprocess_volume(vol, axis)
        assert result.ndim == 4
        assert result.shape[:3] == (4, 4, 2)
        assert result_axis.shape[0] == result.shape[-1]
