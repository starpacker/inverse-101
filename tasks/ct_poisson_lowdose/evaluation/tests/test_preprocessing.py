"""Unit tests for the preprocessing module."""

import os
import sys

import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)

from src.preprocessing import (
    load_ground_truth,
    load_raw_data,
    load_metadata,
    sinogram_to_svmbir,
    weights_to_svmbir,
)

DATA_DIR = os.path.join(TASK_DIR, "data")


class TestLoadGroundTruth:
    def test_shape(self):
        phantom = load_ground_truth(DATA_DIR)
        assert phantom.shape == (256, 256)

    def test_nonnegative(self):
        phantom = load_ground_truth(DATA_DIR)
        assert np.all(phantom >= 0)

    def test_max_value(self):
        phantom = load_ground_truth(DATA_DIR)
        assert phantom.max() <= 0.05  # scaled Shepp-Logan


class TestLoadRawData:
    def test_keys(self):
        raw = load_raw_data(DATA_DIR)
        expected_keys = {
            "sinogram_clean", "sinogram_low_dose", "sinogram_high_dose",
            "weights_low_dose", "weights_high_dose", "angles",
        }
        assert set(raw.keys()) == expected_keys

    def test_sinogram_shapes(self):
        raw = load_raw_data(DATA_DIR)
        assert raw["sinogram_clean"].shape == (256, 367)
        assert raw["sinogram_low_dose"].shape == (256, 367)
        assert raw["sinogram_high_dose"].shape == (256, 367)

    def test_angles_shape(self):
        raw = load_raw_data(DATA_DIR)
        assert raw["angles"].shape == (256,)

    def test_weights_positive(self):
        raw = load_raw_data(DATA_DIR)
        assert np.all(raw["weights_low_dose"] >= 1.0)
        assert np.all(raw["weights_high_dose"] >= 1.0)


class TestLoadMetadata:
    def test_required_keys(self):
        meta = load_metadata(DATA_DIR)
        required = ["image_size", "num_views", "num_channels",
                     "geometry", "I0_low_dose", "I0_high_dose"]
        for key in required:
            assert key in meta, f"Missing key: {key}"

    def test_no_solver_params(self):
        """meta_data.json should not contain solver parameters."""
        meta = load_metadata(DATA_DIR)
        forbidden = ["snr_db", "sharpness", "max_iterations", "p", "q", "T",
                      "sigma_x", "sigma_y", "stop_threshold"]
        for key in forbidden:
            assert key not in meta, f"Solver param leaked: {key}"


class TestFormatConversion:
    def test_sinogram_to_svmbir(self):
        sino = np.zeros((256, 367))
        result = sinogram_to_svmbir(sino)
        assert result.shape == (256, 1, 367)

    def test_weights_to_svmbir(self):
        w = np.ones((256, 367))
        result = weights_to_svmbir(w)
        assert result.shape == (256, 1, 367)
