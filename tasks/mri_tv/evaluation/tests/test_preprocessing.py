"""Unit tests for preprocessing.py."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.preprocessing import load_observation, load_ground_truth, load_metadata, prepare_data

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "../fixtures/preprocessing")


class TestLoadObservation:
    def setup_method(self):
        self.obs = load_observation(DATA_DIR)
        self.fix = np.load(os.path.join(FIXTURE_DIR, "output_load_observation.npz"), allow_pickle=True)

    def test_keys(self):
        assert "masked_kspace" in self.obs
        assert "sensitivity_maps" in self.obs
        assert "undersampling_mask" in self.obs

    def test_masked_kspace_shape(self):
        expected = tuple(self.fix["masked_kspace_shape"])
        assert self.obs["masked_kspace"].shape == expected

    def test_sensitivity_maps_shape(self):
        expected = tuple(self.fix["sensitivity_maps_shape"])
        assert self.obs["sensitivity_maps"].shape == expected

    def test_undersampling_mask_shape(self):
        expected = tuple(self.fix["undersampling_mask_shape"])
        assert self.obs["undersampling_mask"].shape == expected

    def test_masked_kspace_dtype(self):
        assert self.obs["masked_kspace"].dtype == np.complex64

    def test_sensitivity_maps_dtype(self):
        assert self.obs["sensitivity_maps"].dtype == np.complex64

    def test_undersampling_mask_dtype(self):
        assert self.obs["undersampling_mask"].dtype == np.float32

    def test_mask_binary(self):
        mask = self.obs["undersampling_mask"]
        assert np.all((mask == 0) | (mask == 1))

    def test_n_sampled_lines(self):
        mask = self.obs["undersampling_mask"]
        expected = int(self.fix["n_sampled_lines"])
        assert int(mask.sum()) == expected


class TestLoadGroundTruth:
    def setup_method(self):
        self.gt = load_ground_truth(DATA_DIR)

    def test_shape(self):
        assert self.gt.shape == (1, 1, 320, 320)

    def test_dtype(self):
        assert self.gt.dtype == np.complex64

    def test_nonzero(self):
        assert np.any(self.gt != 0)

    def test_positive_magnitude(self):
        assert np.all(np.abs(self.gt) >= 0)


class TestLoadMetadata:
    def setup_method(self):
        self.meta = load_metadata(DATA_DIR)

    def test_required_keys(self):
        required = ["image_size", "n_coils", "acceleration_ratio"]
        for key in required:
            assert key in self.meta, f"Missing key: {key}"

    def test_image_size(self):
        assert self.meta["image_size"] == [320, 320]

    def test_n_coils(self):
        assert self.meta["n_coils"] == 15

    def test_acceleration_ratio(self):
        assert self.meta["acceleration_ratio"] == 8


class TestPrepareData:
    def setup_method(self):
        self.obs_data, self.ground_truth, self.metadata = prepare_data(DATA_DIR)

    def test_obs_data_keys(self):
        assert "masked_kspace" in self.obs_data
        assert "sensitivity_maps" in self.obs_data
        assert "undersampling_mask" in self.obs_data

    def test_ground_truth_shape(self):
        assert self.ground_truth.shape[0] == 1

    def test_metadata_type(self):
        assert isinstance(self.metadata, dict)

    def test_batch_consistency(self):
        n = self.obs_data["masked_kspace"].shape[0]
        assert self.obs_data["sensitivity_maps"].shape[0] == n
        assert self.ground_truth.shape[0] == n
