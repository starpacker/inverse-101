"""Unit tests for preprocessing.py."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.preprocessing import load_observation, load_ground_truth, load_metadata, prepare_data

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")


class TestLoadObservation:
    def test_keys(self):
        obs = load_observation(DATA_DIR)
        assert "masked_kspace" in obs
        assert "sensitivity_maps" in obs
        assert "undersampling_mask" in obs

    def test_shapes(self):
        obs = load_observation(DATA_DIR)
        assert obs["masked_kspace"].shape == (1, 8, 128, 128)
        assert obs["sensitivity_maps"].shape == (1, 8, 128, 128)
        assert obs["undersampling_mask"].shape == (128,)

    def test_dtypes(self):
        obs = load_observation(DATA_DIR)
        assert np.iscomplexobj(obs["masked_kspace"])
        assert np.iscomplexobj(obs["sensitivity_maps"])
        assert obs["undersampling_mask"].dtype == np.float32


class TestLoadGroundTruth:
    def test_shape(self):
        gt = load_ground_truth(DATA_DIR)
        assert gt.shape == (1, 1, 128, 128)

    def test_dtype(self):
        gt = load_ground_truth(DATA_DIR)
        assert np.iscomplexobj(gt)


class TestLoadMetadata:
    def test_keys(self):
        meta = load_metadata(DATA_DIR)
        assert meta["image_size"] == [128, 128]
        assert meta["n_coils"] == 8
        assert meta["acceleration_ratio"] == 8
        assert meta["n_samples"] == 1


class TestPrepareData:
    def test_returns_tuple(self):
        result = prepare_data(DATA_DIR)
        assert len(result) == 3

    def test_ground_truth_batch_first(self):
        _, gt, _ = prepare_data(DATA_DIR)
        assert gt.shape[0] == 1  # batch dimension
