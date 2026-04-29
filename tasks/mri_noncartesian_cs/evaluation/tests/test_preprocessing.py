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
        assert "kdata" in obs
        assert "coord" in obs
        assert "coil_maps" in obs

    def test_kdata_shape(self):
        obs = load_observation(DATA_DIR)
        assert obs["kdata"].ndim == 3  # (N, C, M)
        assert obs["kdata"].shape[0] == 1  # batch dim

    def test_coord_shape(self):
        obs = load_observation(DATA_DIR)
        assert obs["coord"].ndim == 3  # (N, M, 2)
        assert obs["coord"].shape[-1] == 2

    def test_coil_maps_shape(self):
        obs = load_observation(DATA_DIR)
        assert obs["coil_maps"].ndim == 4  # (N, C, H, W)
        assert obs["coil_maps"].shape[0] == 1

    def test_consistency(self):
        obs = load_observation(DATA_DIR)
        n_coils_kdata = obs["kdata"].shape[1]
        n_coils_maps = obs["coil_maps"].shape[1]
        assert n_coils_kdata == n_coils_maps


class TestLoadGroundTruth:
    def test_shape(self):
        gt = load_ground_truth(DATA_DIR)
        assert gt.ndim == 3  # (N, H, W)
        assert gt.shape[0] == 1

    def test_dtype(self):
        gt = load_ground_truth(DATA_DIR)
        assert np.iscomplexobj(gt)


class TestLoadMetadata:
    def test_keys(self):
        meta = load_metadata(DATA_DIR)
        assert "image_size" in meta
        assert "n_coils" in meta
        assert "n_spokes" in meta
        assert "n_readout" in meta

    def test_no_solver_params(self):
        """Metadata should not contain solver parameters."""
        meta = load_metadata(DATA_DIR)
        for key in ["lambda", "lamda", "max_iter", "learning_rate", "regularization"]:
            assert key not in meta, f"Solver parameter '{key}' found in meta_data.json"


class TestPrepareData:
    def test_returns_tuple(self):
        obs, gt, meta = prepare_data(DATA_DIR)
        assert isinstance(obs, dict)
        assert isinstance(gt, np.ndarray)
        assert isinstance(meta, dict)
