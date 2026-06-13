"""Unit tests for src/preprocessing.py."""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.preprocessing import (
    load_observation, load_ground_truth, load_metadata,
    get_complex_noise, get_mask, prepare_data,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")


class TestLoadObservation:
    def setup_method(self):
        self.obs = load_observation(DATA_DIR)

    def test_keys(self):
        expected = {"mask_random", "mask_radial", "mask_cartesian",
                    "noises_real", "noises_imag"}
        assert set(self.obs.keys()) == expected

    def test_mask_shapes(self):
        for key in ["mask_random", "mask_radial", "mask_cartesian"]:
            assert self.obs[key].shape == (1, 256, 256)

    def test_noise_shapes(self):
        assert self.obs["noises_real"].shape == (1, 256, 256)
        assert self.obs["noises_imag"].shape == (1, 256, 256)

    def test_masks_binary(self):
        for key in ["mask_random", "mask_radial", "mask_cartesian"]:
            vals = np.unique(self.obs[key])
            assert set(vals).issubset({0.0, 1.0})


class TestLoadGroundTruth:
    def test_shape(self):
        gt = load_ground_truth(DATA_DIR)
        assert gt.shape == (1, 256, 256)

    def test_range(self):
        gt = load_ground_truth(DATA_DIR)
        assert gt.min() >= 0.0
        assert gt.max() <= 1.0


class TestLoadMetadata:
    def test_required_keys(self):
        meta = load_metadata(DATA_DIR)
        assert "image_shape" in meta
        assert "noise_scale" in meta

    def test_shape(self):
        meta = load_metadata(DATA_DIR)
        assert meta["image_shape"] == [256, 256]


class TestGetComplexNoise:
    def test_shape(self):
        obs = load_observation(DATA_DIR)
        n = get_complex_noise(obs, scale=3.0)
        assert n.shape == (256, 256)
        assert np.iscomplexobj(n)


class TestGetMask:
    def test_all_masks(self):
        obs = load_observation(DATA_DIR)
        for name in ["random", "radial", "cartesian"]:
            m = get_mask(obs, name)
            assert m.shape == (256, 256)
            assert 0.25 < m.sum() / m.size < 0.35


class TestPrepareData:
    def test_returns_four(self):
        im, mask, noises, meta = prepare_data(DATA_DIR)
        assert im.shape == (256, 256)
        assert mask.shape == (256, 256)
        assert noises.shape == (256, 256)
        assert isinstance(meta, dict)
