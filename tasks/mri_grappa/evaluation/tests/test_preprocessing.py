"""Unit tests for src/preprocessing.py."""
import os, sys, pytest, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.preprocessing import *

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")

class TestLoadObservation:
    def test_keys(self):
        obs = load_observation(DATA_DIR)
        for k in ["kspace_full_real", "kspace_full_imag", "sensitivity_maps_real", "sensitivity_maps_imag"]:
            assert k in obs

    def test_shapes(self):
        obs = load_observation(DATA_DIR)
        assert obs["kspace_full_real"].shape == (1, 128, 128, 8)

class TestLoadGroundTruth:
    def test_shape(self):
        assert load_ground_truth(DATA_DIR).shape == (1, 128, 128)

    def test_range(self):
        gt = load_ground_truth(DATA_DIR)
        assert gt.min() >= -1e-10 and gt.max() <= 1.0 + 1e-10

class TestUndersample:
    def test_shapes(self):
        obs = load_observation(DATA_DIR)
        kf = get_full_kspace(obs)
        kus, calib, mask = undersample_kspace(kf, R=2, acs_width=20)
        assert kus.shape == kf.shape
        assert calib.shape == (20, 128, 8)
        assert mask.shape == (128,)

    def test_zeros_in_undersampled(self):
        obs = load_observation(DATA_DIR)
        kf = get_full_kspace(obs)
        kus, _, mask = undersample_kspace(kf, R=2, acs_width=20)
        assert np.all(kus[~mask, :, :] == 0)

    def test_acs_preserved(self):
        obs = load_observation(DATA_DIR)
        kf = get_full_kspace(obs)
        kus, _, _ = undersample_kspace(kf, R=2, acs_width=20)
        ctr = 64
        np.testing.assert_array_equal(kus[54:74, :, :], kf[54:74, :, :])

class TestPrepareData:
    def test_returns(self):
        kus, calib, kf, ph, meta = prepare_data(DATA_DIR)
        assert kus.shape == (128, 128, 8)
        assert ph.shape == (128, 128)
        assert isinstance(meta, dict)
