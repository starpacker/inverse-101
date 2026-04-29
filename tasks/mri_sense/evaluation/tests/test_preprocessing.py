"""Unit tests for src/preprocessing.py."""
import os, sys, pytest, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.preprocessing import *
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")

class TestLoadData:
    def test_kspace_shape(self):
        obs = load_observation(DATA_DIR)
        assert obs["kspace_full_real"].shape == (1, 128, 128, 8)

    def test_ground_truth(self):
        gt = load_ground_truth(DATA_DIR)
        assert gt.shape == (1, 128, 128)
        assert gt.min() >= -1e-10

    def test_metadata(self):
        meta = load_metadata(DATA_DIR)
        assert meta["n_coils"] == 8
        assert meta["acceleration"] == 4

class TestUndersample:
    def test_shapes(self):
        obs = load_observation(DATA_DIR)
        kf = get_full_kspace(obs)
        kus, mask = undersample_kspace(kf, R=4, acs_width=16)
        assert kus.shape == kf.shape
        assert mask.shape == (128,)

    def test_zeros(self):
        obs = load_observation(DATA_DIR)
        kf = get_full_kspace(obs)
        kus, mask = undersample_kspace(kf, R=4, acs_width=16)
        assert np.all(kus[~mask, :, :] == 0)

    def test_sampling_rate(self):
        obs = load_observation(DATA_DIR)
        kf = get_full_kspace(obs)
        _, mask = undersample_kspace(kf, R=4, acs_width=16)
        rate = mask.sum() / len(mask)
        assert 0.25 < rate < 0.50

class TestPrepareData:
    def test_returns(self):
        kus, sens, kf, ph, meta = prepare_data(DATA_DIR)
        assert kus.shape == (128, 128, 8)
        assert sens.shape == (128, 128, 8)
        assert ph.shape == (128, 128)
