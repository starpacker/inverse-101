"""Unit tests for src/preprocessing.py."""
import os, sys, pytest, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.preprocessing import *
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")

class TestLoadData:
    def test_observation_keys(self):
        obs = load_observation(DATA_DIR)
        assert "kspace_real" in obs and "kspace_imag" in obs

    def test_kspace_shape(self):
        obs = load_observation(DATA_DIR)
        assert obs["kspace_real"].shape == (1, 15, 640, 368)

    def test_ground_truth_shape(self):
        assert load_ground_truth(DATA_DIR).shape == (1, 320, 320)

    def test_metadata(self):
        meta = load_metadata(DATA_DIR)
        assert meta["n_coils"] == 15
        assert meta["acceleration"] == 4

class TestApplyMask:
    def test_shapes(self):
        obs = load_observation(DATA_DIR)
        ks = get_complex_kspace(obs)
        masked, mask = apply_mask(ks[0], acceleration=4)
        assert masked.shape == (15, 640, 368, 2)
        assert mask.dtype == torch.bool

    def test_mask_reduces_data(self):
        obs = load_observation(DATA_DIR)
        ks = get_complex_kspace(obs)
        masked, mask = apply_mask(ks[0], acceleration=4)
        # ~25% of columns should be sampled
        frac = mask.float().mean().item()
        assert 0.15 < frac < 0.40

class TestPrepareData:
    def test_returns(self):
        ks, gt, meta = prepare_data(DATA_DIR)
        assert ks.shape[0] == 1
        assert gt.shape == (1, 320, 320)
